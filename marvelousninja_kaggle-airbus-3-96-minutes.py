!pip3 install torch==0.4.1 albumentations

import glob
import math
from multiprocessing.pool import ThreadPool
import time
import numpy as np
import pandas as pd
import torch
import torchvision
from albumentations import (Compose, Normalize, Resize)
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

def get_images_in(path):
    return np.sort(glob.glob(f'{path}/*.jpg'))

def read_image(path):
    return np.array(Image.open(path))

def get_image_generator(image_paths, batch_size):
    batch = []
    for path in image_paths:
        batch.append(read_image(path))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0: yield batch

def collate(batch):
    if isinstance(batch[0], dict):
        return {key: collate([sample[key] for sample in batch]) for key in batch[0].keys()}
    return np.stack(batch)

def from_numpy(obj):
    if isinstance(obj, dict):
        return {key: from_numpy(value) for key, value in obj.items()}

    if torch.cuda.is_available():
        if isinstance(obj, torch.Tensor): return obj.float().cuda(non_blocking=True)
        return torch.cuda.FloatTensor(obj)
    else:
        if isinstance(obj, torch.Tensor): return obj.float()
        return torch.FloatTensor(obj)

def preprocess(pool, pipeline, batch):
    return from_numpy(collate(list(pool.map(pipeline, batch))))

def channels_first(image):
    return np.moveaxis(image, 2, 0)

class ChannelsFirst:
    def __call__(self, **args):
        args['image'] = channels_first(args['image'])
        return args

def classifier_pipeline(image):
    return Compose([
        Resize(224, 224),
        ChannelsFirst()
    ])(image=image)

def segmenter_pipeline(image):
    return Compose([
        ChannelsFirst()
    ])(image=image)

def as_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def extract_instance_masks_from_binary_mask(args):
    _id, binary_mask = args
    masks = []
    labelled_mask = ndimage.label(binary_mask)[0]
    labels, areas = np.unique(labelled_mask, return_counts=True)
    labels = labels[areas >= 80]
    for label in labels:
        if label == 0: continue
        masks.append((_id, labelled_mask == label))
    if len(masks) < 1: return [(_id, None)]
    return masks

def encode_rle(args):
    _id, mask = args
    if mask is None: return (_id, None)
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return (_id, ' '.join(str(x) for x in runs))

def postprocess_segmentation(pool, ids, binary_masks):
    ids_and_instance_masks = map(extract_instance_masks_from_binary_mask, zip(ids, binary_masks))
    return map(encode_rle, sum(ids_and_instance_masks, []))

class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels // 4, (1, 1)),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (3, 3), stride=stride, padding=1, output_padding=output_padding),
            torch.nn.BatchNorm2d(in_channels // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels // 4, out_channels, (1, 1)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Segmenter(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.decoder1 = Decoder(512, 256, stride=2, output_padding=1)
        self.decoder2 = Decoder(256, 128, stride=2, output_padding=1)
        self.decoder3 = Decoder(128, 64, stride=2, output_padding=1)
        self.decoder4 = Decoder(64, 64, stride=1, output_padding=0)
        self.classifier = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, num_classes, (2, 2), stride=2)
        )

    def forward(self, x):
        x = x['image']
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        x = self.decoder1(x4) + x3
        x = self.decoder2(x) + x2
        x = self.decoder3(x) + x1
        x = self.decoder4(x)
        return {'mask': self.classifier(x)}

class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = x['image']
        return {'has_ships': self.resnet(x)}

classifier_path = '/kaggle/input/airbus-model-weights/classifier_weights.pt'
segmenter_path = '/kaggle/input/airbus-model-weights/segmenter_weights.pt'
output_path = '/kaggle/working/submission.csv'
directory_path = '/kaggle/input/airbus-ship-detection/test_v2'
classifier_batch_size = 16
segmenter_batch_size = 16

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)

classifier = Classifier(1).cuda()
classifier.load_state_dict(torch.load(classifier_path))
classifier.eval()

segmenter = Segmenter(1).cuda()
segmenter.load_state_dict(torch.load(segmenter_path))
segmenter.eval()

# Warmup
for i in range(10):
    batch = torch.randn(classifier_batch_size, 3, 224, 224).cuda()
    classifier({'image': batch})

for i in range(10):
    batch = torch.randn(segmenter_batch_size, 3, 768, 768).cuda()
    segmenter({'image': batch})
    
image_paths = get_images_in(directory_path)

total_inference_time = 0

means = from_numpy(np.array([0.485, 0.456, 0.406])[None, :, None, None])
stds = from_numpy(np.array([0.229, 0.224, 0.225])[None, :, None, None])

pool = ThreadPool(2)

# Classification
pred_labels = []
for batch in tqdm(get_image_generator(image_paths, classifier_batch_size), total=math.ceil(len(image_paths) / classifier_batch_size)):
    time_start = time.time()
    batch = preprocess(pool, classifier_pipeline, batch)
    batch['image'] = (batch['image'] / 255 - means) / stds
    pred_labels.extend((classifier(from_numpy(batch))['has_ships'] > 0)[:, 0])
    torch.cuda.synchronize()
    total_inference_time += (time.time() - time_start)

time_start = time.time()
pred_labels = np.array(pred_labels)
positive_image_paths = image_paths[pred_labels == 1]
negative_image_paths = image_paths[pred_labels == 0]
total_inference_time += (time.time() - time_start)

# Segmentation
time_start = time.time()
records = []
remaining_ids = list(map(lambda path: path.split('/')[-1], positive_image_paths))
total_inference_time += (time.time() - time_start)
for batch in tqdm(get_image_generator(positive_image_paths, segmenter_batch_size), total=math.ceil(len(positive_image_paths) / segmenter_batch_size)):
    time_start = time.time()
    batch = preprocess(pool, segmenter_pipeline, batch)
    batch['image'] = (batch['image'] / 255 - means) / stds
    binary_masks = (segmenter(batch)['mask'] > 0)[:, 0]
    records.extend(postprocess_segmentation(pool, remaining_ids[:len(binary_masks)], binary_masks))
    remaining_ids = remaining_ids[len(binary_masks):]
    torch.cuda.synchronize()
    total_inference_time += (time.time() - time_start)

time_start = time.time()
negative_ids = list(map(lambda path: path.split('/')[-1], negative_image_paths))
records.extend(map(lambda _id: (_id, None), negative_ids))

image_ids, encoded_pixels = zip(*records)
df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
df.to_csv(output_path, index=False)
total_inference_time += (time.time() - time_start)
print('Inference Time: %0.2f Minutes'%((total_inference_time)/60))
