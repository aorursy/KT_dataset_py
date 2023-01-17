"""
I'll only be covering the detection of ships in this script, there won't be any segmentation stuff.

This will also not be an information session. The purpose is not to teach but to introduce a new tool
the learning will have to be done on the individual level.
"""
import os
import tqdm

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

import torch
import torchvision

INPUT_FOLDER = '../input'
TRAIN_FOLDER = os.path.join(INPUT_FOLDER, 'train')
TEST_FOLDER  = os.path.join(INPUT_FOLDER, 'test')
TRAIN_SEGMENTATION_FILE = os.path.join(INPUT_FOLDER, 'train_ship_segmentations.csv')

IMAGE_SHAPE = [768, 768]

BAD_IMAGES = ['6384c3e78.jpg']
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask_to_bbox(mask):
    img_h, img_w = mask.shape[:2]

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    x1 = int(max(cmin - 1, 0))
    y1 = int(max(rmin - 1, 0))
    x2 = int(min(cmax + 1, img_w))
    y2 = int(min(rmax + 1, img_h))

    return x1, y1, x2, y2
# First to perform some preprocessing, I suggest skipping this code block and move on to the next see what I'm after
train_df = pd.read_csv(TRAIN_SEGMENTATION_FILE)
train_df_values = train_df.values
train_image_to_ann = {}
for entry in tqdm.tqdm(train_df_values, 'Create image to ann table'):
    image_file = entry[0]
    segmentation = entry[1]
    
    if image_file in BAD_IMAGES or pd.isnull(segmentation):
        continue
        
    # Calc bbox
    mask = rle_decode(segmentation, IMAGE_SHAPE)
    bbox = mask_to_bbox(mask)
    ann = np.array(bbox + (0,))
    
    if image_file in train_image_to_ann:
        train_image_to_ann[image_file].extend([ann])
    else:
        train_image_to_ann[image_file] = [ann]

for image_file, anns in train_image_to_ann.items():
    train_image_to_ann[image_file] = np.array(anns, dtype='float32')

image_files = list(train_image_to_ann.keys())

# Clean up for freeing of memory in the future
del train_df
del train_df_values
# The data format I wanted to make is an image to bounding boxes dataset
# The labels for each image will look a little something like this
# Each row represents a bounding box around a ship
# and the values represent x1, y1, x2, y2, class_id
# If you don't know what those means, go google and unstuck yourself
train_image_to_ann[image_files[5]]
# To further visualize what I'm referring to, let's plot an image with the labels out
sample_img = Image.open(os.path.join(TRAIN_FOLDER, image_files[5]))
draw = ImageDraw.Draw(sample_img)
for ann in train_image_to_ann[image_files[5]]:
    draw.rectangle(ann[:4], outline=(255, 0, 0))
sample_img
# With this dataset, I'm going to make a dataloader
class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_to_ann):
        self.image_to_ann = image_to_ann
        self.image_files = list(image_to_ann.keys())
        self.image_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.image_to_ann)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(os.path.join(TRAIN_FOLDER, image_file))
        image = self.image_to_tensor(image)
        anns = self.image_to_ann[image_file]
        anns = torch.from_numpy(anns)
        return image, anns

dataset = DetectionDataset(train_image_to_ann)
dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)
for batch in dataset_loader:
    break

# We now see that batches of data produced by datadetloader produces batches of image, annotations pair
batch[0].shape, batch[1].shape
# Now to introduce something new, we first install a detection package
!rm -rf torch_collections/
!wget -q 'https://github.com/mingruimingrui/torch-collections/archive/0.4b.zip' -O torch-collections.zip
!unzip -oq torch-collections.zip
!mv torch-collections-0.4b/torch_collections .
!rm -rf torch-collections-0.4b/  torch-collections.zip
# And follow the package use case
from torch_collections import RetinaNet

# Build a model
model = RetinaNet(1).train().cuda()

# And an optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
# And we train the model
max_count = 6000
count = 0
pbar = tqdm.tqdm(total=max_count, desc='training model')
for batch in dataset_loader:
    pbar.update(1)
    optimizer.zero_grad()
    
    count
    
    batch_image = batch[0].cuda()
    batch_annotations = batch[1].cuda()
    
    loss = model(batch_image, batch_annotations)
    
    if loss is not None:
        # loss can be none when no valid anchors are available
        loss.backward()
        optimizer.step()
    
    del batch_image
    del batch_annotations
    del batch
    
    count += 1
    if count >= max_count:
        break
    
pbar.close()
# Now let's look at the results shall we?
model = model.eval()

viz = []
max_viz = 5
count_viz = 0

for image_file in os.listdir(TEST_FOLDER):
    image = Image.open(os.path.join(TEST_FOLDER, image_file))
    image_tensor = dataset.image_to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0).cuda()

    dets = model(image_tensor)[0]
    
    scores = dets['scores'].cpu().data.numpy()
    boxes = dets['boxes'].cpu().data.numpy()[scores > 0.7]
    
    if len(boxes) > 0:
        boxes = boxes.round()
        
        image = Image.open(os.path.join(TEST_FOLDER, image_file))
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(box[:4], outline=(255, 0, 0))
        
        count_viz += 1
        viz.extend([image])
    
    del image_tensor
    del dets
        
    if count_viz >= max_viz:
        break
viz[0]
viz[1]
viz[2]
viz[3]
viz[4]