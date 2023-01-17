import os
import torch
import torch.nn as nn
import pandas as pd


from PIL import Image
import numpy as np
import cv2
import gc
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam,Adagrad,SGD
import torch.nn.functional as F

import random

from torch.utils.data import Dataset, DataLoader, Subset

from torch.utils import data
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Sequential, Conv2d, AvgPool2d, GRU, Linear
from torch.nn.functional import ctc_loss, log_softmax
from torchvision import models


import torchvision
import pickle

import json
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import *

from itertools import chain
from pandas.io.json import json_normalize

import torch.distributed as dist

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.patches as patches


import os
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt



from string import digits, ascii_uppercase

#import utils
import math 

# сид

SEED = 1489


random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
# определение девайса

device = "cuda" if torch.cuda.is_available() else "cpu"
TEST_PATH = "./data/test/" 
TRAIN_PATH = "./data/train/"
SUBMISSION_PATH = "./data/submission.csv"
TRAIN_INFO = "./data/train.json"


IMAGE_WIDTH = 412
IMAGE_HEIGHT = 412


VAL_SIZE = 0.3


N_ITER = 2
BATCH_SIZE = 32

BATCH_SIZE_VAL = 8
LR = 3e-5


COOR_COUNT = 4


EXP_NAME = "resnet34"
file_object = open(TRAIN_INFO, "r")

train = json_normalize(json.load(file_object))

test = pd.read_csv(SUBMISSION_PATH)
train['box'] = train.nums.apply(lambda x: list(chain.from_iterable(x[0]['box'])))
train['text'] = train.nums.apply(lambda x: x[0]['text'])
train['file'] = train.file.apply(lambda x: x.split("/")[1])

train.drop(['nums'], axis =1, inplace = True)

test['text'] = test.file_name.apply(lambda x: x.split("/")[1])

test.drop(['file_name'], axis =1, inplace = True)
train = train.head(25631)
train.head()
train.shape
train.shape
img_names = train.file.values
shapes = [cv2.imread(os.path.join(TRAIN_PATH, name), 0).shape for name in img_names]

train['shapes'] = shapes
def replace(x):
    for i in range(len(x['box'])):
        if i % 2 == 0:
            x['box'][i] = (float(x['box'][i])/x['shapes'][1]) *  IMAGE_WIDTH
        else:
            x['box'][i] = (float(x['box'][i])/x['shapes'][0]) * IMAGE_HEIGHT

    return x

train = train.apply(lambda x: replace(x), axis = 1)
train['xmin'] = train.apply(lambda x: min(x['box'][0], int(x['box'][6])), axis=1)
train['xmax'] = train.apply(lambda x: max(x['box'][2], int(x['box'][4])), axis=1) 
train['ymin'] = train.apply(lambda x: min(x['box'][1], int(x['box'][3])), axis=1) 
train['ymax'] = train.apply(lambda x: max(x['box'][5], int(x['box'][7])), axis=1)
train[((train.ymax <= train.ymin + 5) | (train.xmax <= train.xmin + 5))]

train = train[~((train.ymax <= train.ymin + 5) | (train.xmax <= train.xmin + 5))]
train = train[~((train.xmin < 0 )| (train.ymin < 0))]
train.head()
test.head()
valid_images = np.random.choice(train.file.unique(), size=int(VAL_SIZE * train.file.nunique()), replace=False)
valid_set = train[train.file.isin(valid_images)]

train_set = train[~train.file.isin(valid_images)]
print(valid_set.shape, train_set.shape)
test_ids = test.text
def collate_fn(batch):
    return tuple(zip(*batch))

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class ShapeDataset(Dataset):

    def __init__(self, IMAGE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, data,
                 transform=transforms.Compose([ToTensor(), 
                                               Normalize(
                                                   mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]
                                               )])):
        self.IMAGE_DIR = IMAGE_DIR
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        
        
        self.data = data
        self.transform = transform



    def num_classes(self):
        return len(self.class2index)

    
    def __len__(self, ):
        return len(self.data)
    
    def __getitem__(self, idx):
        
              
        def get_boxes(obj):
            boxes = [[obj[f] for f in ['xmin', 'ymin', 'xmax', 'ymax'] ]]
            return torch.as_tensor(boxes, dtype=torch.float)


        def get_areas(obj):
            areas = [(obj['xmax'] - obj['xmin']) * (obj['ymax'] - obj['ymin']) ]
            return torch.as_tensor(areas, dtype=torch.int64)

        img_name = self.data.iloc[idx]['file']
        
        path = os.path.join(self.IMAGE_DIR, img_name)

        img = cv2.imread(path)
        
        shapes  = img.shape
        

        
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
        


        
        
        img_bbox = self.data.iloc[idx]['box']#.copy()
        
        #print(img_bbox, shapes, img_name)

        obj = {}        
        
        obj['xmin'] = np.min([int(img_bbox[0]), int(img_bbox[6])])
        obj['xmax'] = np.max([int(img_bbox[2]), int(img_bbox[4])])
        obj['ymin'] = np.min([int(img_bbox[1]), int(img_bbox[3])])
        obj['ymax'] = np.max([int(img_bbox[5]), int(img_bbox[7])])

        if self.transform:
            image = self.transform(img)
            
        print(img_name, obj)
        
        target = {}
        target['boxes'] = get_boxes(obj)

        target['labels'] = torch.ones((1,), dtype=torch.int64)#torch.as_tensor(1, dtype=torch.int64)
        target['image_id'] = torch.as_tensor([idx], dtype=torch.int64)
        target['area'] = get_areas(obj)
        target['iscrowd'] = torch.ones((1,), dtype=torch.int64)#get_iscrowds(annot)

        return image, target




    
class ShapeDatasetTest(Dataset):

    def __init__(self, IMAGE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, data,
                 transform=transforms.Compose([ToTensor(),
                                               Normalize(
                                                   mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]
                                               ) ])):
        self.IMAGE_DIR = IMAGE_DIR
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        
        
        self.data = data
        self.transform = transform


    def num_classes(self):
        return len(self.class2index)

    
    def __len__(self, ):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        
        img_name = self.data.iloc[idx]['text']
        
        path = os.path.join(self.IMAGE_DIR, img_name)

        img = cv2.imread(path)
        
        shapes  = img.shape

        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
        
        
        if self.transform:
            image = self.transform(img)
        return image 



train_data = ShapeDataset(TRAIN_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, train_set)
valid_data = ShapeDataset(TRAIN_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, valid_set)



test_data  = ShapeDatasetTest(TEST_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, test)
dataloader_train = DataLoader(
    train_data, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
dataloader_valid = DataLoader(
    valid_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)



import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = get_device()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model = model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, amsgrad=True)
loss_fn = fnn.mse_loss


lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)
train_losses = []
test_losses = []
# best_val_loss = 1
# with open(f"model_{best_val_loss}.pth", "rb") as fp:
#     best_state_dict = torch.load(fp, map_location="cpu")
#     model.load_state_dict(best_state_dict)

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            #sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
num_epochs = 2

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()

    with open(f"model_{epoch}.pth", "wb") as fp:
        torch.save(model.state_dict(), fp)
    #evaluate(model, dataloader_valid, device=device)





gc.collect()
def get_rects(boxes):
    rect = lambda x, y, w, h: patches.Rectangle((x, y), w - x, h - y, linewidth=1, edgecolor='r', facecolor='none')

    return [rect(box[0], box[1], box[2], box[3]) for box in boxes]

def get_clazzes(labels, boxes, index2class):
    return [{'x': box[0].item(), 'y': box[1].item() - 5.0, 's': index2class[label.item()], 'fontsize': 10}
            for label, box in zip(labels, boxes)]

def show_prediction(img, index2class, fig, ax):
    pil_image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    ax.imshow(pil_image)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for rect in get_rects(prediction[0]['boxes']):
        ax.add_patch(rect)


    for label in get_clazzes(prediction[0]['labels'], prediction[0]['boxes'], index2class):
        ax.text(**label)
        
        
def get_prediction(dataset,  model):
  #  img, _ = dataset[idx]

    model.eval()
    cpu_device = torch.device("cpu")

    
    preds = []
    for images in metric_logger.log_every(dataset, 100, header):

        images = torch.stack([images[0][0].to(device), images[1][0].to(device), images[2][0].to(device) ], dim=0).unsqueeze(0)

        torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        preds.append(outputs)

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        
    return preds#img, prediction

import utils
metric_logger = utils.MetricLogger(delimiter="  ")

header = "Test:"
dataloader_test = DataLoader(
    test_data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


predictions = get_prediction(dataloader_test, model) 


len(predictions)

ids =[]
boxes = []
test_ids = test.text
for index, i in enumerate(predictions):
    for j in i:
        for k in j['boxes'].cpu().detach().numpy():
            boxes.append(k)
            ids.append(test_ids[index])
print(len(boxes), len(ids))
test.head()
d = {'file': ids, 'text': boxes}

test = pd.DataFrame(data=d)
test.head()
img_names = test.file.values
shapes = [cv2.imread(os.path.join(TEST_PATH, name), 0).shape for name in img_names]

test['shapes']  = shapes
test.head()
def restore(x, col_name):
    
    box = x[col_name].copy()
    #print(type(box[0]), IMAGE_WIDTH)
    box[0] = int((box[0]/IMAGE_WIDTH) * x['shapes'][1])
    box[1] = int((box[1]/IMAGE_HEIGHT) * x['shapes'][0])
    box[2] = int((box[2]/IMAGE_WIDTH) * x['shapes'][1])
    box[3] = int((box[3]/IMAGE_HEIGHT) * x['shapes'][0])
    
    return box


test['fixed_text'] = test.apply(lambda x: restore(x, 'text'), axis = 1)
test['xmin'] = test.fixed_text.apply(lambda x: x[0])
test['ymin'] = test.fixed_text.apply(lambda x: x[1])
test['xmax'] = test.fixed_text.apply(lambda x: x[2])
test['ymax'] = test.fixed_text.apply(lambda x: x[3])
test.to_csv("test_first.csv", index= None)
test = pd.read_csv("test_first.csv")
test.head()
file_object = open(TRAIN_INFO, "r")

train = json_normalize(json.load(file_object))
train['box'] = train.nums.apply(lambda x: list(chain.from_iterable(x[0]['box'])))
train['text'] = train.nums.apply(lambda x: x[0]['text'])
train['file'] = train.file.apply(lambda x: x.split("/")[1])

train.drop(['nums'], axis =1, inplace = True)

train
train = train.head(25631)


img_names = train.file.values
shapes = [cv2.imread(os.path.join(TRAIN_PATH, name), 0).shape for name in img_names]

train['shapes'] = shapes
train['xmin'] = train.apply(lambda x: min(x['box'][0], int(x['box'][6])), axis=1)
train['xmax'] = train.apply(lambda x: max(x['box'][2], int(x['box'][4])), axis=1) 
train['ymin'] = train.apply(lambda x: min(x['box'][1], int(x['box'][3])), axis=1) 
train['ymax'] = train.apply(lambda x: max(x['box'][5], int(x['box'][7])), axis=1)
train[((train.ymax <= train.ymin + 5) | (train.xmax <= train.xmin + 5))]

train = train[~((train.ymax <= train.ymin + 5) | (train.xmax <= train.xmin + 5))]
train = train[~((train.xmin < 0 )| (train.ymin < 0))]
train.head()
#train['fixed_text'] = train.apply(lambda x: restore(x,  'box'), axis = 1)

train = train[~((train.xmin < 0 )| (train.ymin < 0))]
abc = "0123456789ABEKMHOPCTYX" 
def compute_mask(text):
    """Compute letter-digit mask of text.
    Accepts string of text. 
    Returns string of the same length but with every letter replaced by 'L' and every digit replaced by 'D'.
    e.g. 'E506EC152' -> 'LDDDLLDDD'.
    Returns None if non-letter and non-digit character met in text.
    """
    # YOUR CODE HERE
    mask = []
    for char in text:
        if char in digits:
            mask.append("D")
        elif char in ascii_uppercase:
            mask.append("L")
        else:
            return None
    return "".join(mask)

def check_in_alphabet(text, alphabet=abc):
    """Check if all chars in text come from alphabet.
    Accepts string of text and string of alphabet. 
    Returns True if all chars in text are from alphabet and False else.
    """
    # YOUR CODE HERE
    for char in text:
        if char not in alphabet:
            return False
    return True

def filter_data(config):
    """Filter config keeping only items with correct text.
    Accepts list of items.
    Returns new list.
    """
    config_filtered = []
    for item in tqdm.tqdm(config):
        text = item["text"]
        mask = compute_mask(text)
        if check_in_alphabet(text) and (mask == "LDDDLLDD" or mask == "LDDDLLDDD"):
            config_filtered.append({"file": item["file"],
                                    "text": item["text"]})
    return config_filtered
class RecognitionDataset(Dataset):
    """Class for training image-to-text mapping using CTC-Loss."""

    def __init__(self, df, alphabet=abc, transforms=None):
        """Constructor for class.
        Accepts:
        - config: list of items, each of which is a dict with keys "file" & "text".
        - alphabet: string of chars required for predicting.
        - transforms: transformation for items, should accept and return dict with keys "image", "seq", "seq_len" & "text".
        """
        super(RecognitionDataset, self).__init__()
        self.df = df
        self.alphabet = abc
        self.image_names, self.texts = self._parse_root_()
        self.transforms = transforms

    def _parse_root_(self):
        image_names, texts = [], []
        #for item in self.train:
        for index, item in self.df.iterrows():
            #print(config)
            
            #print(item)
            image_name = item["file"]
            text = item['text']
            texts.append(text)
            image_names.append(image_name)
        return image_names, texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Return dict with keys "image", "seq", "seq_len" & "text".
        Image is a numpy array, float32, [0, 1].
        Seq is list of integers.
        Seq_len is an integer.
        Text is a string.
        """

        row = self.df[self.df.file == self.image_names[item]]

        image = cv2.imread(os.path.join(TRAIN_PATH, self.image_names[item]))#[row['ymin']:row['ymax'], row['xmin']: row['xmax']].astype(np.float32) / 255.
        #print(row['shapes'], int(row['ymin'].values[0]),int(row['ymax'].values[0]), int(row['xmin'].values[0]), int(row['xmax'].values[0]))
        image = image[int(row['ymin'].values[0]):int(row['ymax'].values[0]), int(row['xmin'].values[0]): int(row['xmax'].values[0]) ]
        iamge = image.astype(np.float32) / 255.

        text = self.texts[item]
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        if self.transforms is not None:
            output = self.transforms(output)
        return output

    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        Accepts string of text.
        Returns list of integers where each number is index of corresponding characted in alphabet + 1.
        """
        # YOUR CODE HERE
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq
class RecognitionDatasetTest(Dataset):
    """Class for training image-to-text mapping using CTC-Loss."""

    def __init__(self, test, alphabet=abc, transforms=None):
        """Constructor for class.
        Accepts:
        - config: list of items, each of which is a dict with keys "file" & "text".
        - alphabet: string of chars required for predicting.
        - transforms: transformation for items, should accept and return dict with keys "image", "seq", "seq_len" & "text".
        """
        super(RecognitionDatasetTest, self).__init__()
        self.test = test
        self.alphabet = abc
        self.image_names, self.texts = self._parse_root_()
        self.transforms = transforms

    def _parse_root_(self):
        image_names, texts = [], []
        #for item in self.train:
        for index, item in test.iterrows():

            image_name = item["file"]
            text = item['text']
            texts.append(text)
            image_names.append(image_name)
        return image_names, texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        """Return dict with keys "image", "seq", "seq_len" & "text".
        Image is a numpy array, float32, [0, 1].
        Seq is list of integers.
        Seq_len is an integer.
        Text is a string.
        """
        

        row = test.loc[item]
        

        image = cv2.imread(os.path.join(TEST_PATH, self.image_names[item]))#[row['ymin']:row['ymax'], row['xmin']: row['xmax']].astype(np.float32) / 255.
        image = image[int(row['ymin']):int(row['ymax']), int(row['xmin']): int(row['xmax']) ]
        iamge = image.astype(np.float32) / 255.

        text = self.texts[item]
        seq = self.text_to_seq(text)
        seq_len = len(seq)
        output = dict(image=image, seq=seq, seq_len=seq_len, text=text)
        
        if self.transforms is not None:
            output = self.transforms(output)
        return output

    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        Accepts string of text.
        Returns list of integers where each number is index of corresponding characted in alphabet + 1.
        """
        # YOUR CODE HERE
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq
class Resize(object):

    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        """Accepts item with keys "image", "seq", "seq_len", "text".
        Returns item with image resized to self.size.
        """
        # YOUR CODE HERE
        item['image'] = cv2.resize(item['image'], self.size, interpolation=cv2.INTER_AREA)
        return item
transforms = Resize(size=(320, 64))
dataset = RecognitionDataset(train, alphabet=abc, transforms=transforms)
def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched images, sequences, and so.
    """
    images, seqs, seq_lens, texts = [], [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample["image"]).permute(2, 0, 1).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
    return batch

def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched images, sequences, and so.
    """
    images, seqs, seq_lens, texts = [], [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample["image"]).permute(2, 0, 1).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts}
    return batch

class FeatureExtractor(Module):
    
    def __init__(self, input_size=(64, 320), output_len=20):
        super(FeatureExtractor, self).__init__()
        
        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = Sequential(*list(resnet.children())[:-2])
        
        self.pool = AvgPool2d(kernel_size=(h // 32, 1))        
        self.proj = Conv2d(w // 32, output_len, kernel_size=1)
  
        self.num_output_features = self.cnn[-1][-1].bn2.num_features    
    
    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W').
        """
        # YOUR CODE HERE
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
   
    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)
        
        # Pool to make height == 1
        features = self.pool(features)
        
        # Apply projection to increase width
        features = self.apply_projection(features)
        
        return features
feature_extractor = FeatureExtractor()
x = torch.randn(1, 3, 64, 320)
y = feature_extractor(x)
class SequencePredictor(Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SequencePredictor, self).__init__()
        
        self.num_classes = num_classes        
        self.rnn = GRU(input_size=input_size,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       dropout=dropout,
                       bidirectional=bidirectional)
        
        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = Linear(in_features=fc_in,
                         out_features=num_classes)
    
    def _init_hidden_(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.
        Accepts batch size.
        Returns tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        # YOUR CODE HERE
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        
    def _prepare_features_(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C=1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        # YOUR CODE HERE
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x
    
    def forward(self, x):
        x = self._prepare_features_(x)
        
        batch_size = x.size(1)
        h_0 = self._init_hidden_(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)
        
        x = self.fc(x)
        return x
sequence_predictor = SequencePredictor(input_size=512, 
                                       hidden_size=128, 
                                       num_layers=2, 
                                       num_classes=len(abc) + 1)
x = torch.randn(1, 1, 512, 20)
y = sequence_predictor(x)
class CRNN(Module):
    
    def __init__(self, alphabet=abc,
                 cnn_input_size=(64, 320), cnn_output_len=20,
                 rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.alphabet = alphabet
        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)
        self.sequence_predictor = SequencePredictor(input_size=self.features_extractor.num_output_features,
                                                    hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                                    num_classes=len(alphabet)+1, dropout=rnn_dropout,
                                                    bidirectional=rnn_bidirectional)
    
    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence
def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out

def decode(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs
crnn = CRNN()
x = torch.randn(1, 3, 64, 320)
y = crnn(x)
decode(y, abc)
ACTUALLY_TRAIN = True
crnn = CRNN()
num_epochs =  8
batch_size = 128
num_workers = 0
device = torch.device("cuda: 0") if torch.cuda.is_available() else torch.device("cpu")
crnn.to(device);
optimizer = torch.optim.Adam(crnn.parameters(), lr=3e-4, amsgrad=True, weight_decay=1e-4)

train_size = int(len(train) * 0.8)
config_train = train[:train_size]
config_val = train[train_size:]

transforms = Resize(size=(320, 64))


config_val.head()
train_dataset = RecognitionDataset(config_train, alphabet=abc, transforms=Resize())
val_dataset = RecognitionDataset(config_val, alphabet=abc, transforms=Resize())
train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, 
                              drop_last=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, 
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, 
                            drop_last=False, collate_fn=collate_fn)
crnn.train()
if ACTUALLY_TRAIN:
    for i, epoch in enumerate(range(num_epochs)):
        epoch_losses = []

        for j, b in enumerate(tqdm.tqdm(train_dataloader, total=len(train_dataloader))):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            seqs_pred = crnn(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = ctc_loss(log_probs=log_probs,  # (T, N, C)
                            targets=seqs_gt,  # N, S or sum(target_lengths)
                            input_lengths=seq_lens_pred,  # N
                            target_lengths=seq_lens_gt)  # N

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        print(i, np.mean(epoch_losses))

        val_losses = []
        for i, b in enumerate(tqdm.tqdm(val_dataloader, total=len(val_dataloader))):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]

            with torch.no_grad():
                seqs_pred = crnn(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()
            loss = ctc_loss(log_probs=log_probs,  # (T, N, C)
                            targets=seqs_gt,  # N, S or sum(target_lengths)
                            input_lengths=seq_lens_pred,  # N
                            target_lengths=seq_lens_gt)  # N

            val_losses.append(loss.item())

        print(np.mean(val_losses))
else:
    image_train_log = cv2.imread("./resources/train_log.png")
    plt.figure(figsize=(15, 20))
    plt.imshow(image_train_log[:, :, ::-1], interpolation="bilinear")
    plt.axis("off")
    plt.show()
crnn.eval()
if ACTUALLY_TRAIN:
    val_losses = []
    for i, b in enumerate(tqdm.tqdm(val_dataloader, total=len(val_dataloader))):
        images = b["image"].to(device)
        seqs_gt = b["seq"]
        seq_lens_gt = b["seq_len"]

        with torch.no_grad():
            seqs_pred = crnn(images).cpu()
        log_probs = log_softmax(seqs_pred, dim=2)
        seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()
        loss = ctc_loss(log_probs=log_probs,  # (T, N, C)
                        targets=seqs_gt,  # N, S or sum(target_lengths)
                        input_lengths=seq_lens_pred,  # N
                        target_lengths=seq_lens_gt)  # N

        val_losses.append(loss.item())

    print(np.mean(val_losses))
else:
    image_val_log = cv2.imread("./resources/val_log.png")
    plt.figure(figsize=(15, 20))
    plt.imshow(image_val_log[:, :, ::-1], interpolation="bilinear")
    plt.axis("off")
    plt.show()
# test['xmin'] = test.fixed_text.apply(lambda x: x[0])
# test['ymin'] = test.fixed_text.apply(lambda x: x[1])
# test['xmax'] = test.fixed_text.apply(lambda x: x[2])
# test['ymax'] = test.fixed_text.apply(lambda x: x[3])
test['text'] = "A232BC41"
test = test[~((test.ymin == test.ymax) | (test.xmin == test.xmax) )].reset_index()
test.head()
dataset_test = RecognitionDatasetTest(test, alphabet=abc, transforms=transforms)
test_dataloader = DataLoader(dataset_test , 
                              batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, 
                              drop_last=False, collate_fn=collate_fn)
numbers = {}

for i, b in enumerate(tqdm.tqdm(test_dataloader, total=len(test_dataloader))):
    #print(b)
    images = b["image"].to(device)
    seqs_gt = b["seq"]
    seq_lens_gt = b["seq_len"]

    with torch.no_grad():
        seqs_pred = crnn(images).cpu()
    log_probs = log_softmax(seqs_pred, dim=2)
    seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()
    img_name = test[test.index == i]['file'].values[0]
    if (img_name not in numbers.keys()):

        #numbers[img_name] =  str(decode(log_probs, crnn.alphabet)[0]) +" " +  numbers[img_name]
        numbers[img_name] = str(decode(log_probs, crnn.alphabet)[0])
#     else:
#         numbers[img_name] = str(decode(log_probs, crnn.alphabet)[0])
    print(img_name, test[test.index == i]['file'].values[0], decode(log_probs, crnn.alphabet))


test.head()
ids_final = []
num_final = []

for i in numbers.keys():
    
    ids_final.append("test/" + i)
    num_final.append(numbers[i])
d = {'file_name': ids_final, 'plates_string':num_final}


output = pd.DataFrame(data=d)

sub = pd.read_csv("submission.csv")
final = pd.merge(sub.drop(['plates_string'], axis =1 ) , output, how = 'left', left_on = "file_name", right_on = "file_name" )
final.to_csv("output.csv", index = None)
