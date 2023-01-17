# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from pathlib import Path

#import seaborn as sns



import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle



from PIL import Image, ImageDraw, ImageFont

#import cv2



import regex as re

import math

import random



from itertools import compress



import torch

import torch.distributed as dist

import torch.nn.functional as F

from torch import nn

from torch.utils.data import DataLoader

from torch.utils.data import Dataset



import torchvision

import torchvision.transforms as transforms

from tqdm import tnrange, tqdm_notebook



from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.rpn import RPNHead



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



path = Path("/kaggle/input/kuzushiji-recognition")

#path = Path("kuzushiji-recognition")



train = pd.read_csv(path/"train.csv")

unicode = pd.read_csv(path/"unicode_translation.csv")

submission = pd.read_csv(path/"sample_submission.csv")



train = train.dropna(axis = 0, how ='any') 



train_dir = Path(path/"train_images")

test_dir = Path(path/"test_images")



def toPath(string):

    if ".jpg" not in string:

        string = string + ".jpg"

    return string



def toID(string):

    if string[-4:] ==".jpg":

        string = string[:-4]

    return string



def displayImage(image):

    plt.figure(figsize=(15,15))

    this_img = Image.open(train_dir/toPath(image))

    plt.imshow(this_img)

    return plt



def getImageArray(image):

    this_img = Image.open(train_dir/toPath(image))

    return plt



def drawBoxAndText(ax, label):

    codepoint, x, y, w, h = label

    x, y, w, h = int(x), int(y), int(w), int(h)

    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")

    ax.add_patch(rect)

    ax.text(x, y - 20, getUnicode(codepoint),

            fontproperties=prop,

            color="r",

           size=16)

    return ax



def collate_fn(batch):

    return tuple(zip(*batch))



def getUnicode(code):

    char = unicode.loc[unicode["Unicode"] == code]["char"].values

    if len(char) > 0:

        return char[0]

    else:

        return '0'

# Any results you write to the current directory are saved as output.



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#device = torch.device('cpu')
def is_dist_avail_and_initialized():

    if not dist.is_available():

        return False

    if not dist.is_initialized():

        return False

    return True





def get_world_size():

    if not is_dist_avail_and_initialized():

        return 1

    return dist.get_world_size()



def reduce_dict(input_dict, average=True):

    """

    Args:

        input_dict (dict): all the values will be reduced

        average (bool): whether to do average or sum

    Reduce the values in the dictionary from all processes so that all processes

    have the averaged results. Returns a dict with the same fields as

    input_dict, after reduction.

    """

    world_size = get_world_size()

    if world_size < 2:

        return input_dict

    with torch.no_grad():

        names = []

        values = []

        # sort the keys so that they are consistent across processes

        for k in sorted(input_dict.keys()):

            names.append(k)

            values.append(input_dict[k])

        values = torch.stack(values, dim=0)

        dist.all_reduce(values)

        if average:

            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict
# download a font that can display the characters

import matplotlib.font_manager as font_manager



fontsize = 50



# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip



path = './NotoSansCJKjp-Regular.otf'

prop = font_manager.FontProperties(fname=path)

train.head(5)
# sample id 100241706_00004_2

image_id = "100241706_00004_2"

labels = train[train["image_id"] == image_id]["labels"].values[0]

labels_list = np.array(labels.split(" ")).reshape(-1, 5)

#labels_list
getUnicode("U+0031")
plt = displayImage(image_id)

ax = plt.gca()



for label in labels_list:

    ax = drawBoxAndText(ax, label)
class KuzushijiDataset(object):

    def __init__(self, df_data, root,  mode="train"):

        self.root = root

        self.mode = mode

        self.img = None

        self.dataset = df_data

        self.num_objs = num_classes

        # self.transforms = transforms

        # load all image files, sorting them to

        # ensure that they are aligned

        self.transform = transforms.ToTensor()

        self.imgs = list(sorted(os.listdir(root)))

    

    def getMaskAndLabel(self, labels_list):

        l_len = len(labels_list)

        boxes = np.zeros((l_len, 4))

        labels = np.zeros((l_len))

        #print(ll)

        #print(ll / label_length)

        #label_tensor = torch.zeros((l_len, self.num_objs), dtype=torch.int64)

        #for label in attribute_id.split():

        for idx in range(l_len):

            labels[idx] = unicode2labels[labels_list[idx][0]]

            #labels = unicode2labels[labels_list[idx][0]]

            #label_tensor[idx, labels] = 1

            boxes[idx] = labels_list[idx][1:]

            boxes[idx][2] = boxes[idx][2] + boxes[idx][0]

            boxes[idx][3] = boxes[idx][3] + boxes[idx][1]

        

        #label_tensor[l_len-1, self.num_objs-1] = 1

        #print(masks) # l

        #print(labels) # l

        return boxes, labels



    def __getitem__(self, idx):

        # load images ad masks

        #print(idx)

        #temp_dataset = self.dataset.iloc[[idx]]

        temp_dataset = self.dataset[idx]

        #print(temp_dataset)

        #print(temp_dataset["image_id"].values[0])

        #print("-------------------------------------------------------")

        

        img_path = os.path.join(self.root, toPath(temp_dataset["image_id"].values[0]))

        img = Image.open(img_path).convert("RGB")        

        

        # print(np.array(img).shape)

        

        labels_col = temp_dataset["labels"].values[0]

        labels_list = np.array(str(labels_col).split(" ")).reshape(-1, 5)

        l_len = len(labels_list)

        

        boxes, labels = self.getMaskAndLabel(labels_list)        

        masks = np.zeros((np.array(img).shape[0], np.array(img).shape[1]))

        # print(masks.shape)

        for i in range(l_len):

            x, y, w, h = boxes[i]

            x, y, w, h = int(x), int(y), int(w), int(h)

            masks[y:h, x:w] = 1

            #masks[x:w, y:h] = 1

            #print(masks[x:h, y:w])

        

        image_id = torch.tensor([idx])

        # suppose all instances are not crowd

        

        labels = torch.as_tensor(labels, dtype=torch.int64)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        masks = torch.as_tensor(masks, dtype=torch.float32)

        

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((self.num_objs,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"] = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd

    

        if self.transform is not None:

            img = self.transform(img)



        return img, target



    def __len__(self):

        return len(self.imgs)

"""

kd = KuzushijiDataset(train_dir)

img, target = kd.__getitem__(0)



im = Image.fromarray(target["masks"])

#im = im.convert('RGB')

plt.figure(figsize=(15,15))

plt.imshow(im, cmap='gray')

"""

def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # create an anchor_generator for the FPN

    # which by default has 5 outputs

    

    anchor_generator = AnchorGenerator(sizes=((8, 16, 32, 64, 128),),aspect_ratios=((0.5, 1.0, 2.0),))

    #print(anchor_generator)

    #model.rpn.anchor_generator = anchor_generator



    # 256 because that's the number of features that FPN returns

    #model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

    # replace the pre-trained head with a new one

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    return model


unicode_map = {codepoint: char for codepoint, char in unicode.values}

unicode2labels = dict(zip(unicode_map.keys(), range(len(unicode_map.keys()))))



# our dataset has two classes only - background and Kuzushiji unicode

num_classes = len(unicode_map.keys()) + 1

# use our dataset and defined transformations

dataset = KuzushijiDataset(train, train_dir, mode="train")

#dataset_valid = KuzushijiDataset(train, train_dir, mode="train")

#dataset_test = KuzushijiDataset(train, test_dir, mode="test")



#indices = torch.randperm(len(dataset)).tolist()

#dataset = torch.utils.data.Subset(dataset, indices)

#dataset_valid = torch.utils.data.Subset(dataset, indices)



data_loader = torch.utils.data.DataLoader(

    dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)

hello
model = get_model_instance_segmentation(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.0001,momentum=0.9, weight_decay=0.00001)
num_epochs = 1

model_file = "model_saved.pkl"



model.train()

for epoch in range(num_epochs):

    final_loss_value = np.inf

    total_loss_value = 0

    loss_value = 0

    losses = 0

    # train for one epoch, printing every 10 iterations

    for batch_idx, (images, target) in enumerate(data_loader):

        images = list(image.to(device) for image in images)

        target = [{k: v.to(device) for k, v in t.items()} for t in target]

        loss_dict = model(images, target)

        #print(batch_idx)

        #print(loss_dict['loss_classifier'])

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        #print(loss_dict_reduced)

        #loss_value =+ losses_reduced.item()

        loss_value += losses_reduced.item() * len(images)

    

    loss_value = loss_value / len(data_loader.dataset)

    #train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=4)

    optimizer.zero_grad()

    losses.backward()

    optimizer.step()

    # update the learning rate

    #lr_scheduler.step()

    if loss_value < final_loss_value:

        final_loss_value = loss_value

        with open(model_file, "wb") as f:

            torch.save(model.state_dict(), f)

            print("Model saved :", model_file)