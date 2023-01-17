!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null

package_path = '../input/unetmodelscript'

import sys

sys.path.append(package_path)



import pdb

import os

import cv2

import torch

import pandas as pd

import numpy as np

from tqdm import tqdm_notebook as tqdm

import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader, Dataset

from albumentations import (Normalize, Compose)

from albumentations.pytorch import ToTensor

import torch.utils.data as data

from model import Unet



#https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



class TestDataset(Dataset):

    '''Dataset for test prediction'''

    def __init__(self, root, df, mean, std):

        self.root = root

        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

        self.fnames = df['ImageId'].unique().tolist()

        self.num_samples = len(self.fnames)

        self.transform = Compose(

            [

                Normalize(mean=mean, std=std, p=1),

                ToTensor(),

            ]

        )



    def __getitem__(self, idx):

        fname = self.fnames[idx]

        path = os.path.join(self.root, fname)

        image = cv2.imread(path)

        images = self.transform(image=image)["image"]

        return fname, images



    def __len__(self):

        return self.num_samples

    

def post_process(probability, threshold, min_size):

    '''Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored'''

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((256, 1600), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num



sample_submission_path = '../input/severstal-steel-defect-detection/train.csv'

test_data_folder = "../input/severstal-steel-defect-detection/train_images"



# initialize test dataloader

best_threshold = 0.5

num_workers = 2

batch_size = 4

print('best_threshold', best_threshold)

min_size = 3500

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

df = pd.read_csv(sample_submission_path)

testset = DataLoader(

    TestDataset(test_data_folder, df, mean, std),

    batch_size=batch_size,

    shuffle=False,

    num_workers=num_workers,

    pin_memory=True

)



# Initialize mode and load trained weights

ckpt_path = "../input/unetstartermodelfile/model.pth"

device = torch.device("cuda")

model = Unet("resnet18", encoder_weights=None, classes=4, activation=None)

model.to(device)

model.eval()

state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

model.load_state_dict(state["state_dict"])



# start prediction

predictions = []

for i, batch in enumerate(tqdm(testset)):

    fnames, images = batch

    batch_preds = torch.sigmoid(model(images.to(device)))

    batch_preds = batch_preds.detach().cpu().numpy()

    for fname, preds in zip(fnames, batch_preds):

        for cls, pred in enumerate(preds):

            pred, num = post_process(pred, best_threshold, min_size)

            rle = mask2rle(pred)

            name = fname + f"_{cls+1}"

            predictions.append([name, rle])



preds = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
import numpy as np # linear algebra

import pandas as pd

pd.set_option("display.max_rows", 101)

import os

print(os.listdir("../input"))

import cv2

import json

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["font.size"] = 15

import seaborn as sns

from collections import Counter

from PIL import Image

import math

import seaborn as sns

from collections import defaultdict

from pathlib import Path

import cv2



train_df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")



palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]



def name_and_mask(start_idx):

    col = start_idx

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError



    labels = train_df.iloc[col:col+4, 1]

    mask = np.zeros((256, 1600, 4), dtype=np.uint8)



    for idx, label in enumerate(labels.values):

        if label is not np.nan:

            mask_label = np.zeros(1600*256, dtype=np.uint8)

            label = label.split(" ")

            positions = map(int, label[0::2])

            length = map(int, label[1::2])

            for pos, le in zip(positions, length):

                mask_label[pos-1:pos+le-1] = 1

            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names[0], mask



def show_mask_image(col):

    name, mask = name_and_mask(col)

    img = cv2.imread(str(train_path / name))

    fig, ax = plt.subplots(figsize=(10, 10))



    for ch in range(4):

        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):

            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title('True mask of '+name)

    ax.imshow(img)

    plt.show()

    

train_path = Path("../input/severstal-steel-defect-detection/train_images/")



idx_no_defect = []

idx_class_1 = []

idx_class_2 = []

idx_class_3 = []

idx_class_4 = []

idx_class_multi = []

idx_class_triple = []



for col in range(0, len(train_df), 4):

    img_names = [str(i).split("_")[0] for i in train_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError

        

    labels = train_df.iloc[col:col+4, 1]

    if labels.isna().all():

        idx_no_defect.append(col)

    elif (labels.isna() == [False, True, True, True]).all():

        idx_class_1.append(col)

    elif (labels.isna() == [True, False, True, True]).all():

        idx_class_2.append(col)

    elif (labels.isna() == [True, True, False, True]).all():

        idx_class_3.append(col)

    elif (labels.isna() == [True, True, True, False]).all():

        idx_class_4.append(col)

    elif labels.isna().sum() == 1:

        idx_class_triple.append(col)

    else:

        idx_class_multi.append(col)

        

pred_df = preds.copy()

pred_df = pred_df.replace('', np.nan, regex=True)



def name_and_mask_pred(start_idx):

    col = start_idx

    img_names = [str(i).split("_")[0] for i in pred_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError



    labels = pred_df.iloc[col:col+4, 1]

    mask = np.zeros((256, 1600, 4), dtype=np.uint8)



    for idx, label in enumerate(labels.values):

        if label is not np.nan:

            mask_label = np.zeros(1600*256, dtype=np.uint8)

            label = label.split(" ")

            positions = map(int, label[0::2])

            length = map(int, label[1::2])

            for pos, le in zip(positions, length):

                mask_label[pos-1:pos+le-1] = 1

            mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names[0], mask



def show_mask_image_pred(col):

    name, mask = name_and_mask_pred(col)

    img = cv2.imread(str(train_path / name))

    fig, ax = plt.subplots(figsize=(10, 10))



    for ch in range(4):

        contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i in range(0, len(contours)):

            cv2.polylines(img, contours[i], True, palet[ch], 2)

    ax.set_title('Pred mask of '+name)

    ax.imshow(img)

    plt.show()

    

idx_no_defect_pred = []

idx_class_1_pred = []

idx_class_2_pred = []

idx_class_3_pred = []

idx_class_4_pred = []

idx_class_multi_pred = []

idx_class_triple_pred = []



for col in range(0, len(pred_df), 4):

    img_names = [str(i).split("_")[0] for i in pred_df.iloc[col:col+4, 0].values]

    if not (img_names[0] == img_names[1] == img_names[2] == img_names[3]):

        raise ValueError

        

    labels = pred_df.iloc[col:col+4, 1]

    if labels.isna().all():

        idx_no_defect_pred.append(col)

    elif (labels.isna() == [False, True, True, True]).all():

        idx_class_1_pred.append(col)

    elif (labels.isna() == [True, False, True, True]).all():

        idx_class_2_pred.append(col)

    elif (labels.isna() == [True, True, False, True]).all():

        idx_class_3_pred.append(col)

    elif (labels.isna() == [True, True, True, False]).all():

        idx_class_4_pred.append(col)

    elif labels.isna().sum() == 1:

        idx_class_triple_pred.append(col)

    else:

        idx_class_multi_pred.append(col)

        
from random import sample 



defects = list(set(idx_class_1).intersection(set((idx_class_1_pred))))

len(idx_class_1), len(idx_class_1_pred), len(defects)
for idx in sample(defects,10):

    show_mask_image_pred(idx)

    show_mask_image(idx)
defects = list(set(idx_class_2).intersection(set((idx_class_2_pred))))

len(idx_class_2), len(idx_class_2_pred), len(defects)
defects = list(set(idx_class_3).intersection(set((idx_class_3_pred))))

len(idx_class_3), len(idx_class_3_pred), len(defects)
for idx in sample(defects,10):

    show_mask_image_pred(idx)

    show_mask_image(idx)
defects = list(set(idx_class_4).intersection(set((idx_class_4_pred))))

len(idx_class_4), len(idx_class_4_pred), len(defects)
for idx in sample(defects,10):

    show_mask_image_pred(idx)

    show_mask_image(idx)
defects = list(set(idx_class_multi).intersection(set((idx_class_multi_pred))))

len(idx_class_multi), len(idx_class_multi_pred), len(defects)
for idx in sample(defects,10):

    show_mask_image_pred(idx)

    show_mask_image(idx)
defects = list(set(idx_class_4_pred).difference(set((idx_class_4))))

print(len(defects))

for idx in sample(defects,10):

    show_mask_image_pred(idx)

    show_mask_image(idx)