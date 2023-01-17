import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from ast import literal_eval
train_df = pd.read_csv("../input/global-wheat-detection/train.csv")

train_df.head()
def convert(size, box):

    dw = 1. / size[0]

    dh = 1. / size[1]

    x = (box[0] + box[1]) / 2.0

    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]

    h = box[3] - box[2]

    x = x * dw

    w = w * dw

    y = y * dh

    h = h * dh

    return [x, y, w, h]



def convert_to_yolo_label(coco_format_box, w = 1024, h = 1024):

    bbox = literal_eval(coco_format_box)

    xmin = bbox[0]

    xmax = bbox[0] + bbox[2]

    ymin = bbox[1]

    ymax = bbox[1] + bbox[3]

    b = (float(xmin), float(xmax), float(ymin), float(ymax))

    yolo_box = convert((w, h), b)

    if np.max(yolo_box) > 1 or np.min(yolo_box) < 0: # Take this opportunity to check that conversion works

        print("BOX HAS AN ISSUE")

    return yolo_box
train_df['yolo_box'] = train_df.bbox.apply(convert_to_yolo_label)
unique_img_ids = train_df.image_id.unique()
len(unique_img_ids)
from tqdm import tqdm
for img_id in tqdm(unique_img_ids):

    filt_df = train_df.query("image_id == @img_id")

    all_boxes = filt_df.yolo_box.values

    file_name = img_id+".txt"



    s = "0 %s %s %s %s \n"

    with open(file_name, 'a') as file:

        for i in all_boxes:

            new_line = (s % tuple(i))

            file.write(new_line)  
import glob

all_imgs = glob.glob("../input/global-wheat-detection/train/*.jpg")

all_imgs = [i.split("/")[-1].replace(".jpg", "") for i in all_imgs]
negative_images = set(all_imgs) - set(unique_img_ids)
for i in tqdm(list(negative_images)):

    file_name = i+".txt"

    with open(file_name, 'w') as fp: 

        pass