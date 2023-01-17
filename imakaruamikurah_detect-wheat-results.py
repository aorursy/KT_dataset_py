import numpy as np

import pandas as pd

import os

import cv2



df = pd.read_csv("/kaggle/input/wheat-detect/wheat_detect.csv")

df = df.drop(df.columns[0], axis=1)

df.head()
submission = []

for i in range(len(df)):

    img = df.iloc[i]["img"]

    prediction = df.iloc[i]["prediction"]

    bbox = df.iloc[i]["bbox"]

    

    submission.append([img, prediction, bbox])
import matplotlib.pyplot as plt

from matplotlib import rcParams



img_path = "../input/detected-wheat-plots"



rcParams["figure.figsize"] = 15, 15

for i in os.listdir(img_path):

    img = cv2.imread(img_path + "/" + i)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.axis('off')

    plt.imshow(img, interpolation='nearest')

    plt.show()
from ast import literal_eval
boxes = []



for vals in submission:

    cur_id, score, box = vals[0], vals[1], vals[2]

    

    box = literal_eval(box)

    

    boxes.append([cur_id, f"{round(score, 4)} {box[0]} {box[1]} {box[2]-box[0]} {box[3]-box[1]}"])
df_new = pd.DataFrame(boxes)

df_new.columns = ["pic_id", "bbox"]



df_new
prev_pid = df_new.iloc[0]["pic_id"]

bbox_string = ""

bbox_array = []



for row_id in range(len(df_new)):

    pid = df_new.iloc[row_id]["pic_id"]

    bbox = df_new.iloc[row_id]["bbox"]

    

    bbox_string += "  " + bbox    

    

    if prev_pid != pid:

        bbox_array.append([prev_pid[:-4], bbox_string])

        prev_pid = pid



        bbox_string = ""

bbox_array.append([prev_pid[:-4], bbox_string])
df = pd.DataFrame(bbox_array, columns=["image_id", "PredictionString"])

df
sub_df = pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

sub_df
for i in range(10):

    sub = sub_df.iloc[i]["image_id"]

    for j in range(10):

        pre = df.iloc[j]["image_id"]

        if pre == sub:

            sub_df.iloc[i]["PredictionString"] = df.iloc[j]["PredictionString"]

            break
sub_df
sub_df.to_csv("submission.csv", index=False)