import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mp

import matplotlib.image as mpimg

import os

import random
file_names = os.listdir("../input/train/")

labels = pd.read_csv('../input/train.csv')

annotation = pd.read_csv("../input/labels.csv")
my_dpi = 96

def get_image(name):

    path = os.path.join("../input/train", name)

    image = mpimg.imread(path)

    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

    imgplot = plt.imshow(image)

def get_label(name):

    att = []

    img_id = name.split(".")[0]

    num_lab = labels.loc[labels["id"] == img_id]["attribute_ids"].values[0].split()

    for lab in num_lab:

        att.append(annotation.loc[annotation["attribute_id"] == int(lab)]["attribute_name"].values[0])

    return(att)
name = random.choice(file_names)

get_image(name)
get_label(name)
import csv

with open("file_names.csv", 'w', newline='') as myfile:

     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

     wr.writerow(file_names)