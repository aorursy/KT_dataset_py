# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import csv
def id_loader(file_link):
    table=dict()
    table['photo_id']=list()
    table['business_id']=list()
    with open(file_link) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            table['photo_id'].append(row["photo_id"])
            table['business_id'].append(row["business_id"])
    return table
def label_loader(file_link):
    table=dict()
    table['business_id']=list()
    table['labels']=list()
    with open(file_link) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            table['business_id'].append(row["business_id"])
            table['labels'].append(row["labels"])
    return table
id_label=label_loader('../input/train.csv')
id_id=id_loader('../input/train_photo_to_biz_ids.csv')
def load_jpg(folder, id_label, id_id):
    tmplist=list()
    x_size = 500
    y_size = 375
    my_size = x_size, y_size
    pixel=3
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), y_size, x_size, pixel), dtype=np.float32)
    label = np.ndarray(shape=(len(image_files), 9), dtype=np.int32)
    image_index = 0
    for image in os.listdir(folder):
        if image.startswith("."):
            continue
        templst=image.split(".")
        bzid=id_id["business_id"][id_id["photo_id"].index(templst[0])]
        label_str=id_label["labels"][id_label["business_id"].index(bzid)]
        label_list=label_str.split(" ")
        for i in range(len(label_list)):
            try:
                index=int(label_list[i])
                label[image_index][index]=1
            except:
                label[image_index][0]=0
        #print(image_index)
        image_file = os.path.join(folder, image)
        #print(image_file)
        im = Image.open(image_file)
        if im.width < im.height:
            size=im.height,im.width
            im=im.rotate(90,expand=1).resize(size)
        if im.width is not 500 or im.height is not 375:
            im=im.resize(my_size, Image.ANTIALIAS)
        dataset[image_index, :, :,:] = im
        image_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :,:]
    return dataset, label
folder = '../input/train_photos/'
train_data, label=load_jpg(folder, id_label, id_id)
