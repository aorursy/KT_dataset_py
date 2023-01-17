# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import pandas as pd

from glob import glob

import os

import cv2

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt
input_path = "../input/"
def load_df(path):    

    def get_filename(image_id):

        return os.path.join(input_path, "train_images", image_id + ".png")



    df_node = pd.read_csv(path)

    df_node["file"] = df_node["id_code"].apply(get_filename)

    df_node = df_node.dropna()

    

    return df_node



df = load_df(os.path.join(input_path, "train.csv"))

len(df)



df.head()
import math



def get_filelist(diagnosis=0):

    return df[df['diagnosis'] == diagnosis]['file'].values



def subplots(filelist):

    plt.figure(figsize=(16, 9))

    ncol = 3

    nrow = math.ceil(len(filelist) // ncol)

    

    for i in range(0, len(filelist)):

        plt.subplot(nrow, ncol, i + 1)

        img = cv2.imread(filelist[i])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img)
filelist = get_filelist(diagnosis=0)

subplots(filelist[:9])
filelist = get_filelist(diagnosis=1)

subplots(filelist[:9])
filelist = get_filelist(diagnosis=2)

subplots(filelist[:9])
filelist = get_filelist(diagnosis=3)

subplots(filelist[:9])
filelist = get_filelist(diagnosis=4)

subplots(filelist[:9])
Counter(df['diagnosis'])
plt.hist(df['diagnosis'], bins=5)