# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
import imagehash

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

TRAIN_IMG_PATH = r"../input/train"

def getImageMetaData(file_path):
    with Image.open(file_path) as img:
        img_hash = imagehash.phash(img)
        return img.size, img.mode, img_hash

def get_train_input():
    train_input = pd.read_csv(r"../input/train.csv")
    
    m = train_input.Image.apply(lambda x: getImageMetaData(TRAIN_IMG_PATH + "/" + x))
    train_input["Hash"] = [str(i[2]) for i in m]
    train_input["Shape"] = [i[0] for i in m]
    train_input["Mode"] = [str(i[1]) for i in m]
    train_input["Length"] = train_input["Shape"].apply(lambda x: x[0]*x[1])
    train_input["Ratio"] = train_input["Shape"].apply(lambda x: x[0]/x[1])
    train_input["New_Whale"] = train_input.Id == "new_whale"
    
    
    img_counts = train_input.Id.value_counts().to_dict()
    train_input["Id_Count"] = train_input.Id.apply(lambda x: img_counts[x])
    return train_input

train_input = get_train_input()

t = train_input.Hash.value_counts()
t = t.loc[t>1]
print("There are {} duplicate images.".format(np.sum(t)-len(t)))
t.head(20)
import collections

def plot_images(path, imgs):
    assert(isinstance(imgs, collections.Iterable))
    imgs_list = list(imgs)
    nrows = len(imgs_list)
    if (nrows % 2 != 0):
        nrows = nrows + 1 

    plt.figure(figsize=(18, 6*nrows/2))
    for i, img_file in enumerate(imgs_list):
        with Image.open(path + "/" + img_file) as img:
            ax = plt.subplot(nrows/2, 2, i+1)
            ax.set_title("#{}: '{}'".format(i+1, img_file))
            ax.imshow(img)
        
    plt.show()

print("Some examples:")
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[0]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[3]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[8]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[77]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[431]].Image)
plot_images(TRAIN_IMG_PATH, train_input[train_input.Hash==t.index[522]].Image)

