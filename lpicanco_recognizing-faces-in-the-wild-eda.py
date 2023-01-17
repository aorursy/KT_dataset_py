# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/test")[:5])

print(os.listdir("../input/train")[:5])



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train_relationships.csv")

submit_df = pd.read_csv("../input/sample_submission.csv")
display("train_df", train_df.head())

display(train_df.describe())
display("submit_df", submit_df.head())
from PIL import Image

import matplotlib.pyplot as plt



def show_images(path):

    fig,ax = plt.subplots(2,4, figsize=(10,10))

    

    img_list = os.listdir(path)

    for i in range(8):

        img = Image.open(path + img_list[i])

        ax[i%2][i//2].imshow(img)

    fig.show()    

    

show_images(path = "../input/train/" + train_df.p1[0] + "/")

show_images(path = "../input/train/" + train_df.p2[0] + "/")