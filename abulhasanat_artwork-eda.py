!ls ../input
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
os.listdir("../input/dsnet-kaggledays-hackathon/")
os.listdir("../input/dsnet-kaggledays-hackathon/train/train")
# Create a dataframe to analyze data easily

BASE_PATH = "../input/dsnet-kaggledays-hackathon/train/train"

df_arr = []

for dirname in os.listdir(BASE_PATH):

    for filename in os.listdir(os.path.join(BASE_PATH, dirname)):

        df_dict = {"y": dirname, "path": os.path.join(BASE_PATH,dirname,filename)}

        df_arr.append(df_dict)
df = pd.DataFrame(df_arr)
# Q: How many training rows are present?

# df.nunique()

len(df)
# Q: Print some random rows

df.sample(10)
# How many classes are present?

df['y'].nunique()
# What's the class distribution?

df['y'].value_counts()
!pip install chart_studio
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import matplotlib.cm as cm

import re
iplot([go.Bar(

x=df["y"].value_counts().keys(),

y=df["y"].value_counts())])
!pip install imagesize
import imagesize

df["width"] = 0

df["height"] = 0

df["aspect_ratio"] = 0.0

for idx, row in df.iterrows():

    width, height = imagesize.get(row["path"])

    df.at[idx, "width"] = width

    df.at[idx, "height"] = height

    df.at[idx, "aspect_ratio"] = float(height) / float(width)
df["height"].hist()
df["width"].hist()
df["aspect_ratio"].hist()
df["aspect_ratio"].hist(bins=30)
# FastAI quick image loader

from fastai.vision import *
tfms = get_transforms()

data = ImageDataBunch.from_folder(BASE_PATH,valid_pct=0.1, ds_tfms=tfms, size=224)
data.show_batch(rows=2, figsize=(5,5))
df.sample()["path"].values[0]
from IPython.display import Image

Image(filename=df.sample()["path"].values[0]) 
# Refer: https://stackoverflow.com/questions/36006136/how-to-display-images-in-a-row-with-ipython-display

import matplotlib.pyplot as plt

import cv2

def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10)):

    fig = plt.figure(figsize=figsize)

    column = 0

    for i in range(len(list_of_images)):

        column += 1

        #  check for end of column and create a new figure

        if column == no_of_columns+1:

            fig = plt.figure(figsize=figsize)

            column = 1

        fig.add_subplot(1, no_of_columns, column)

        image = cv2.imread(list_of_images[i])

        im2 = image.copy()

        im2[:, :, 0] = image[:, :, 2]

        im2[:, :, 2] = image[:, :, 0]

        plt.imshow(im2)

        plt.axis('off')

        if len(list_of_titles) >= len(list_of_images):

            plt.title(list_of_titles[i])
temp_df = df[df["y"] == "Raphael"].sample(n=9)

images = temp_df["path"].values

titles = temp_df["y"].values

grid_display(images, titles, 3, (10,10))