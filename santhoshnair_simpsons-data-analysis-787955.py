# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# Any results you write to the current directory are saved as output.
import glob

train_arr = []

for file in glob.glob("../input/train/train/*/*"):

    train_arr.append({"name": file, "label": file.split("/")[-2]})

df = pd.DataFrame(train_arr)
df.sample(frac=1).head()
df["label"].nunique()
df["label"].value_counts()
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode,iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

import matplotlib.cm as cm

import re
iplot([go.Bar(

x=df["label"].value_counts().keys(),

y=df["label"].value_counts())])
!pip install imagesize
import imagesize

df["width"] = 0

df["height"] = 0

df["aspect_ratio"] = 0.0

for idx, row in df.iterrows():

    width, height = imagesize.get(row["name"])

    df.at[idx, "width"] = width

    df.at[idx, "height"] = height

    df.at[idx, "aspect_ratio"] = float(height) / float(width)
df.head()
df["height"].hist()
df["width"].hist()
df["aspect_ratio"].hist()
from fastai.vision import *
path = Path("../input")

tfms = get_transforms()

data = ImageDataBunch.from_folder(path/"train",valid_pct=0.1, ds_tfms=tfms, size=224)
data.show_batch(rows=2, figsize=(5,5))
df.head()
data = ImageDataBunch.from_df("", df=df[["name", "label"]], label_col="label", folder="", size=64)
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet18, metrics=error_rate)
learn.fit_one_cycle(2)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(25,25))
interp.plot_top_losses(9, figsize=(25,25))
interp.most_confused()