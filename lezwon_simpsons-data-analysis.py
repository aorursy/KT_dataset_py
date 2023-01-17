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
test_df = pd.read_csv(f"../input/sample_submission.csv")
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

tfms = get_transforms(do_flip=True, max_rotate=10, max_zoom=1.2, max_lighting=0.3, max_warp=0.15)

data = ImageDataBunch.from_folder(path/"train",valid_pct=0.3, ds_tfms=tfms, size=224)
data.show_batch(rows=2, figsize=(5,5))
df.head()
data = ImageDataBunch.from_df("", df=df[["name", "label"]], label_col="label", folder="", size=64)

data.add_test(ImageList.from_df(test_df, '../input', folder="test/test"))
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 1e-2)
learn.lr_find();
learn.recorder.plot()
learn.fit_one_cycle(3, 1e-25)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(25,25))
interp.plot_top_losses(9, figsize=(25,25))
interp.most_confused()
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
test_df.predicted_class = test_preds

test_df.to_csv("submission.csv", index=False)