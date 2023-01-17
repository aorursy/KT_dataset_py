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
work_dir = Path('/kaggle/working/')

path = Path("../input")

tfms = get_transforms()

data = ImageDataBunch.from_folder(path/"train",valid_pct=0.1, ds_tfms=tfms, size=224,bs=64,seed=42)
data.add_test(ImageList.from_folder(path=path/"test/test"))
data.normalize(imagenet_stats)
print(data.valid_ds)

print("...")

print(data.train_ds)

print("...")

print(data.test_ds)
data.show_batch(rows=3, figsize=(5,5))
# df.head()
# data = ImageDataBunch.from_df("", df=df[["name", "label"]], label_col="label", folder="", size=64)
# data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=[accuracy],model_dir="/tmp/model/")
learn.fit_one_cycle(4)
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-2))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(25,25))
interp.plot_top_losses(9, figsize=(25,25))
interp.most_confused()
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
test_df = pd.read_csv(f"../input/sample_submission.csv")
fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'id':fnames, 'predicted_class':test_preds}, columns=['id', 'predicted_class'])

df['id'] = df['id'].astype(str) + '.jpg'

df.to_csv('submission-f.csv', index=False)
test_df.predicted_class = test_preds

test_df.to_csv(work_dir/"submission.csv", index=False)
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe



# create a link to download the dataframe

create_download_link(df)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
