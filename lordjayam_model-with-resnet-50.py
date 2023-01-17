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
from fastai.vision import *
path = Path("../input")

train_path = path/"train/train"



test_path = path/"test/test"



sub_df = pd.read_csv(f"{path}/sample_submission.csv")

test_df = pd.read_csv(f"{path}/sample_submission.csv")
train = get_image_files(train_path)

test  = get_image_files(test_path)
data= ImageDataBunch.from_folder(train_path,valid_pct = 0.2,test = test_path,ds_tfms = get_transforms(),size = 224).normalize()

data.add_test(ImageList.from_df(test_df, path, folder="test/test"))
data.show_batch()
learn = cnn_learner(data,models.resnet50,metrics = error_rate,model_dir="/tmp/model/")
data.c
lr_find(learn)
learn.recorder.plot()
lr = 1e-03
learn.fit_one_cycle(5, slice(lr))
learn.fit_one_cycle(5, slice(lr))
learn.fit_one_cycle(5, slice(lr))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(25,25))

learn.save("model-1")
learn.unfreeze()

lr_find(learn)

learn.recorder.plot()

lr_a = 1e-4
learn.fit_one_cycle(5,slice(lr_a,lr/5))

learn.save("model-2")
learn.fit_one_cycle(5,slice(1e-5,lr/5))

learn.save("model-3")
learn.fit_one_cycle(5,slice(1e-5,lr/5))
learn.load("model-3")
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
sub_df.predicted_class = test_preds

sub_df.to_csv("submission.csv", index=False)

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub_df)
from IPython.display import FileLink

FileLink("/tmp/model/export.pkl")

lr_find(learn)
learn.recorder.plot()
learn.save("model-2")
learn.load("model-2")
data_1= ImageDataBunch.from_folder(train_path,valid_pct = 0.2,test = test_path,ds_tfms = get_transforms(),size = 128).normalize(imagenet_stats)
learn.data=data_1

learn = learn.to_fp16()
learn.freeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5,1e-3)
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,1e-6)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(25,25))

learn.save("model1")
learn.unfreeze()
lr_find(learn)
learn.recorder.plot()
df = pd.read_csv("submission.csv")
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]

sub_df.predicted_class = test_preds

sub_df.to_csv("submission.csv", index=False)

sub_df.head()
import base64

from IPython.display import HTML

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(df)
learn.export('/tmp/model/learn.pkl')
