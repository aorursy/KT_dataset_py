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
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

import fastai
path = Path('../input')
train_df = pd.read_csv(path/'train.csv')

train_df.head()
test_df = pd.read_csv(path/'sample_submission.csv')

print(test_df.shape)

test_df.head()
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)

train_data = ImageDataBunch.from_df(path/'train'/'train', train_df, ds_tfms=tfms, size=128)
train_data.show_batch(rows=3, figsize=(5,6))
train_data.classes,train_data.c
learn = cnn_learner(train_data, models.resnet50, metrics=[accuracy],model_dir="/tmp/model/")
learn.lr_find()
learn.recorder.plot(suggestion=True)
lr =1.0e-2

learn.fit_one_cycle(7,slice(lr))
learn.recorder.plot_losses()
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4, lr/5))
solution = pd.DataFrame(columns=test_df.columns)

solution
for index,row in test_df.iterrows():

  img_name = row['id']

  img = open_image(path/'test'/'test'/img_name)

  pred_class,pred_idx,outputs = learn.predict(img)

  solution.loc[len(solution)] = [img_name,outputs.numpy()[1]]
solution.to_csv('submission.csv', index=False)
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

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(df)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 