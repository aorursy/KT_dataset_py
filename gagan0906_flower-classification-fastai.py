import pandas as pd

import numpy as np

import os



from fastai import *

from fastai.vision import *
files_ = []

categories = []



path = "../input/flowers-recognition/flowers/flowers/"

for category in os.listdir(path):

    for file in os.listdir(os.path.join(path, category)):

        if file.split('.')[-1]=='jpg':

            files_.append(os.path.join(category, file))

            categories.append(category)



df = pd.DataFrame({'name': files_, 'label': categories})

df.head()
data = ImageDataBunch.from_df(path='../input/flowers-recognition/flowers/flowers/',df=df, size=224, bs=8)
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate], 

                   model_dir=Path("/kaggle/working/"), 

                   path=Path("."))
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(2)