import os

import numpy as np

import pandas as pd

from fastai.collab import *

from fastai.tabular import *
user,item,title="userId","movieId","title"
path=untar_data(URLs.ML_SAMPLE)

path.ls()
ratings=pd.read_csv(path/"ratings.csv")

ratings.head()
# 建立databunch

data=CollabDataBunch.from_df(ratings,seed=42)
y_range=(ratings.rating.min(),ratings.rating.max())

y_range
# 建立模型

learn=collab_learner(data,n_factors=50,y_range=y_range)
# 训练

learn.fit_one_cycle(3,max_lr=5e-3)