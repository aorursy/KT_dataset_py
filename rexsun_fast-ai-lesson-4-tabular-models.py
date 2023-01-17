import os 

import numpy as np

import pandas as pd

from fastai.tabular import *
path=untar_data(URLs.ADULT_SAMPLE)

path.ls()
df=pd.read_csv(path/"adult.csv")

df.head()
dep_var="salary"   # 目标特诊

cat_names=["workclass","education","marital-status","occupation","relationship","race"]  # 离散特征

con_names=["age","fnlwgt","education-num"]  # 连续特征

proc=[FillMissing,Categorify,Normalize]  # 预处理方法(处理缺失值、类别处理、标准化)
df.shape
# 测试集

test=TabularList.from_df(df.iloc[800:1000],path=path,cat_names=cat_names,cont_names=con_names)
# 建立databunch

data=(TabularList.from_df(df,path=path,cat_names=cat_names,cont_names=con_names,procs=proc)

                         .split_by_idx(list(range(800,1000)))

                         .label_from_df(cols=dep_var)

                         .add_test(test)

                         .databunch(bs=128))
# 展示数据

data.show_batch(rows=10)
# 建立200,100的DNN

learn=tabular_learner(data,layers=[200,100],metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(10,max_lr=1e-2)
# 预测

row=df.iloc[0]

row
learn.predict(row)