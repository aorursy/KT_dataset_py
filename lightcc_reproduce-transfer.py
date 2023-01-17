import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')



# subsample

df = pd.read_csv("../input/PS_20174392719_1491204439457_log.csv")#, nrows=int(1e6))

df = df.iloc[:, : 10] #删掉最后一列“isFlaggedFraud”

df = df.query('type=="TRANSFER"')#仅保留TRANSFER类型的记录

df.head()
a = df.groupby('step')['nameOrig'].nunique() #pandas.core.series.Series

b = df.groupby('step')['nameDest'].nunique()

df1=pd.concat([a,b], axis=1)

df1.head() #已导出df1并且画好图
df1.to_csv('stepCountPopu.csv')