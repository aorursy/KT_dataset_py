import numpy as np

import pandas as pd

import seaborn as sns

import random

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn import ensemble, metrics, linear_model

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

import plotly.express as px

%matplotlib inline
df = pd.read_csv('../input/bits-f464-l1/train.csv')

df_test = pd.read_csv('../input/bits-f464-l1/test.csv')

df.head()
df_total = df.drop(['id','time','label'], axis=1)

df_total.head()
y_total = df['label']
df_total = df_total[[c for c in list(df_total) if len(df_total[c].unique()) > 1]]
df_total.shape
rf_total = RandomForestRegressor()

rf_total.fit(df_total, y_total)
mean_squared_error(rf_total.predict(df_total),y_total,squared=False)
df_test_total = df_test.drop(["id"], axis=1)

df_test_total.head()
df_test_total = df_test_total[df_total.columns]
df_test_total.head()
df_total.head()
ytotal = rf_total.predict(df_test_total)
test_res2 = pd.DataFrame(ytotal,index=df_test_total.index+1)
test_res2.to_csv('test_res2.csv')