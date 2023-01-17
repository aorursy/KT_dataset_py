import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/bits-f464-l1/train.csv')

df.head()
X = df.drop("label",axis=1)

X = X.drop("id",axis=1)

X.head()
y=df["label"]

y.head()
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X = scaler.fit_transform(X)
from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor



adaregr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=None),n_estimators=350)
adaregr.fit(X,y)
df_test=pd.read_csv('../input/bits-f464-l1/test.csv')

df_test.head()
X_tt = df_test.drop("id",axis=1)

X_tt.head()
X_tt = scaler.transform(X_tt)
y_ada = adaregr.predict(X_tt) 
final = pd.read_csv('../input/bits-f464-l1/sampleSubmission.csv')

final["label"] = y_ada
final.to_csv('2017A7PS0029G_final.csv',index=False)