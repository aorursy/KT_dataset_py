import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
dftrain = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

dftest = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
dftrain=dftrain.fillna(dftrain.mean())

dftest=dftest.fillna(dftest.mean())
dftrain = pd.get_dummies(dftrain, columns=["type"])

dftest = pd.get_dummies(dftest, columns=["type"])
features = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9',

            'feature10','feature11','type_new']
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(n_estimators=2000)
etr.fit(dftrain[features],dftrain['rating'])
y_pred=etr.predict(dftest[features])
y_pred
y_pred=y_pred.round().astype('int64')
ta = []
for i in range(len(y_pred)):

    ta.append([dftrain['id'][i],y_pred[i]])
ta=pd.DataFrame(ta)
ta
ta['id']=ta[0]

ta['rating']=ta[1]
ta=ta.drop(0,axis=1)

ta=ta.drop(1,axis=1)
ta
ta.to_csv('ans.csv',index=False)