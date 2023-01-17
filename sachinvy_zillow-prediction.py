# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/properties_2016.csv")

out=pd.read_csv("../input/train_2016_v2.csv")
data.shape
final=pd.merge(data, out,on="parcelid",how='left')
final.shape

final.columns
corr=final.corr()
#filter = corr['logerror'] > .01 & corr['logerror'] < .01



corr = corr[(corr['logerror'] > .01) | (corr['logerror'] < -.01)]

corr=corr['logerror'].sort_values().reset_index()

columnlist=list(corr['index'].values)
train=final[columnlist]

train.isnull().any()
corr=train.corr()
ax=plt.figure(figsize=(15,15))

sns.heatmap(corr,vmax=1)
train
X_train=train.drop(['logerror'],axis=1)

Y_train=train['logerror']
rf = RandomForestRegressor(n_estimators=100,oob_score=True,random_state=42,max_features=.3)

rf.fit(X_train,Y_train)
rf.score(X_train,Y_train)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

y = rf.oob_prediction_

print(y)

roc_auc_score(Y_train,y)

dict=pd.read_excel("../input/zillow_data_dictionary.xlsx")
dict