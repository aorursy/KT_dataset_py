# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe=pd.read_csv("../input/compresive_strength_concrete.csv")
dataframe.isnull().sum()

dataframe.columns=['cement','blast','flyash','water','superplasticiser','coarse','fine','age','ccs']
col=list(dataframe.columns)
for i in col:
    dataframe[i]=(dataframe[i]-dataframe[i].mean())/dataframe[i].std(ddof=0)
dataframe
sns.heatmap(dataframe)
target=dataframe.ccs
features=dataframe.drop('ccs',axis=1)
target
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
X_train,X_test,y_train,y_test=train_test_split(features,target,test_size=0.2,train_size=0.8)
params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
print(r2_score(y_test, clf.predict(X_test)))