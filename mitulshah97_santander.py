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
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
x=train.drop(['ID','TARGET'], axis=1)
y=train.TARGET
test1=test.drop(['ID'], axis=1)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
xscaled=scaler.transform(x)


from sklearn.decomposition import PCA
pca = PCA(svd_solver = 'auto')
pca.fit(xscaled)
principalComponents = pca.transform(xscaled)
  

from xgboost import XGBClassifier

model_rf = XGBClassifier(random_state=1211)
model_rf.fit(x, y)
y_pred = model_rf.predict(test1)
id1= test.ID
Label = y_pred

Label = pd.Series(Label)

submit = pd.concat([id1,Label],axis=1, ignore_index=True)
submit.columns=['ID','TARGET']

submit.to_csv("santander.csv",index=False)
