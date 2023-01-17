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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
wine = pd.read_csv('../input/wineQualityReds.csv')
wine.head()
wine.info()
wine.describe()
wine.shape
wine.head()
wine.isnull().sum()
wine[wine.quality>7]
model_rf=RandomForestClassifier()

X=wine.loc[:,:'alcohol']
y=wine.quality
print(X.head(),y.head())
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=15)
model_rf.fit(X_train,y_train)
y_predict=model_rf.predict(X_test)
accuracy_score(y_test,y_predict)
