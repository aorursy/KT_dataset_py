# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation,linear_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation,linear_model
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
dataFrame=pd.read_csv('../input/data.csv')

datFrame=dataFrame.drop(['Unnamed: 32'],1,inplace=True)
labels=dataFrame['diagnosis']

data_dummy=pd.get_dummies(dataFrame)
print(data_dummy.head())

features=data_dummy.loc[:,'radius_mean':'fractal_dimension_worst']
print(features)
print(labels)

print(features.shape)
print(labels.shape)

X_train,X_test,y_train,y_test=cross_validation.train_test_split(features,labels,test_size=0.2)

classifier=linear_model.LogisticRegression()
classifier.fit(X_train,y_train)

print("Train model score : ",classifier.score(X_train,y_train))
print("Test model score : ",classifier.score(X_test,y_test))


