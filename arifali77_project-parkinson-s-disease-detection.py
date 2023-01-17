# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os, sys

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df=pd.read_csv('/kaggle/input/parkinsons-data-set/parkinsons.data')

df.head()
df.shape
# Get the features and labels

features = df.loc[:,df.columns != 'status'].values[:,1:]

labels=df.loc[:,'status'].values
# Get the label of each label (0 and 1) in labels

print(labels[labels==1].shape[0], labels[labels==0].shape[0])
# Scale the features to between -1 and 1

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels
# Split the dataset

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
# Train the model

model=XGBClassifier()

model.fit(x_train,y_train)
#Calculate the accuracy

y_pred=model.predict(x_test)

print(accuracy_score(y_test, y_pred)*100)