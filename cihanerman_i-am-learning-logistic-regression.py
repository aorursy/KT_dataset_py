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
data = pd.read_csv("../input/voice.csv") # inport data
data.info() # data info
data.label = [1 if each == 'female' else 0 for each in data.label]
#print(data)
y = data.label.values
x_data = data.drop(['label'],axis = 1)
x = (x_data - np.min(x_data) / (np.max(x_data) - np.min(x_data))).values
print ('y: ',y)
print ('x_data: ',x_data)
print('x: ',x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print ('test accuracy: {}'.format(lr.score(x_test,y_test)) )