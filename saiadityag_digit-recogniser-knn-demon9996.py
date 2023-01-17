# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
submission=pd.read_csv("../input/sample_submission.csv")
test = pd.read_csv("../input/test.csv")


test.head(10)
train1,validate = train_test_split(train,test_size = 0.3,random_state = 100)
train_y=train1['label']
train_x=train1.drop(['label'],axis=1)
validate_y = validate['label']
validate_x = validate.drop('label',axis = 1)
test_x=test

model = KNeighborsClassifier()
model.fit(train_x,train_y)
validate_predict = model.predict(validate_x)
print(accuracy_score(validate_y,validate_predict))
test_pred = model.predict(test)

df_predict = pd.DataFrame(test_pred,columns = ['Label'])
df_predict['ImageId'] = test.index+1
df_predict.to_csv('submission.csv',index = False)
del df_predict
del train_x
del train_y
del test_x
del train
del submission
del test
