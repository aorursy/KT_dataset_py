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
from sklearn.datasets import load_iris

from sklearn import linear_model

import pandas as pd

iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")
iris.head()
print('\n\nColumn Names\n\n')

print(iris.columns)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

encode = LabelEncoder()

iris.species = encode.fit_transform(iris.species)



print(iris.head())
# train-test-split   

train , test = train_test_split(iris,test_size=0.2,random_state=0)



print('shape of training data : ',train.shape)

print('shape of testing data',test.shape)
# seperate the target and independent variable

train_x = train.drop(columns=['species'],axis=1)

train_y = train['species']



test_x = test.drop(columns=['species'],axis=1)

test_y = test['species']

model = LogisticRegression()



model.fit(train_x,train_y)



predict = model.predict(test_x)



print('Predicted Values on Test Data',encode.inverse_transform(predict))



print('\n\nAccuracy Score on test data : \n\n')

print(accuracy_score(test_y,predict))