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
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

print(Iris.head())
print('\n\nColumn Names\n\n')

print(Iris.columns)
encode = LabelEncoder()

Iris.Species = encode.fit_transform(Iris.Species)

print(Iris.head())
train , test = train_test_split(Iris,test_size=0.2,random_state=0)

print('shape of training data :',train.shape)

print('shape of testing data',test.shape)
train_x = train.drop(columns=['Species'],axis=1)

train_y=train['Species']



test_x = test.drop(columns=['Species'],axis=1)

test_y = test['Species']
model = LogisticRegression()

model.fit(train_x,train_y)

predict = model.predict(test_x)

print('Predict Values on Test Data',encode.inverse_transform(predict))

print('\n\nAcuracy Score on test data: \n\n')

print(accuracy_score(test_y,predict))