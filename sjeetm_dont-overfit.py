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
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_data.head()
train_data.target = train_data.target.astype(int)
import pandas_profiling as ppf

ppf.ProfileReport(train_data)
data = train_data.drop('id',axis=1)
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder



y = train_data['target']

train_data_X = train_data.drop('target',axis=1)

mlp_clf = MLPClassifier(alpha=1)

mlp_clf.fit(train_data_X,y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_data_X,y,test_size = 0.1)

from sklearn.metrics import accuracy_score as ac

mlp_clf = MLPClassifier(alpha=3)

mlp_clf.fit(X_train,y_train)

pred = mlp_clf.predict(X_test)

print(ac(y_test,pred))
submition = pd.DataFrame(mlp_clf.predict(test_data))

submition.reset_index(level = 0, inplace = True)

submition.columns = ['id', 'target']

submition.id = test_data.id

submition.target = submition.target.astype(int)

submition.head()
#print(submition)

submition.to_csv('submission.csv',index = False)