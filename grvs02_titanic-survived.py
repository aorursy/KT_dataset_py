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
import pandas as pd
train = pd.read_csv('../input/train.csv')
train.head()
feature_columns =['Parch', 'Pclass']
x = train.loc[:, feature_columns]
x.shape
y = train.Survived
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x,y)

test = pd.read_csv('../input/test.csv')
test.head()
x_new = test.loc[:, feature_columns]
x_new.shape
new_predict = clf.predict(x_new)
new_predict.shape

pd.DataFrame({'PassengerId':test.PassengerId,'Survived':new_predict}).set_index('PassengerId').to_csv('sub.csv')

