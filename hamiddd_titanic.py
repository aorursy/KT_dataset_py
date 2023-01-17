# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/titanic/train.csv")
features = ['Pclass', 'Sex', 'Age', 'Fare']

train.Sex[train.Sex=='male'] = 1

train.Sex[train.Sex=='female'] = 2

train['Age'] = train['Age'].fillna(0)

X = train[features]

X
y =train.Survived
y
classifier_linear = svm.SVC(kernel='linear')

classifier_linear.fit(X,y)