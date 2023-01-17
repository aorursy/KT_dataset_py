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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train
simple_train = train.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)

simple_test = test.drop(['PassengerId','Ticket','Cabin','Name'],axis=1)
simple_data = [simple_train,simple_test]

for d in simple_data:

    d['family'] = d['SibSp'] + d['Parch']

    d.drop(['SibSp','Parch'],axis=1,inplace=True)
simple_train
simple_train.isna().sum().sort_values(ascending=False)