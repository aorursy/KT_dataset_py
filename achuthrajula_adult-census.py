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
import numpy as np

import pandas as pd

import seaborn as sn

import matplotlib.pyplot as plt
data = pd.read_csv("../input/adult.csv")

data.head()
data = data.drop(['fnlwgt', 'education.num', 'marital.status', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week'], axis = 1)

data = data.rename({'native.country': 'nation'}, axis='columns')

data
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

data['age'] = label_encoder.fit_transform(data['age'])

data['workclass'] = label_encoder.fit_transform(data['workclass'])

data['education'] = label_encoder.fit_transform(data['education'])

data['occupation'] = label_encoder.fit_transform(data['occupation'])

data['nation'] = label_encoder.fit_transform(data['nation'])

data['income'] = label_encoder.fit_transform(data['income'])

data.head()
x = data.iloc[:, data.columns != 'occupation']

y = data.iloc[:, 3]
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values=0, strategy='most_frequent', axis=0)

x = imputer.fit_transform(x)

print(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

y_train = sc_y.fit_transform(y_train)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

from sklearn import metrics

print(metrics.accuracy_score(y_test, y_pred))