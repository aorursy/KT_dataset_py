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
data = pd.read_csv("/kaggle/input/crimeagainstwomen/CrimeAgainstWomen.csv")
data.describe
data.columns
for col in data.columns:

    print (col , ":", len(data[col].unique()) )
pd.get_dummies(data , drop_first = True).shape
data['Crime Head'].value_counts().sort_values(ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:,:].values
labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_0 = LabelEncoder()

X[:, 0] = labelencoder_X_0.fit_transform(X[:, 0])
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("Crime Head", OneHotEncoder(), [1])],    remainder = 'passthrough')

X = ct.fit_transform(X).toarray()