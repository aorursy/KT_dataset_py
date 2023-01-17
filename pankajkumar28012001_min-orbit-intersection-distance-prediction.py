# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv('../input/near-earth-comets/near-earth-comets.csv')

X = dataset.iloc[:, 1:10].values

y = dataset.iloc[:, 10].values

sns.distplot(dataset['MOID'],bins=100)



sns.jointplot(x='e',y='MOID',data=dataset,kind='reg')
sns.barplot(x='Epoch',y='MOID',data=dataset)
sns.lmplot(x='Epoch',y='MOID',data=dataset)

sns.lmplot(x='Node',y='MOID',data=dataset)

sns.lmplot(x='TP',y='MOID',data=dataset)

sns.lmplot(x='e',y='MOID',data=dataset)

sns.lmplot(x='i',y='MOID',data=dataset)

sns.lmplot(x='w',y='MOID',data=dataset)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.set_printoptions(precision=2)

# column 0 reprsents predicted data 

# column 1 reprents test data

print("column 0 reprsents predicted data ")

print("column 1 reprents test data")

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))