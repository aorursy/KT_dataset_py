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
import matplotlib.pyplot as plt

import seaborn as sns
ins_table = pd.read_csv('/kaggle/input/insurance/insurance.csv')
ins_table.head()
ins_table.info()
ins_table.isnull().sum()
ins_table.head()
plt.figure(figsize=(10, 5))

sns.countplot(data = ins_table, x = 'sex')
plt.figure(figsize=(10,5))

sns.countplot(data = ins_table, x= 'region')
plt.figure(figsize=(10,5))

sns.countplot(data = ins_table, x='smoker', hue = 'sex')
plt.figure(figsize=(10,5))

sns.countplot(data = ins_table, x='children')
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, r2_score
ins_table.head()
ins_table['sex'] = pd.get_dummies(ins_table['sex'], drop_first=True)
ins_table.head()
ins_table['smoker'] = pd.get_dummies(ins_table['smoker'], drop_first=True)
def reg(arr):

    if arr['region'] == 'northeast':

        return 1

    elif arr['region'] == 'northwest':

        return 2

    elif arr['region'] == 'southeast':

        return 3

    else:

        return 4

ins_table['region'] = ins_table.apply(reg, axis = 1)
ins_table.head()
X = ins_table.drop('charges',axis = 1)

y = ins_table['charges']
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size = 0.3, random_state = 100)
lin = LinearRegression()
lin.fit(X_train, y_train)
pred_1 = lin.predict(X_test)
print(r2_score(y_test, pred_1))
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()

dtree.fit(X_train, y_train)
pred_2 = dtree.predict(X_test)
print(r2_score(y_test, pred_2))
from sklearn.ensemble import RandomForestRegressor

rfor = RandomForestRegressor()

rfor.fit(X_train, y_train)
pred_3 = rfor.predict(X_test)
print(r2_score(y_test, pred_3))