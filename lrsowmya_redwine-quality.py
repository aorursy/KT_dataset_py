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
import pandas as pd
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.dropna()
df.describe()
df.corr()
df.columns
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
X = df[features]
y = df['quality']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
k3 = KNeighborsRegressor(n_neighbors=3)
k5 = KNeighborsRegressor(n_neighbors=5)
k10 = KNeighborsRegressor(n_neighbors=10)
k30 = KNeighborsRegressor(n_neighbors=10)
k50 = KNeighborsRegressor(n_neighbors=50)

k3.fit(X_train, y_train)
k5.fit(X_train, y_train)
k10.fit(X_train, y_train)
k30.fit(X_train, y_train)
k50.fit(X_train, y_train)
print(f'K = 3 train_score {k3.score(X_train,y_train)} & test_score {k3.score(X_test, y_test)}')
print(f'K = 5 train_score {k5.score(X_train,y_train)} & test_score {k5.score(X_test, y_test)}')
print(f'K = 10 train_score {k10.score(X_train,y_train)} & test_score {k10.score(X_test, y_test)}')
print(f'K = 30 train_score {k30.score(X_train,y_train)} & test_score {k30.score(X_test, y_test)}')
print(f'K = 50 train_score {k50.score(X_train,y_train)} & test_score {k50.score(X_test, y_test)}')
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

print(f'intercept found is {linear_model.intercept_} and coef found in {linear_model.coef_}')
print(f'train_score {linear_model.score(X_train, y_train)} and test_score {linear_model.score(X_test, y_test)}')