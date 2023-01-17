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
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
# Импорт нужной функции

from sklearn.model_selection import train_test_split



# Создание X, y

# X --- вся таблица без таргета

# y --- таргет (целевая переменная)



X = df.drop('quality', axis=1) 

y = df['quality'] 



# Разделение

# test_size --- доля исходных данных, которую оставляем для валидации

# random_state --- произвольное целое число, для воспроизводимости случайных результатов



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=12)
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=10) # max_depth --- один из гиперпараметров дерева
tree.fit(X_train, y_train)
y_pred = tree.predict(X_valid)
from sklearn.metrics import mean_squared_error, explained_variance_score

print('Mean squared error:', mean_squared_error(y_valid, y_pred))

print('Explained variance score:', explained_variance_score(y_valid, y_pred))