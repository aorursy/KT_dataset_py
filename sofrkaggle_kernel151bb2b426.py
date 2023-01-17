# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Импорт данных

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test  = pd.read_csv('/kaggle/input/titanic/test.csv')



# Создаем обучающую переменную

survived_train = df_train.Survived



# Объединяем обучающие и тестовые данные

data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])



display(data)

data.info()

# Опустим столбцы

data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)

data.head()
data.info()

# Заполним отсутствующие значения средними значениями

data['Age'] = data.Age.fillna(data.Age.mean())

data['Fare'] = data.Fare.fillna(data.Fare.mean())

data['Embarked'] = data['Embarked'].fillna('S')

data.info()

data.describe()
# Преобразуем в двоичные переменные

data_dum = pd.get_dummies(data, drop_first=True)

data_dum.head()
data_train = data_dum.iloc[:891]

data_test = data_dum.iloc[891:]



X = data_train.values

test = data_test.values

y = survived_train.values
dep = np.arange(1,9)

param_grid = {'max_depth' : dep}



clf = tree.DecisionTreeClassifier()



clf_cv = GridSearchCV(clf, param_grid=param_grid, cv=5)



clf_cv.fit(X, y)



print(clf_cv.best_params_)

print(clf_cv.best_score_)