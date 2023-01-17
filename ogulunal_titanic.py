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
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

from termcolor import colored

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from catboost import Pool, CatBoostClassifier, cv

import pandas as pd

from sklearn.model_selection import train_test_split

#train_full=pd.read_csv('/kaggle/input/titanic/train.csv')

#test=pd.read_csv('/kaggle/input/titanic/test.csv')

#full=full.drop(['Passenger'],axis=0)
#!pip install catboost==0.14.2   
rnd_state = 1923



# read data

df = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId') #данные обучения (датафрейм)



df.fillna(-1000, inplace=True)   #заполнить пропущенные значения или NAN



X = df.drop('Survived', axis=1)  #целевой столбец для прогнозирования



y = df.Survived   #целевой столбец для прогнозирования





X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rnd_state) # данные тренировки и данные тестирования 0,8 - 0,2
categorical_features_indices = np.where(X.dtypes != np.float)[0]  #определить значения, которые не являются числами

clf = CatBoostClassifier(random_seed=rnd_state, custom_metric='CrossEntropy')   #алгоритм: повышение градиента с помощью дерева решений

#наша задача состоит в том, чтобы определить, кто может выжить в титановой аварии

clf.fit(X_train, y_train, cat_features=categorical_features_indices)

clf.score(X_val, y_val)

test_df = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')  #данные тестирования (датафрейм)

test_df.fillna(-1000, inplace=True)   #заполнить пропущенные значения или NAN (test data)

clf_od = CatBoostClassifier(iterations=1200,random_seed=rnd_state, od_type='Iter', od_wait=50, eval_metric='CrossEntropy', learning_rate=0.01, l2_leaf_reg=3.5, depth=10, loss_function= 'Logloss') #Если происходит переоснащение, CatBoost может остановить тренировку раньше, чем диктуют параметры тренировки.

#cross entropy Кросс-энтропийная потеря обычно используется как функция потерь для задачи классификации.

# 0.01 learning rate is enough more causes underfit and yet this work has still around %80 accuracy. The problem is not related to learning rate

# l2_leaf_reg  it is L2 regularization.

# od_wait stops the training when it detects overfitting.

# for CATBOOST, depth should be around 4- 10

clf_od.fit(X, y, cat_features=categorical_features_indices)  # fit

# нужно больше параметров настройки для улучшения результатов

# можеть быть\ добавить +1 функцию для тренировочного набора



clf_od.predict(test_df).astype('int')  # предсказать модель


output=pd.DataFrame({'PassengerId':test_df.index,'Survived':clf_od.predict(test_df).astype('int')})

output.to_csv('submissionOgul.csv',index=False)  # представить код для соревнований Kaggle в формате .csv,  it takes target column(Survived and test data as well)