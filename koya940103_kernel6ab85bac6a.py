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
# set dataframe

df = pd.read_csv("/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv")



print(df.dtypes) 



df = df.iloc[:, :1000]

#df = df.drop('nameOrig',axis=4)

#df = df.drop('nameDest',axis=7)



#X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

#y = pd.Series(dataset.target, name='isFraud')



X = df.iloc[:,:-2]  # isFraud以外をXとする

y = df.iloc[:,-2] # isFraudをyとする



# check the shape

print('----------------------------------------')

print('X shape: (%i,%i)' %X.shape)

print('-----------------------------------------')

print(y.describe())

print('----------------------------------------')

print('Check null count of target variable: %i' % y.isnull().sum())

print('----------------------------------------')

display(X.join(y).head(5))





print(X['type'].value_counts())

print('----------------------------------------')

print(X['nameOrig'].value_counts())

print('----------------------------------------')

print(X['nameDest'].value_counts())



print(X['nameDest'].str[:1].value_counts())



print(X['nameOrig'].str[:1].value_counts())



X = X.drop('nameOrig', axis=1)



X['nameDest']  = X['nameDest'].str[:1]
ohe_columns = ['type', 'nameDest']



X_new = pd.get_dummies(X,

                       dummy_na=True,

                       columns=ohe_columns)



display(X_new.head())
from sklearn.impute import SimpleImputer



# インピュータークラスのインスタンス化と（列平均の）学習

imp = SimpleImputer()

imp.fit(X_new)



# 学習済みImputerの適用：各列の欠損値の置換

X_new_columns = X_new.columns.values

X_new = pd.DataFrame(imp.transform(X_new),columns=X_new_columns)



# 結果表示

display(X_new.head())
# 分類モデルの評価指標計算のための関数の読込

from sklearn.metrics import accuracy_score
# import libraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



# Holdout

X_train,X_test,y_train,y_test = train_test_split(X_new,

                                                 y,

                                                 test_size=0.20,

                                                 random_state=1)

# set pipelines for two different algorithms

pipelines ={

    'logistic': Pipeline([('scl',StandardScaler()),

                          ('est',LogisticRegression(random_state=1))])    

}



# fit the models

for pipe_name, pipeline in pipelines.items():

    pipeline.fit(X_train,y_train)

    print(pipe_name, ': Fitting Done')
scores = {}

for pipe_name, pipeline in pipelines.items():

    scores[(pipe_name,'train')] = accuracy_score(y_train, pipeline.predict(X_train))

    scores[(pipe_name,'test')] = accuracy_score(y_test, pipeline.predict(X_test))



pd.Series(scores).unstack()