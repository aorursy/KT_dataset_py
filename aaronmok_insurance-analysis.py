# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import SGDRegressor

from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

df = pd.read_csv('../input/insurance.csv')

df.head()



# Any results you write to the current directory are saved as output.
#no missing values



df.info()
sns.pairplot(df[['age','bmi','children','smoker','charges']])
#heat map, smoking, age and bmi have a stronger relationship with charges

sns.heatmap(df.corr(), annot=True)
#male vs female and their charges, # of chilren and their charges

fig, ax = plt.subplots(1,2, figsize=(14,8))

sns.boxplot(x = 'sex',y='charges', data = df, ax=ax[0])

ax[0].set_title('Male vs Female')

sns.boxplot(x = 'children',y='charges', data = df, ax=ax[1])

ax[1].set_title('Number of children')

#binning, grouping # of children into family size

bins = [0,1,3,5]

labels = ['Small_Family','Medium_Family','Large_Family']

df['children_binned'] = pd.cut(df['children'], bins=bins, labels=labels)

sns.boxplot(x = 'children_binned',y='charges', data = df)



df.info()

#define the functions and the models



column_name = ['age', 'bmi']

def get_model(X_train, X_test, y_train, y_test, model):



    x_preprocessing = ColumnTransformer([('transformer', StandardScaler(), ['age', 'bmi'])], remainder='passthrough')

    x_preprocessing.fit(X_train)

    x_train_pre = x_preprocessing.transform(X_train)

    x_test_pre = x_preprocessing.transform(X_test)

              

    y_preprocessing = StandardScaler()

    y_preprocessing.fit(y_train)

    y_train_pre = y_preprocessing.transform(y_train)

    y_test_pre = y_preprocessing.transform(y_test)



    if model == 'LR':

        model = LinearRegression()

        model.fit(x_train_pre,y_train_pre)

        score = model.score(x_test_pre,y_test_pre)

        print(score)

        

    if model == 'RidgeCV':

        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])

        model.fit(x_train_pre,y_train_pre) 

        score = model.score(x_test_pre,y_test_pre) 

        print(score)

        

    if model == 'SGD':

        model = SGDRegressor(max_iter=1000, tol=1e-3)

        model.fit(x_train_pre,y_train_pre)

        score = model.score(x_test_pre,y_test_pre) 

        print(score)

        

    if model == 'xgboost':

        model = XGBRegressor()

        model.fit(x_train_pre,y_train_pre, eval_metric = 'error',eval_set = [(x_test_pre,y_test_pre)],verbose = False) 

        score = model.score(x_test_pre,y_test_pre) 

        print(score)

        

        

#dummy variables, creating x and y inputs



df_2 = pd.get_dummies(df, columns = ['children_binned','region'])

df_2.head()

df_2 = df_2.drop(['children'], axis = 1)



df_x = df_2.loc[:, df_2.columns != 'charges']

df_y = df_2[['charges']]



X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
#run the models

for i in ['LR','RidgeCV','SGD','xgboost']:

    print(f"{i} score:")

    get_model(X_train = X_train_1, X_test = X_test_1, y_train = y_train_1, y_test =y_test_1, model = i)

    

#XGboost has the highest accuracy