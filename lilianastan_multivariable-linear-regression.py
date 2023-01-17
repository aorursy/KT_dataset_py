# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file_path = '/kaggle/input/kc-housesales-data/kc_house_data.csv'

df = pd.read_csv(file_path, delimiter=',')

df['age'] = 2020 - df.yr_built 

df.info()

#df=df[['price', 'bedrooms', 'sqft_living', 'floors', 'view', 'age']]
import seaborn as sns

# df.info()

corr_matrix = df.corr()

#print(corr_matrix)

plt.figure(figsize=(10,5))

# sns.heatmap(corr_matrix, annot=True)
# price, bedrooms, sqft_living, floors, view 

df_feature = df.sqft_living

# sns.countplot(df_feature, order=df_feature.value_counts().index)



plt.figure(figsize=(15,5))

sns.regplot(df_feature, df.price, scatter=True, marker="+",

            line_kws={"color": "orange"},)
X = df[['bedrooms', 'sqft_living', 'sqft_basement', 'floors', 'condition', 'grade', 'view', 'age']].values

y = df['price'].values

if X.shape[0] == y.shape [0]:

    print("Sanity check: equal number of rows for X and y :)")

else:

    print('Check your input data')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor



classifiers = dict(ols=linear_model.LinearRegression(fit_intercept=True, normalize=True),

                   #ridge=linear_model.Ridge(alpha=10000, solver='sag', tol=0.1), 

                   #lasso=linear_model.Lasso(alpha=0.1), #alpha = 0 is equivalent to an ordinary least square,

                   #enet=linear_model.ElasticNet(alpha=0.001, l1_ratio=0.001), #For l1_ratio = 0 the penalty is an L2 penalty.

                   RFR = RandomForestRegressor(n_estimators=500,criterion='mse',max_features='auto',max_depth=None,min_samples_leaf=2,min_samples_split=2,random_state=0)

                   ) 



for name, clf in classifiers.items():

    clf.fit(X_train, y_train)

    y_predicted_train = clf.predict(X_train)

    y_predicted_test = clf.predict(X_test)



    fig, ax = plt.subplots(figsize=(15,5))

    ax.set_xlabel('actual')

    ax.set_ylabel('predicted by ' + name)

    ax.scatter(y_train, y_predicted_train)

    ax.scatter(y_test, y_predicted_test, c='orange')

    ax.plot([y_train.min(), y_train.max()],

            [y_train.min(), y_train.max()],

            'k--', lw=1, c='black')

    print(name, " Train accuracy: %.4f"%clf.score(X_train, y_train))

    print(name, " Test accuracy: %.4f"%clf.score(X_test, y_test))

plt.show()    

    
