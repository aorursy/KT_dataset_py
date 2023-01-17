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
import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

import seaborn as sns

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import mean_squared_error

from math import sqrt

%matplotlib inline
input_train = '/kaggle/input/eval-lab-1-f464-v2/train.csv'

input_test = '/kaggle/input/eval-lab-1-f464-v2/test.csv'



train = pd.read_csv(input_train)

test = pd.read_csv(input_test)
productId = test['id']





# train.drop(labels='id', axis=1, inplace=True)

# test.drop(labels='id', axis=1, inplace=True)



train.fillna(train.mean(), inplace=True)
def normalize(column):

    upper = column.max()

    lower = column.min()

    y = (column - lower)/(upper-lower)

    return y
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(train['feature1'])

# #seabornInstance.distplot(normalize(np.log(train['feature1'])))

# # train['feature1'].describe()
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(train['feature2'])
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(np.log(train['feature3']))
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(train['feature4'])
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(np.log(train['feature5']))
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(normalize(train['feature6']))
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(np.log(train['feature7']+1))
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(np.log(train['feature8']+1))
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(np.log(train['feature9']+1))
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(train['feature10'])
# plt.figure(figsize=(15,10))

# plt.tight_layout()

# seabornInstance.distplot(train['feature11'])
def preprocess(df):

    df['type'] = np.where(df['type']=='new', 1, 0)

    df['type'].fillna(df['type'].mode()[0], inplace=True)

    df.fillna(df.mean(), inplace=True)

    df['feature3'] = np.log(df['feature3'])

    df['feature5'] = np.log(df['feature5'])

    

    return(df)
train = preprocess(train)
corrmat = train.drop('id', axis=1).corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmin=-0.8, vmax=0.8, square=False, annot=True);
# correlation = train.corr(method='pearson')

# columns = correlation.nlargest(10, 'rating').index

# columns
X = train.drop(['id', 'rating'], axis=1)

y = train['rating']

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [60, 70, 80],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [5, 7, 9],

    'n_estimators': [1500, 1600, 1700],

    'bootstrap' : [False],

    'max_features': ['sqrt']

}

clf = RandomForestRegressor()

clf_grid = GridSearchCV(estimator = clf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



clf_grid.fit(X, y)

test = pd.read_csv(input_test)

test = preprocess(test)

test.fillna(test.mean(), inplace=True)
y_pred_test = np.round(clf_grid.predict(test.drop('id', axis=1)))

final = pd.concat([test.id, pd.Series(y_pred_test)], axis=1)

final.columns = ['id', 'rating']

final.to_csv('final.csv', index=False)