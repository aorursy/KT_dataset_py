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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
dataset_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

dataset_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

dataset_train.sample(5)
corm = dataset_train.corr()

k = 11 

cols = corm.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(dataset_train[cols].values.T)

sns.set(font_scale=1.)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
dataset_train.columns
new_train = dataset_train[['SalePrice', 'SaleCondition', 'OverallCond','OverallQual', 'LowQualFinSF', 'GrLivArea','FullBath', 'TotRmsAbvGrd',

                           'GarageCars', 'GarageArea', 'TotalBsmtSF', 'YearBuilt', 'KitchenQual',

                           'GarageQual', 'Condition1', 'Condition2']]

new_train.head()
new_train = pd.get_dummies(new_train, columns=['OverallQual', 'SaleCondition', 'OverallCond',

                                               'KitchenQual', 'GarageQual',   'Condition1', 'Condition2'])

new_train.head()
new_train = new_train.drop(columns = ["Condition2_RRAe", "Condition2_RRAn", "Condition2_RRNn",'GarageQual_Ex',], axis = 1)

new_train.head()
x = new_train.iloc[:, 1:].values

y = new_train.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print('mean_squared_error is equal: ', mean_squared_error(y_test, y_pred),

     '\nr2_score is equal: ',r2_score(y_test, y_pred)

     )
test = dataset_test[['OverallQual', 'LowQualFinSF', 'GrLivArea','FullBath', 

                     'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'YearBuilt',

                     'SaleCondition', 'OverallCond','KitchenQual','GarageQual','Condition1', 'Condition2']]



test.isnull().sum()

test.TotalBsmtSF = test.TotalBsmtSF.fillna(value = 0.0)

test.GarageArea = test.GarageArea.fillna(value = 0.0)

test.GarageCars = test.GarageCars.fillna(value = 0)

test = pd.get_dummies(test, columns=['OverallQual', 'SaleCondition', 'OverallCond', 'KitchenQual',

                                     'GarageQual', 'Condition1', 'Condition2'])

test.shape
y_pred = regressor.predict(test)
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = y_pred

sub.to_csv('8.csv', index=False)