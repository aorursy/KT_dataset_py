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


dataset = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
dataset.head()
test.head()
dataset1 = dataset[['SalePrice', 'SaleCondition', 'OverallCond',

                        'OverallQual',

                        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

                        'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 

                         'TotalBsmtSF', 'YearBuilt']]



dataset1.head()
dataset1 = pd.get_dummies(dataset1, columns=['OverallQual', 'SaleCondition', 'OverallCond'])

dataset1.head()
dataset1['SalePrice'].describe()
#histogram

sns.distplot(dataset1['SalePrice']);
var = 'GrLivArea'

data = pd.concat([dataset1['SalePrice'], dataset1[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
k = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(dataset1[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.set()

cols = ['SalePrice', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(dataset1[cols], size = 2.5)

plt.show();
x = dataset1.iloc[:, 1:].values

y = dataset1.iloc[:, 0].values
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
from sklearn.metrics import mean_squared_error, r2_score
print('mean_squared_error: ', mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
test1 = test[['OverallQual',

                        '1stFlrSF','2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

                        'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 

                         'TotalBsmtSF', 'YearBuilt', 'SaleCondition', 'OverallCond']]



test1.isnull().sum()

test1.TotalBsmtSF = test1.TotalBsmtSF.fillna(value = 0.0)

test1.GarageArea = test1.GarageArea.fillna(value = 0.0)

test1.GarageCars = test1.GarageCars.fillna(value = 0)

test1 = pd.get_dummies(test1, columns=['OverallQual', 'SaleCondition', 'OverallCond'])

test1.shape
x.shape
y_pred = regressor.predict(test1)
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = y_pred

sub.to_csv('final.csv', index=False)