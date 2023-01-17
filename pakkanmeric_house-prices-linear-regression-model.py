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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model, metrics
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
print(train.shape)

print(test.shape)
(train.isna().sum() / len(train) * 100).sort_values(ascending=False).head(10)
(test.isna().sum() / len(test) * 100).sort_values(ascending=False).head(10)
for col in train.columns:

    if (train[col].isna().sum() / len(train[col])) > 0.80:

        train.drop(columns = col, inplace = True)

        test.drop(columns = col, inplace = True)
train.isna().sum().sort_values(ascending=False).head(15)
test.isna().sum().sort_values(ascending=False).head(29)
train.info()
for col in test.columns:

    train[col].fillna(train[col].mode, inplace = True)

    test[col].fillna(train[col].mode, inplace = True)
train.isna().sum().sort_values(ascending=False).head(15)
test.isna().sum().sort_values(ascending=False).head(29)
for col in train.columns:

    if ((train[col].value_counts()/len(train[col])) > 0.7).any() == True:

        train.drop(columns = col, inplace = True)

        test.drop(columns = col, inplace = True)
train.corr().iloc[-1:,:].T.sort_values(by='SalePrice', ascending=False)
train.SalePrice.hist()
np.log(train.SalePrice).hist()
#train.SalePrice = np.log(train.SalePrice)



train.OverallQual = np.exp(train.OverallQual)

test.OverallQual = np.exp(test.OverallQual)
sns.set_style("whitegrid")

#sns.set_context("poster")

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1})

sns.boxplot(train.OverallQual, train.SalePrice)
vars_object = list()



for col in train.columns:

    if train[col].dtype == 'object':

        vars_object.append(col)

        

vars_object
for col in train.columns:

    if train[col].dtype.name == "object":

        train.drop(col, axis = 1, inplace = True)

        test.drop(col, axis = 1, inplace = True)

    elif col == 'SalePrice':

        continue

    elif test[col].dtype.name == "object":

        train.drop(col, axis = 1, inplace = True)

        test.drop(col, axis = 1, inplace = True)
for col in test.columns:

    if test[col].dtype.name == "object":

        print(col)
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(train.drop(train.columns[-1],axis=1), train.iloc[:,-1:], test_size=0.2, 

                                                    random_state=0)
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print('Coefficients: \n', reg.coef_)

print('Variance score: {}'.format(reg.score(X_test, y_test)))
plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, 

            color = "green", s = 10, label = 'Train data') 
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, 

            color = "blue", s = 10, label = 'Test data')
test.head()
test.info()
test_pred = reg.predict(test)
test_pred
my_submission = pd.DataFrame(data={'Id': test['Id'].values, 'SalePrice': test_pred.reshape(-1, )})

my_submission.head()



my_submission.to_csv(r'submission.csv', index=False)
my_submission.head()