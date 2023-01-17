# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from math import sqrt



from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_squared_error



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.info()
test.info()
numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
numeric_features_cols = numeric_features.columns.tolist()
print('The dataset has',len(numeric_features.dtypes),'numeric features')
categorical_features = train.select_dtypes(exclude=[np.number])

categorical_features_cols = categorical_features.columns.tolist()

categorical_features.dtypes
print('The dataset has',len(categorical_features.dtypes),'categorical variables.')
msno.bar(train[numeric_features_cols])
msno.bar(train[categorical_features_cols])
sns.distplot(train['SalePrice'], kde=False);

plt.title('House Prices', fontsize=18);

plt.xlabel('Price', fontsize=16);

plt.ylabel('Frequency', fontsize=16);
train.SalePrice.describe()
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(train[numeric_features_cols].corr(), cmap="YlGnBu");
corr_saleprice = train[numeric_features_cols].corr()

corr_saleprice['SalePrice'][corr_saleprice['SalePrice'] > 0.5]
#We obtain the numeric columns that are positively correlated with SalePrice

pos_corr_saleprice_cols = corr_saleprice['SalePrice'][corr_saleprice['SalePrice'] > 0.5].index.tolist()

pos_corr_saleprice_cols
#We remove SalePrice

pos_corr_saleprice_cols.remove('SalePrice')

pos_corr_saleprice_cols
#Review if these columns have missing values

msno.bar(train[pos_corr_saleprice_cols])
X = train[pos_corr_saleprice_cols]

y = np.log(train.SalePrice)
X_train, X_test, y_train, y_test = train_test_split(

                          X, y, random_state=42, test_size=.3)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
#Function score returns R^2

model.score(X_test, y_test)
predictions = model.predict(X_test)
sqrt(mean_squared_error(y_test, predictions))
sqrt(mean_squared_error(np.log(y_test), np.log(predictions)))