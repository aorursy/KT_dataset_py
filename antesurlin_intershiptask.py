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
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
test_data.head()
train_data.isnull().sum().sort_values(ascending=False)
test_data.isnull().sum().sort_values(ascending=False)
features = ['OverallQual', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'YearBuilt', 'LotArea', 'YearRemodAdd', 'OverallCond', 'Fireplaces']
y_train = train_data['SalePrice']
x_train = train_data[features]
x_test = test_data[features]
from sklearn import linear_model
import math
lm = linear_model.LinearRegression()
lm.fit(x_train,y_train)
print('R sq: ',lm.score(x_train,y_train))
print('Correlation: ', math.sqrt(lm.score(x_train,y_train)))
price_predict = lm.predict(x_test)
print('Coefficients: \n', lm.coef_)
plt.plot(y_train, color='blue', linewidth=3)
plt.plot(price_predict, color='blue', linewidth=3)