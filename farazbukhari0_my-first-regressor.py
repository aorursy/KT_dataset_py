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
import sklearn as skt
import seaborn as sns
pd.set_option('display.max_columns', None)
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
print('Test Shape',test.shape)
print('Train Shape',train.shape)
test.head()
train.head()
i=0
for column in train.columns:
    if train[column].dtypes != 'object':
        i=i+1
i
i = 1
fig, ax = plt.subplots(10, 4, figsize = (20,20))
fig.tight_layout(pad=3.0)

for column in train.columns:
    if train[column].dtypes != 'object':
        plt.subplot(8,5,i)
        plt.xlabel(column)
        plt.scatter(train[column],train['SalePrice'])
        i=i+1


train.fillna(0,inplace=True)
#X = train.drop(['SalePrice'],axis=1)
X = train[['LotFrontage','EnclosedPorch','ScreenPorch','GarageArea','WoodDeckSF','OpenPorchSF','GrLivArea','2ndFlrSF','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1']]
y = train['SalePrice']
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X,y)
print('Linear Model Coeff (m)', regressor.coef_)
print('Linear Model Coeff (b)', regressor.intercept_)

test.fillna(0,inplace=True)
y_predict = regressor.predict(test[['LotFrontage','EnclosedPorch','ScreenPorch','GarageArea','WoodDeckSF','OpenPorchSF','GrLivArea','2ndFlrSF','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1']])
