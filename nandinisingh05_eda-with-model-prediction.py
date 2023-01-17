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
import matplotlib as pyplot

% matplotlib inline
df2 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df1 = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df1.head()
df1.info()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import math

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
df1.shape
print(df1.columns[df1.isna().any()].tolist())

len(df1.columns[df1.isna().any()].tolist())
plt.rcParams['figure.figsize']=35,35

g = sns.heatmap(df1.corr(),annot=True, fmt = ".1f", cmap = "coolwarm")
sns.barplot(x='YearRemodAdd', y='SalePrice', data=df1)
sns.barplot(x='MSSubClass', y='SalePrice', data=df1)
sns.barplot(x='YearBuilt', y='SalePrice', data=df1)

sns.barplot(x='Fireplaces', y='SalePrice', data=df1)
lot_price = df1['LotArea'] + df1['SalePrice']

sns.distplot(lot_price)
df1.select_dtypes(include=int)
import seaborn as sns, pystan, statsmodels.api as sm

from sklearn import linear_model

# Creating the model based on Id

model = sm.OLS(df1.SalePrice,df1.Id).fit()

predictions = model.predict(df1.SalePrice) 

model.summary()
plt.scatter(df1.SalePrice,predictions, marker = "v")

np.corrcoef(df1.SalePrice,predictions)
df2.SalePrice = predictions
model = sm.OLS(df2.SalePrice,df1.OverallQual).fit()

predictions = model.predict(df1.OverallQual) 

model.summary()
submission = pd.DataFrame({

        "Id": df2['Id'],

        "SalePrice": predictions

    })
submission.to_csv('EDA with model prediction.csv', index=False)