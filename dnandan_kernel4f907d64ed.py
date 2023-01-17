#invite people for the Kaggle party

import pandas as pd

import pandas_profiling as pdp

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline





from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression



#bring in the six packs

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df_train.columns
predictors=['LotArea', 'WoodDeckSF','ScreenPorch','YearBuilt','YearRemodAdd','GrLivArea','TotalBsmtSF','OverallQual','OverallCond','GarageArea','Fireplaces','BedroomAbvGr','KitchenAbvGr']
df_train_num= df_train[predictors]
df_train_id=df_train[['Id']]
df_train_sale= df_train[['SalePrice']]
imp=Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
regmod=LinearRegression()
steps=[('imputation', imp),('regress', regmod)]

pipeline=Pipeline(steps)
pipeline.fit(df_train_num,df_train_sale)
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_test_num= df_test[predictors]

df_test_id=df_test[['Id']]

print(df_test_num.columns,df_train_num.columns)
regmod_pred= pipeline.predict(df_test_num)
output =pd.DataFrame(regmod_pred, index = df_test['Id'], columns =['SalePrice'])

output.to_csv('output.csv', header = True, index_label = 'Id')