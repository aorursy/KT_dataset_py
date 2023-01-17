import numpy as np 

import pandas as pd 

import warnings

import seaborn as sns

warnings.filterwarnings('ignore')

%matplotlib inline

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

missing_columns = [col for col in data.columns if data[col].isnull().any()]

total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(len(missing_columns))
data['PoolQC'].fillna("none", inplace = True)
sns.distplot(data['LotFrontage'],color='#FFD700')
data['LotFrontage'].fillna(data.LotFrontage.mean(), inplace = True)