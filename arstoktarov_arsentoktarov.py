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
import seaborn as sns

import matplotlib.pyplot as plt

import cufflinks as cf

cf.go_offline()

import plotly.express as px

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

import os

from plotly.subplots import make_subplots

from plotly import tools 

import plotly.graph_objects as go

from scipy import stats
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
test.shape
train.shape
train.info()
train.isnull().sum()
test.isnull().sum()
train.corr().iplot(kind='heatmap')
train['SaleType'].iplot(kind="histogram", bins=20, theme="white", title="SaleType",xTitle='Sale Type', yTitle='Count')
def mostCommon(columnName):

    if train[columnName].value_counts().index[0] == 'None':

        return train[columnName].value_counts().index[1]

    else:

        return train[columnName].value_counts().index[0]
train['Alley'] = train['Alley'].fillna(mostCommon('Alley'))

train['Electrical'] = train['Electrical'].fillna(mostCommon('Electrical'))

train['MasVnrType'] = train['MasVnrType'].replace(to_replace='None', value=mostCommon('MasVnrType'))

train['LotFrontage']= train.LotFrontage.fillna(train.LotFrontage.mean())

train['BsmtQual'] = train['BsmtQual'].fillna(mostCommon('BsmtQual'))

train['MasVnrType'] = train['MasVnrType'].fillna(mostCommon('MasVnrType'))

train['MasVnrArea']= train.MasVnrArea.fillna(train.MasVnrArea.mean())

train['BsmtCond'] = train['BsmtCond'].fillna(mostCommon('BsmtCond'))

train['BsmtExposure'] = train['BsmtExposure'].fillna(mostCommon('BsmtExposure'))

train['BsmtFinType1'] = train['BsmtFinType1'].fillna(mostCommon('BsmtFinType1'))

train['BsmtFinType2'] = train['BsmtFinType2'].fillna(mostCommon('BsmtFinType2'))

train['Electrical'] = train['Electrical'].fillna(mostCommon('Electrical'))

train['FireplaceQu'] = train['FireplaceQu'].fillna(mostCommon('FireplaceQu'))

train['GarageType'] = train['GarageType'].fillna(mostCommon('GarageType'))

train['GarageYrBlt'] = train['GarageYrBlt'].fillna(mostCommon('GarageYrBlt'))

train['GarageFinish'] = train['GarageFinish'].fillna(mostCommon('GarageFinish'))

train['GarageQual'] = train['GarageQual'].fillna(mostCommon('GarageQual'))

train['GarageCond'] = train['GarageCond'].fillna(mostCommon('GarageCond'))

train['PoolQC'] = train['PoolQC'].fillna(mostCommon('PoolQC'))

train['Fence'] = train['Fence'].fillna(mostCommon('Fence'))

train['MiscFeature'] = train['MiscFeature'].fillna(mostCommon('MiscFeature'))

train = train.drop(['Utilities'], axis=1)
train.shape
train = train[(np.abs(stats.zscore(train.select_dtypes(exclude='object'))) < 3).all(axis=1)]
train.shape
cor_train = train.select_dtypes(include=[np.number])

cor_train = cor_train.drop('Id', 1)
corrDf = cor_train.corr()
categorical_list = []

for i in train.columns.tolist():

    if train[i].dtype=='object':

        categorical_list.append(i)

labelCols = categorical_list + ['MSSubClass', 'MasVnrArea', 'MoSold', 'OverallCond', 'YrSold']
for c in labelCols:

    lbl = LabelEncoder() 

    lbl.fit(list(train[c].values)) 

    train[c] = lbl.transform(list(train[c].values))
salePrice = train['SalePrice']

train = train.drop(['SalePrice'], axis=1)

X = train.values

y = salePrice.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
test['Alley'] = test['Alley'].fillna(mostCommon('Alley'))

test['Electrical'] = test['Electrical'].fillna(mostCommon('Electrical'))

test['MasVnrType'] = test['MasVnrType'].replace(to_replace='None', value=mostCommon('MasVnrType'))

test['LotFrontage']= test.LotFrontage.fillna(test.LotFrontage.mean())

test['BsmtQual'] = test['BsmtQual'].fillna(mostCommon('BsmtQual'))

test['MasVnrType'] = test['MasVnrType'].fillna(mostCommon('MasVnrType'))

test['MasVnrArea']= test.MasVnrArea.fillna(test.MasVnrArea.mean())

test['BsmtCond'] = test['BsmtCond'].fillna(mostCommon('BsmtCond'))

test['BsmtExposure'] = test['BsmtExposure'].fillna(mostCommon('BsmtExposure'))

test['BsmtFinType1'] = test['BsmtFinType1'].fillna(mostCommon('BsmtFinType1'))

test['BsmtFinType2'] = test['BsmtFinType2'].fillna(mostCommon('BsmtFinType2'))

test['Electrical'] = test['Electrical'].fillna(mostCommon('Electrical'))

test['FireplaceQu'] = test['FireplaceQu'].fillna(mostCommon('FireplaceQu'))

test['GarageType'] = test['GarageType'].fillna(mostCommon('GarageType'))

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(mostCommon('GarageYrBlt'))

test['GarageFinish'] = test['GarageFinish'].fillna(mostCommon('GarageFinish'))

test['GarageQual'] = test['GarageQual'].fillna(mostCommon('GarageQual'))

test['GarageCond'] = test['GarageCond'].fillna(mostCommon('GarageCond'))

test['PoolQC'] = test['PoolQC'].fillna(mostCommon('PoolQC'))

test['Fence'] = test['Fence'].fillna(mostCommon('Fence'))

test['MiscFeature'] = test['MiscFeature'].fillna(mostCommon('MiscFeature'))



test['BsmtFinSF1']= test.BsmtFinSF1.fillna(test.BsmtFinSF1.mean())

test['BsmtFinSF2']= test.BsmtFinSF2.fillna(test.BsmtFinSF2.mean())

test['BsmtUnfSF']= test.BsmtUnfSF.fillna(test.BsmtUnfSF.mean())

test['TotalBsmtSF']= test.TotalBsmtSF.fillna(test.TotalBsmtSF.mean())

test['BsmtFullBath'] = test['BsmtFullBath'].fillna(mostCommon('BsmtFullBath'))

test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(mostCommon('BsmtHalfBath'))

test['GarageCars'] = test['GarageCars'].fillna(mostCommon('GarageCars'))

test['GarageArea']= test.GarageArea.fillna(test.GarageArea.mean())



test = test.drop(['Utilities'], axis=1)
for c in labelCols:

    lbl = LabelEncoder() 

    lbl.fit(list(test[c].values)) 

    test[c] = lbl.transform(list(test[c].values))

GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)

GBR.fit(X_train, y_train)

print("Accuracy: ", GBR.score(X_test, y_test)*100)
submissionPredicts  = GBR.predict(test)
submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = submissionPredicts

submission.to_csv('submission.csv',index=False)