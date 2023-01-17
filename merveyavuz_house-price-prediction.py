# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
train.head()
train.shape
train.info()
null_counts = train.isnull().sum()
null_counts[null_counts > 0]
train.describe()
train._get_numeric_data().columns
train._get_numeric_data().iplot(kind="histogram", bins=50)
train.corr().iplot(kind='heatmap',colorscale="Blues", title="Feature Correlation Matrix")
bp = train[['YearBuilt', 'SalePrice']]
bp.pivot(columns='YearBuilt', values='SalePrice').iplot(kind='box')
train.iplot(asFigure=True, x='KitchenQual', y='SalePrice', mode='markers')
train['1stFlrSF'].describe()
train['SaleType'].iplot(kind="histogram", bins=20, theme="white", title="SaleType",xTitle='Sale Type', yTitle='Count')
bp2 = train[['Street', 'SalePrice']]
bp2.pivot(columns='Street', values='SalePrice').iplot(kind='box')
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
fig = make_subplots(rows=2, cols=2)
fig.add_trace(
    go.Scatter(x=list(train['GrLivArea']), y=list(train['SalePrice']),mode="markers+text"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=list(train['TotalBsmtSF']), y=list(train['SalePrice']) ,mode="markers+text"),
    row=1, col=2 
)
fig.add_trace(
    go.Scatter(x=list(train['1stFlrSF']), y=list(train['SalePrice']),mode="markers+text"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=list(train['LotArea']), y=list(train['SalePrice']) ,mode="markers+text"),
    row=2, col=2 
)

fig.update_layout(height=600, width=800, title_text="Before Removing Outliers")
fig.show()
train.shape
train = train[(np.abs(stats.zscore(train.select_dtypes(exclude='object'))) < 3).all(axis=1)]
fig = make_subplots(rows=2, cols=2)
fig.add_trace(
    go.Scatter(x=list(train['GrLivArea']), y=list(train['SalePrice']),mode="markers+text"),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=list(train['TotalBsmtSF']), y=list(train['SalePrice']) ,mode="markers+text"),
    row=1, col=2 
)
fig.add_trace(
    go.Scatter(x=list(train['1stFlrSF']), y=list(train['SalePrice']),mode="markers+text"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=list(train['LotArea']), y=list(train['SalePrice']) ,mode="markers+text"),
    row=2, col=2 
)

fig.update_layout(height=600, width=800, title_text="After Removing Outliers")
fig.show()
train.shape
cor_train = train.select_dtypes(include=[np.number])
cor_train = cor_train.drop('Id', 1)
cor_train.shape
corrDf = cor_train.corr()
corrDf.corr().iplot(kind='heatmap',colorscale="Blues", title="Feature Correlation Matrix")
top_features = corrDf.index[abs(corrDf['SalePrice']>0.5)]
top_corr = train[top_features].corr()
top_corr.corr().iplot(kind='heatmap',colorscale="Blues", title="Feature Correlation Matrix")
train_corr = train.corr()
train_corr_sorted = train_corr.sort_values(['SalePrice'], ascending=False)
corrSp = pd.DataFrame()
corrSp['Label'] = train_corr_sorted.SalePrice.index
corrSp['Corr'] = train_corr_sorted.SalePrice.values
fig = px.pie(corrSp, values='Corr', names='Label', title='Sale Price Correlation chart')
fig.show()
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
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
GBR.fit(X_train, y_train)
print("Accuracy: ", GBR.score(X_test, y_test)*100)
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
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
submissionPredicts  = GBR.predict(test)
submissionPredicts
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['SalePrice'] = submissionPredicts
submission.to_csv('submission.csv',index=False)