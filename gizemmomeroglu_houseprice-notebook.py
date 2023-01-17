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
from scipy import stats
pd.set_option('display.max_columns', 100)
import seaborn as sns
import matplotlib.pyplot as plt
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test['SalePrice']=0
test['label']='test'
train['label']='train'
f=[test,train]
x=pd.concat(f)
x.head()
x.columns[train.isnull().any()]
x.PoolQC.value_counts()
x=x.drop('PoolQC',axis=1)
x.head()
x.info()
x.describe()
sns.distplot(train['SalePrice']);
x.GarageYrBlt.fillna(x.GarageYrBlt.mean(), inplace=True)
x.MasVnrArea.fillna(x.MasVnrArea.mean(), inplace=True)
x.LotFrontage.fillna(x.LotFrontage.mean(), inplace=True)
x.GarageArea.fillna(x.GarageArea.mean(), inplace=True)
x.GarageCars.fillna(x.GarageCars.mean(), inplace=True)
x.TotalBsmtSF.fillna(x.TotalBsmtSF.mean(), inplace=True)
x.BsmtUnfSF.fillna(x.BsmtUnfSF.mean(), inplace=True)
x.BsmtFinSF2.fillna(x.BsmtFinSF2.mean(), inplace=True)
x.BsmtFinSF1.fillna(x.BsmtFinSF1.mean(), inplace=True)
x.BsmtFullBath.fillna(x.BsmtFinSF1.mean(), inplace=True)
x.BsmtHalfBath.fillna(x.BsmtFinSF1.mean(), inplace=True)

x.MSZoning.fillna('notknown', inplace=True)
x.Utilities.fillna('notknown', inplace=True)
x.Exterior1st.fillna('notknown', inplace=True)
x.Exterior2nd.fillna('notknown', inplace=True)
x.BsmtFinType1.fillna('notknown', inplace=True)
x.BsmtFinType2.fillna('notknown', inplace=True)
x.Electrical.fillna('notknown', inplace=True)
x.KitchenQual.fillna('notknown', inplace=True)
x.Alley.fillna('notknown', inplace=True)
x.MasVnrType.fillna('notknown', inplace=True)
x.BsmtQual.fillna('notknown', inplace=True)
x.BsmtCond.fillna('notknown', inplace=True)
x.BsmtExposure.fillna('notknown', inplace=True)
x.Functional.fillna('notknown', inplace=True)
x.FireplaceQu.fillna('notknown', inplace=True)
x.GarageType.fillna('notknown', inplace=True)
x.GarageFinish.fillna('notknown', inplace=True)
x.GarageQual.fillna('notknown', inplace=True)
x.GarageCond.fillna('notknown', inplace=True)
x.Fence.fillna('notknown', inplace=True)
x.MiscFeature.fillna('notknown', inplace=True)
x.SaleType.fillna('notknown', inplace=True)
x.columns[x.isnull().any()]
x.SaleType.value_counts()
df2 =pd.get_dummies(data=x, columns=['MSZoning', 'Street','Alley','LotShape','LandContour', 'Utilities','LotConfig',
                                     'LandSlope','Neighborhood','Condition1','Condition2', 'BldgType','HouseStyle',
                                      'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
                                     'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                                     'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
                                    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
                                     'PavedDrive','Fence', 'MiscFeature', 'SaleType','SaleCondition'])
df2.MSZoning_RL
train1=df2[df2['label']=='train']
test1=df2[df2['label']=='test']
test1=test1.drop(['label','SalePrice'],axis=1)
train1=train1.drop('label',axis=1)
x=train1.isin(['BLQ']).any(1)
train1.tail()
train1.boxplot(column = 'SalePrice', by='YrSold')

train1.plot.scatter(x='GrLivArea', y='SalePrice');
fig = sns.boxplot(x='SaleCondition', y="SalePrice", data=train)
fig.axis(ymin=0, ymax=800000);
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
from sklearn.model_selection import train_test_split
X = train1.drop(['Id','SalePrice','GarageYrBlt'],axis=1)
y = train1['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
from sklearn import metrics
import numpy as np
def mean_absolute_percentage_error(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('MAPE: {}'.format(mean_absolute_percentage_error(y_test,y_pred)))
test1.columns[test1.isnull().any()]
X_testt = test1.drop(['Id','GarageYrBlt'],axis=1)
X_testt.columns[X_testt.isnull().any()]
predictions = lm.predict(X_testt)
predictions
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
def mean_absolute_percentage_error(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('MAPE: {}'.format(mean_absolute_percentage_error(y_test,predictions)))
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = predictions
submission
submission.to_csv('submission-1.csv', index=False)
