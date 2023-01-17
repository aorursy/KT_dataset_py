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

%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chart_studio.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head()
train.shape
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence','Id'], axis=1)
train.shape
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16.0
fig_size[1] = 4.0

x =train['SalePrice']
plt.hist(x, density=True, bins=400)
plt.ylabel('SalePrice');
def reject_outliers(SalePrice):
    filtered= [e for e in (train['SalePrice']) if (e < 450000)]
    return filtered

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16.0
fig_size[1] = 4.0

filtered = reject_outliers('SalePrice')
plt.hist(filtered, 50)
fig_size[0]=16.0
fig_size[1]=8.0
plt.show()

df_no_outliers = pd.DataFrame(filtered)
df_no_outliers.shape
train = train[train['SalePrice']<450000]
train.head()
X_train = train.drop(['SalePrice'], axis=1)

X_train.shape
Y_labels = train['SalePrice']
Y_labels.shape
X_train.info()
# 39 categorical values remain, the rest are numerical. These are the categrocial values:
cat_values= ['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope',
             'Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
             'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',
             'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir',
             'Electrical','KitchenQual','Functional','GarageType','GarageFinish','GarageQual','GarageCond',
             'PavedDrive','SaleType','SaleCondition']
X_train = X_train.apply(lambda x:x.fillna(x.value_counts().index[0])) #= fills every numercial column with its own most frequent value

X_train = X_train.fillna(X_train['GarageFinish'].value_counts().index[0]) #fill NaNs with the most frequent value from that column.
X_train = X_train.fillna(X_train['BsmtQual'].value_counts().index[0])
X_train = X_train.fillna(X_train['GarageType'].value_counts().index[0])
X_train = X_train.fillna(X_train['GarageQual'].value_counts().index[0])
X_train = X_train.fillna(X_train['GarageCond'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtCond'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtExposure'].value_counts().index[0])
X_train = X_train.fillna(X_train['BsmtFinType1'].value_counts().index[0])
X_train = X_train.fillna(X_train['FireplaceQu'].value_counts().index[0])
X_train = pd.get_dummies(X_train, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour',
                                           'Utilities','LotConfig','LandSlope','Neighborhood','Condition1',
                                           'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                                           'Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',
                                           'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                                           'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical',
                                           'KitchenQual','Functional','GarageType','GarageFinish','GarageQual',
                                           'GarageCond','PavedDrive','SaleType','SaleCondition'])  
X_train.describe()
X_train = X_train.drop(['Condition2_RRAe','Exterior2nd_Other','Condition2_RRAn','Condition2_RRNn',
                        'HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal',
                        'RoofMatl_Roll','Exterior1st_ImStucc','Heating_Floor','Heating_OthW',
                        'Electrical_Mix','GarageQual_Ex', 'Exterior1st_Stone','Utilities_NoSeWa'], axis=1)
X_train.shape
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
test.shape
total = test.isnull().sum().sort_values(ascending=False)
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
test = test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)
test.shape
test = test.apply(lambda x:x.fillna(x.value_counts().index[0])) #= fills every column with its own most frequent value

test = test.fillna(test['GarageFinish'].value_counts().index[0]) #fill NaNs with the most frequent value from that column.
test = test.fillna(test['BsmtQual'].value_counts().index[0])
test = test.fillna(test['FireplaceQu'].value_counts().index[0])
test = test.fillna(test['GarageType'].value_counts().index[0])
test = test.fillna(test['GarageQual'].value_counts().index[0])
test = test.fillna(test['GarageCond'].value_counts().index[0])
test = test.fillna(test['GarageFinish'].value_counts().index[0])
test = test.fillna(test['BsmtCond'].value_counts().index[0])
test = test.fillna(test['BsmtExposure'].value_counts().index[0])
test = test.fillna(test['BsmtFinType1'].value_counts().index[0])
test = test.fillna(test['BsmtFinType2'].value_counts().index[0])
test = test.fillna(test['BsmtUnfSF'].value_counts().index[0])
test = pd.get_dummies(test, columns=['FireplaceQu','MSZoning','Street','LotShape','LandContour','Utilities',
                                           'LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
                                           'BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st',
                                           'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
                                           'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                                           'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                                           'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                           'SaleType','SaleCondition'])
test.shape
test.head()
# Removing the feature 'Id' before implementing the model
X_test = test.drop(['Id'], axis=1)

X_test.shape
from xgboost import XGBRegressor
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_clf.fit(X_train, Y_labels)
from sklearn.model_selection import cross_val_score

xgb_clf_cv = cross_val_score(xgb_clf,X_train, Y_labels, cv=10, ) 

print(xgb_clf_cv.mean())
xgb_clf = XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_clf.fit(X_train, Y_labels)
xgb_predictions_test = xgb_clf.predict(X_test) 

xgb_predictions_test
xgb_predictions_test.shape
submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": xgb_predictions_test
    })

submission.to_csv("kaggleXGB_HousePrices.csv", index=False)

from IPython.display import FileLink
FileLink('kaggleXGB_HousePrices.csv')
