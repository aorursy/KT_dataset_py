# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()

# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(df_train.SalePrice,kde = False)
X = df_train.drop(['SalePrice','Id'],axis = 1)

y = np.log(df_train.SalePrice)

X_test = df_test.drop('Id',axis= 1)
X.head()

print(X.shape,X_test.shape)
full_data = pd.concat([X,X_test],axis = 0)
full_data.shape
full_data.isnull().sum().apply(lambda x : x/2919*100).sort_values(ascending = False).head(35)
missin_cols = [col for col in full_data.columns if full_data[col].isnull().sum() > 0]
for col in missin_cols:

    if full_data[col].dtype == object:

        full_data[col] = full_data[col].fillna('no')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):

    full_data[col] = full_data[col].fillna(0)
full_data.LotFrontage = full_data.LotFrontage.fillna(full_data.LotFrontage.median()) 
for col in ('Electrical','MSZoning','Functional','KitchenQual','Exterior1st','Exterior2nd','SaleType'):

    full_data[full_data[col] == 'no'].loc[:,col] = full_data[col].mode()[0]
full_data.isnull().sum().sort_values(ascending = False).head()
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
X_train = full_data[:1460]

y_train = df_train.SalePrice
cat_cols = [col for col in X_train.columns if X_train[col].dtype == object]

num_cols = [col for col in X_train.columns if X_train[col].dtype != object]

print(len(cat_cols),len(num_cols),len(X_train.columns))
# fig,axs = plt.subplots(10,4,figsize = (25,40))

# for i in range(10):

#     for j in range(4):

#         sns.scatterplot(y = y_train, x = X_train[cat_cols[i*4+j]],ax = axs[i,j],alpha = 0.4)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    le = LabelEncoder() 

    le.fit(list(full_data[c].values)) 

    full_data[c] = le.transform(list(full_data[c].values))

corr = df_train.corr()

plt.subplots(figsize = (12,12))

sns.heatmap(corr,square=True)
low_card_cols = [col for col in cat_cols if full_data[col].nunique() < 10]
high_corr_cols = list(np.abs(corr)['SalePrice'].nlargest(32).index)
high_corr_cols.append('TotalSF')

high_corr_cols.remove('SalePrice')
high_corr_cols
reduced_data = full_data[low_card_cols+high_corr_cols]
full_data = pd.get_dummies(full_data)
reduced_data = pd.get_dummies(reduced_data)
reduced_data.shape
X_r = reduced_data[:1460]

X_test_r = reduced_data[1460:]
X = full_data[:1460]

X_test = full_data[1460:]
print(X.shape,X_test.shape)
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

import xgboost as xgb
def score(model):

    rmse= np.sqrt(-cross_val_score(model, X_r, y_train, scoring="neg_mean_squared_error", cv = 5))

    return rmse.mean()
model_1 = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1100,verbose=False)
model_2 = RandomForestRegressor(n_estimators= 1000,random_state=42)
score(model_1)
model_1.fit(X_r,y_train)

preds = model_1.predict(X_test_r)

#preds = np.e**preds
submissions = pd.DataFrame({'Id' : df_test.Id,'SalePrice' : preds})

submissions.to_csv('submission.csv',index = False)