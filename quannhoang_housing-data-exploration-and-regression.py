# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.pipeline import make_pipeline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading data
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df = pd.concat([train_df,test_df],axis=0)

#Checking data columns
df.columns
df.info()
#Check null value count
nan_sorted = df.isnull().sum().sort_values(ascending=False).head(40).drop('SalePrice')
nan_sorted
#Get too high count of null values
top_nan_cols = nan_sorted.head(6).keys()
top_nan_cols
#Drop columns that has too high null ratio
df.drop(top_nan_cols, axis=1, inplace=True)
#Fill NaN 
df.fillna(value={'GarageQual':'NA',
                 'GarageCond':'NA',
                 'GarageType':'NA',
                 'GarageYrBlt':df['GarageYrBlt'].mean(),
                 'GarageFinish':'NA',
                 'Electrical':'Mix',
                 'BsmtFinType2':'NA',
                 'BsmtFinType1':'NA',
                 'BsmtExposure':'NA',
                 'BsmtCond':'NA',
                 'BsmtQual':'NA',
                 'MasVnrArea':df['MasVnrArea'].mean(),
                 'MasVnrType':'None',
                 'MSZoning':'RM',
                 'Utilities':'Allpub',
                 'BsmtFullBath':df['BsmtFullBath'].mean(),
                 'BsmtHalfBath':df['BsmtHalfBath'].mean(),
                 'Functional': 'Typ',
                 'TotalBsmtSF':df['TotalBsmtSF'].mean(),
                 'GarageArea':df['GarageArea'].mean(),
                 'BsmtFinSF2':df['BsmtFinSF2'].mean(),
                 'BsmtUnfSF':df['BsmtUnfSF'].mean(),
                 'SaleType': 'Oth',
                 'Exterior2nd': 'Other',
                 'Exterior1st': 'Other',
                 'KitchenQual': 'TA',
                 'GarageCars':df['GarageCars'].mean(),
                 'BsmtFinSF1':df['BsmtFinSF1'].mean(),
                }, inplace=True)

nan_sorted = df.isnull().sum().sort_values(ascending=False).head(20).drop('SalePrice')
nan_sorted
#Convert continuous category-columns to numerical value
ordinal_encoder = OrdinalEncoder()
cat_col_list = ['BldgType','HeatingQC','ExterQual','ExterCond','KitchenQual','GarageQual','GarageCond']
df_cat = df[cat_col_list]
df_cat_encoded = ordinal_encoder.fit_transform(df_cat)
#Put encoded category columns back to data frame
for i in range(len(cat_col_list)):
    df[cat_col_list[i]] = df_cat_encoded[:,i:i+1]
#Convert non-continuous category-columns to one hot encoding
df_cat = df.select_dtypes(include='object')

#Remove GarageYrBlt from one hot encoding
#df_cat.drop(['GarageYrBlt'], axis=1, inplace=True)

#Drop non-continuous category-columns from original data frame
df.drop(df_cat.keys(), axis=1, inplace=True)
for col in df_cat.keys():
    df_cat = pd.get_dummies(df_cat, columns = [col], prefix=[col])


df_processed = pd.concat([df, df_cat], axis=1)
df_processed
#Separate Train and test data
train_df = df_processed.iloc[:1460,:]
test_df = df_processed.iloc[1460:,:]
test_id = test_df['Id']
#Correlation data (only up to SalePrice, which has index value of 44)
corr = train_df.iloc[:,:44].corr()
#Correlation heatmap
f, x = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmax=0.9,square=True,annot=False)
corr['SalePrice'].sort_values(ascending=False)

train_df
#Drop correlated columns
train_df.drop(['MSSubClass','TotRmsAbvGrd','KitchenQual','BsmtFullBath','1stFlrSF','GarageQual','GarageArea','Id'],axis=1, inplace=True)
test_df.drop(['MSSubClass','TotRmsAbvGrd','KitchenQual','BsmtFullBath','1stFlrSF','GarageQual','GarageArea','Id'],axis=1, inplace=True)
corr = train_df.iloc[:,:36].corr()
non_corr_features = []
for col, value in (corr['SalePrice'].sort_values(ascending=False)).items():
    if value < 0.3 and value > -0.2:
        non_corr_features.append(col)
non_corr_features
test_train_df = train_df.drop(non_corr_features, axis=1)
#Train and validate set split
train_set, validate_set = train_test_split(test_train_df, test_size=0.2, random_state=20)
#Create data and label pairs
X_train = train_set.drop(['SalePrice'],axis=1)
y_train = train_set['SalePrice']

X_validate = validate_set.drop(['SalePrice'],axis=1)
y_validate = validate_set['SalePrice']
#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
lin_mse = mse(y_train, lin_reg.predict(X_train))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Linear Regression test performance
lin_test_mse = mse(y_validate.to_numpy(), lin_reg.predict(X_validate))
lin_test_rmse = np.sqrt(lin_test_mse)
lin_test_rmse
#Polynomial Regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

poly_lin_reg = LinearRegression()
poly_lin_reg.fit(X_poly, y_train)
lin_mse = mse(y_train, poly_lin_reg.predict(X_poly))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Polynomial linear regression test performance
poly = PolynomialFeatures(degree=2, include_bias=False)
X_validate_poly = poly.fit_transform(X_validate)

Poly_lin_test_mse = mse(y_validate.to_numpy(), poly_lin_reg.predict(X_validate_poly))
Poly_lin_test_rmse = np.sqrt(Poly_lin_test_mse)
Poly_lin_test_rmse
#Regularized Linear Regression with Elastic Net

elastic_net = ElasticNet(alpha=1, l1_ratio=1, tol=0.001)
elastic_net.fit(X_train, y_train)
lin_mse = mse(y_train, elastic_net.predict(X_train))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Elastic Net test performance

lin_mse = mse(y_validate, elastic_net.predict(X_validate))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Stochastic Gradient Descent

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', eta0=0.1)
sgd_reg.fit(X_train, y_train)
lin_mse = mse(y_train, sgd_reg.predict(X_train))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Stochastic Gradient Descent test performance
lin_mse = mse(y_validate, sgd_reg.predict(X_validate))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Random Forest Regressor
RFG = RandomForestRegressor(n_estimators=1000, max_leaf_nodes=200,n_jobs=-1)
RFG.fit(X_train,y_train)
RFG_mse = mse(y_train, RFG.predict(X_train))
RFG_rmse = np.sqrt(RFG_mse)
RFG_rmse
#Random Forest Regressor Test performance
RFG_mse = mse(y_validate, RFG.predict(X_validate))
RFG_rmse = np.sqrt(RFG_mse)
RFG_rmse
#Gradient boosting
gdb = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.05)
gdb.fit(X_train,y_train)
lin_mse = mse(y_train, gdb.predict(X_train))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Gradient boosting test performance
lin_mse = mse(y_validate, gdb.predict(X_validate))
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#Gradient boosting fine tune using Grid Search
param_dict = {'n_estimators':[500,700,100],
              'learning_rate':[0.01, 0.02, 0.03, 0.04, 0.05]
                }
n_iter=10
gdb = GradientBoostingRegressor()
model = GridSearchCV(gdb, param_dict)
model.fit(X_train,y_train)
print(model.best_estimator_.get_params())
lin_mse = mse(y_train, model.predict(X_train))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
lin_mse = mse(y_validate, model.predict(X_validate))
lin_rmse = np.sqrt(lin_mse)
lin_rmse
#Test set prediction
test_sale_price= model.predict(test_df.drop(non_corr_features, axis=1).drop(['SalePrice'], axis=1))
test_sale_price = pd.Series(test_sale_price)
frame = {'Id':test_id, 'SalePrice':test_sale_price}
result = pd.DataFrame(frame)
result.to_csv('submission.csv', index=False)
result.head()
