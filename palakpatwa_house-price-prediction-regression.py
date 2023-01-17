import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/train.csv')
plt.figure(figsize = (18,8))
sns.distplot(df.SalePrice)
plt.show()
df.Alley.value_counts()
df['Alley'] = df.Alley.fillna('No alley access') 
df[['BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].head(50)
df.BsmtQual.value_counts()
df.BsmtQual = df.BsmtQual.fillna('No Basement')
df.BsmtCond = df.BsmtCond.fillna('No Basement')
df.BsmtExposure = df.BsmtExposure.fillna('No Basement')
df.BsmtFinType1 = df.BsmtFinType1.fillna('No Basement')
df.BsmtFinType2 = df.BsmtFinType2.fillna('No Basement')

df.isnull().sum().tail(50)
df[['GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond']].head(50)
df[df.GarageArea == 0][['GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond']]
df.GarageType = df.GarageType.fillna('No Garage')
df.GarageFinish = df.GarageFinish.fillna('No Garage')
df.GarageQual = df.GarageQual.fillna('No Garage')
df.GarageCond = df.GarageCond.fillna('No Garage')
df.GarageYrBlt = df.GarageYrBlt.fillna(0)
df[['MiscFeature','MiscVal']].head(50)
df.MiscFeature.loc[df.MiscVal == 0]
df.MiscFeature.loc[df.MiscVal == 0] = df.MiscFeature.loc[df.MiscVal == 0].fillna('None')
df[['Fireplaces', 'FireplaceQu' ]].head(50)
df.FireplaceQu[df.Fireplaces == 0] = df.FireplaceQu[df.Fireplaces == 0].fillna('N0 Fireplace')
df.isnull().sum().tail(50)
df[['PoolArea','PoolQC' ]]
df.PoolQC[df.PoolArea == 0] = df.PoolQC[df.PoolArea == 0].fillna('No Pool')
df.isnull().sum().head(50)
df.Fence = df.Fence.fillna('No fence')
final_df = df[~np.isnan(df.LotFrontage)]
final_df[['MasVnrType', 'MasVnrArea']]
final_df[['Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation']].head(50)
final_df = df.dropna()
final_df.shape
final_df.info()
final_df.describe()
final_df['MSSubClass']= final_df['MSSubClass'].astype('object')

final_df.MSSubClass
num_var = final_df.select_dtypes(include=['float64', 'int64'])
num_var.head()
cat_var =final_df.select_dtypes(include=['object'])
cat_var.head()
cat_dum = pd.get_dummies(cat_var, drop_first=True)
cat_dum.head()
plt.figure(figsize = (30,20))
sns.heatmap(num_var.drop('Id', 1).corr(), annot = True)
plt.show()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
col = num_var.drop('Id', 1).columns
num_var = scale.fit_transform(num_var.drop('Id', 1))
num_var = pd.DataFrame(num_var)
num_var.columns = col
num_var.shape
cat_dum.shape
num_var
final_df = pd.concat([final_df, cat_dum], axis =1)
master_df = final_df.drop(list(cat_var.columns), axis =1)
master_df[list(num_var.columns)] = scale.fit_transform(num_var)
master_df.shape
master_df.info()
master_df.describe()
master_df.isnull().sum().max()
x = master_df.drop(['Id', 'SalePrice'], axis = 1)
y = master_df['SalePrice']
y.head()
from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, test_size= 0.3, random_state = 100)
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

params = {'alpha' : [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
ridge = Ridge()
folds = 5

model = GridSearchCV(estimator = ridge,
                    scoring = 'neg_mean_absolute_error',
                    return_train_score = True,
                    param_grid= params,
                    cv = folds,
                    verbose = 1)

model.fit(x_train,y_train)
help(GridSearchCV)
res =pd.DataFrame(model.cv_results_)
res.head()
plt.figure(figsize = (16,8))
plt.plot(res.param_alpha, res.mean_test_score)
plt.plot(res.param_alpha, res.mean_train_score)
plt.xlabel('alpha')
plt.ylabel('mean neg error')
plt.legend(['test','train'])
plt.show()
model.best_params_
alpha = 50 
ridge = Ridge(alpha = alpha)

ridge.fit(x_train,y_train)
ridge.coef_
lasso = Lasso()
folds = 5

model2 = GridSearchCV(estimator = lasso,
                    scoring = 'neg_mean_absolute_error',
                    return_train_score = True,
                    param_grid= params,
                    cv = folds,
                    verbose = 1)

model2.fit(x_train,y_train)
res2 =pd.DataFrame(model.cv_results_)
res2.head()
plt.figure(figsize = (16,8))
plt.plot(res2.param_alpha, res2.mean_test_score)
plt.plot(res2.param_alpha, res2.mean_train_score)
plt.xlabel('alpha')
plt.ylabel('mean neg error')
plt.legend(['test','train'])
plt.show()
model2.best_params_
alpha = 0.001
lasso = Lasso(alpha = alpha)

lasso.fit(x_train,y_train)
lasso.coef_
y_pred = lasso.predict(x_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['sale price prediction']
from sklearn.metrics import accuracy_score, regression , r2_score
r2_score(y_pred, y_test)        ############lasso on test set
y_pred2 = ridge.predict(x_test)
y_pred2 = pd.DataFrame(y_pred2)
r2_score(y_pred2 , y_test)            ##############ridge on test set
y_train_pred = lasso.predict(x_train)
y_train_pred = pd.DataFrame(y_train_pred)
y_train_pred.head()
r2_score(y_train_pred, y_train)          ###############lasso predict on train set
ytrain = ridge.predict(x_train)
ytrain = pd.DataFrame(ytrain)
ytrain
r2_score(ytrain, y_train)          ######ridge predicton on train set

