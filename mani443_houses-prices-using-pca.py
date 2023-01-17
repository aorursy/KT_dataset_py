import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
# Load the train data in a dataframe

train = pd.read_csv("../input/train.csv")



# Load the test data in a dataframe

test = pd.read_csv("../input/test.csv")
train.info()
train.head()
nulls = train.isnull().sum().sort_values(ascending=False)

nulls.head(20)
train = train.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)
train[['Fireplaces','FireplaceQu']].head(10)
train['FireplaceQu'].isnull().sum()
train['Fireplaces'].value_counts()
train['FireplaceQu']=train['FireplaceQu'].fillna('NF')
train['LotFrontage'] =train['LotFrontage'].fillna(value=train['LotFrontage'].mean())
train['GarageType'].isnull().sum()
train['GarageCond'].isnull().sum()
train['GarageFinish'].isnull().sum()
train['GarageYrBlt'].isnull().sum()
train['GarageQual'].isnull().sum()
train['GarageArea'].value_counts().head()
train['GarageType']=train['GarageType'].fillna('NG')

train['GarageCond']=train['GarageCond'].fillna('NG')

train['GarageFinish']=train['GarageFinish'].fillna('NG')

train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')

train['GarageQual']=train['GarageQual'].fillna('NG')
train.BsmtExposure.isnull().sum()
train.BsmtFinType2.isnull().sum()
train.BsmtFinType1.isnull().sum()
train.BsmtCond.isnull().sum() 
train.BsmtQual.isnull().sum()
train.TotalBsmtSF.value_counts().head()
train['BsmtExposure']=train['BsmtExposure'].fillna('NB')

train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')

train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')

train['BsmtCond']=train['BsmtCond'].fillna('NB')

train['BsmtQual']=train['BsmtQual'].fillna('NB')
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['MasVnrType'] = train['MasVnrType'].fillna('none')
train.Electrical = train.Electrical.fillna('SBrkr')
train.isnull().sum().sum()
num_train = train._get_numeric_data()
num_train.columns
def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



num_train.apply(lambda x: var_summary(x)).T

sns.boxplot([num_train.LotFrontage])
train['LotFrontage']= train['LotFrontage'].clip_upper(train['LotFrontage'].quantile(0.99)) 
sns.boxplot(num_train.LotArea)
train['LotArea']= train['LotArea'].clip_upper(train['LotArea'].quantile(0.99)) 
sns.boxplot(train['MasVnrArea'])
train['MasVnrArea']= train['MasVnrArea'].clip_upper(train['MasVnrArea'].quantile(0.99))
sns.boxplot(train['BsmtFinSF1']) 
sns.boxplot(train['BsmtFinSF2']) 
train['BsmtFinSF1']= train['BsmtFinSF1'].clip_upper(train['BsmtFinSF1'].quantile(0.99)) 

train['BsmtFinSF2']= train['BsmtFinSF2'].clip_upper(train['BsmtFinSF2'].quantile(0.99)) 
sns.boxplot(train['TotalBsmtSF'])
train['TotalBsmtSF']= train['TotalBsmtSF'].clip_upper(train['TotalBsmtSF'].quantile(0.99))
sns.boxplot(train['1stFlrSF'])
train['1stFlrSF']= train['1stFlrSF'].clip_upper(train['1stFlrSF'].quantile(0.99))
sns.boxplot(train['2ndFlrSF'])
train['2ndFlrSF']= train['2ndFlrSF'].clip_upper(train['2ndFlrSF'].quantile(0.99))
sns.boxplot(train['GrLivArea'])
train['GrLivArea']= train['GrLivArea'].clip_upper(train['GrLivArea'].quantile(0.99))
sns.boxplot(train['BedroomAbvGr'])
train['BedroomAbvGr']= train['BedroomAbvGr'].clip_upper(train['BedroomAbvGr'].quantile(0.99))

train['BedroomAbvGr']= train['BedroomAbvGr'].clip_lower(train['BedroomAbvGr'].quantile(0.01))
sns.boxplot(train['GarageCars'])
train['GarageCars']= train['GarageCars'].clip_upper(train['GarageCars'].quantile(0.99))
sns.boxplot(train['GarageArea'])
train['GarageArea']= train['GarageArea'].clip_upper(train['GarageArea'].quantile(0.99))
sns.boxplot(train['WoodDeckSF'])
train['WoodDeckSF']= train['WoodDeckSF'].clip_upper(train['WoodDeckSF'].quantile(0.99))
sns.boxplot(train['OpenPorchSF'])
train['OpenPorchSF']= train['OpenPorchSF'].clip_upper(train['OpenPorchSF'].quantile(0.99))
sns.boxplot(train['EnclosedPorch'])
train['EnclosedPorch']= train['EnclosedPorch'].clip_upper(train['EnclosedPorch'].quantile(0.99))
sns.boxplot(train['3SsnPorch'])
train['3SsnPorch']= train['3SsnPorch'].clip_upper(train['3SsnPorch'].quantile(0.99))
sns.boxplot(train['ScreenPorch'])
train['ScreenPorch']= train['ScreenPorch'].clip_upper(train['ScreenPorch'].quantile(0.99))
sns.boxplot(train['PoolArea'])
train['PoolArea']= train['PoolArea'].clip_upper(train['PoolArea'].quantile(0.99))
sns.boxplot(train['MiscVal'])
sns.boxplot(train.SalePrice)
train['SalePrice']= train['SalePrice'].clip_upper(train['SalePrice'].quantile(0.99))

train['SalePrice']= train['SalePrice'].clip_lower(train['SalePrice'].quantile(0.01))
train['MiscVal']= train['MiscVal'].clip_upper(train['MiscVal'].quantile(0.99))
num_corr=num_train .corr()

plt.subplots(figsize=(13,10))

sns.heatmap(num_corr,vmax =.8 ,square = True)
k = 14

cols = num_corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(num_train[cols].values.T)

sns.set(font_scale=1.35)

f, ax = plt.subplots(figsize=(10,10))

hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values)
from sklearn.preprocessing import StandardScaler
train_d = pd.get_dummies(train)

train_d1 = train_d.drop(['SalePrice'],axis = 1)

y = train_d.SalePrice

scaler = StandardScaler()

scaler.fit(train_d1)                

t_train = scaler.transform(train_d1)



from sklearn.decomposition import PCA
pca_hp = PCA(30)

x_fit = pca_hp.fit_transform(t_train)

np.exp(pca_hp.explained_variance_ratio_)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(x_fit,y)
X_train , X_test, Y_train, Y_test = train_test_split(

        x_fit,

        y,

        test_size=0.20,

        random_state=123)
y_pred = linear.predict(X_test)
from sklearn import metrics
metrics.r2_score(Y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))

rmse
from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor, export_graphviz, export 

from sklearn.grid_search import GridSearchCV
depth_list = list(range(1,20))

for depth in depth_list:

    dt_obj = DecisionTreeRegressor(max_depth=depth)

    dt_obj.fit(X_train, Y_train)

    print ('depth:', depth, 'R_squared:', metrics.r2_score(Y_test, dt_obj.predict(X_test)))
param_grid = {'max_depth': np.arange(3,20)}

tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10)

tree.fit(X_train, Y_train)
tree.best_params_
tree.best_score_
tree_final = DecisionTreeRegressor(max_depth=5)

tree_final.fit(X_train, Y_train)
tree_test_pred = pd.DataFrame({'actual': Y_test, 'predicted': tree_final.predict(X_test)})
metrics.r2_score(Y_test, tree_test_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))

rmse
from sklearn.ensemble import RandomForestRegressor
depth_list = list(range(1,20))

for depth in depth_list:

    dt_obj = RandomForestRegressor(max_depth=depth)

    dt_obj.fit(X_train, Y_train)

    print ('depth:', depth, 'R_Squared:', metrics.r2_score(Y_test, dt_obj.predict(X_test)))
radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100,max_depth = 11)

radm_clf.fit( X_train, Y_train )
radm_test_pred = pd.DataFrame( { 'actual':  Y_test,

                            'predicted': radm_clf.predict( X_test ) } )
metrics.r2_score( radm_test_pred.actual, radm_test_pred.predicted )
rmse = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))

rmse
from sklearn.ensemble import BaggingRegressor
param_bag = {'n_estimators': list(range(100, 801, 100)),

             }
from sklearn.grid_search import GridSearchCV

bag_cl = GridSearchCV(estimator=BaggingRegressor(),

                  param_grid=param_bag,

                  cv=5,

                  verbose=True, n_jobs=-1)
bag_cl.get_params()
bag_cl.fit(X_train, Y_train)
bag_cl.best_params_
bagclm = BaggingRegressor(oob_score=True, n_estimators=700)

bagclm.fit(X_train, Y_train)
y_pred = pd.DataFrame( { 'actual':  Y_test,

                            'predicted': bagclm.predict( X_test) } )
metrics.r2_score(y_pred.actual, y_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred.predicted))

rmse
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
[10**x for x in range(-3, 3)]
paragrid_ada = {'n_estimators': [100, 200, 400, 600, 800],

               'learning_rate': [10**x for x in range(-3, 3)]}
from sklearn.grid_search import GridSearchCV

ada = GridSearchCV(estimator=AdaBoostRegressor(),

                  param_grid=paragrid_ada,

                  cv=5,

                  verbose=True, n_jobs=-1)
ada.fit(X_train, Y_train)
ada.best_params_
ada_clf = AdaBoostRegressor(learning_rate=0.1, n_estimators=600)
ada_clf.fit(X_train, Y_train)
ada_test_pred = pd.DataFrame({'actual': Y_test,

                            'predicted': ada_clf.predict(X_test)})
metrics.r2_score(ada_test_pred.actual, ada_test_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred.predicted))

rmse
param_test1 = {'n_estimators': [100, 200, 400, 600, 800],

              'max_depth': list(range(1,10))}

gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,

                                                               max_features='sqrt',subsample=0.8,verbose = 0), 

                        param_grid = param_test1, scoring='r2',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, Y_train)
gsearch1.best_params_
gbm = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,max_depth=5, n_estimators=400,

                                                               max_features='sqrt',subsample=0.8, random_state=10)
gbm.fit(X_train, Y_train)
gbm_test_pred = pd.DataFrame({'actual': Y_test,

                            'predicted': gbm.predict(X_test)})
metrics.r2_score(gbm_test_pred.actual, gbm_test_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(gbm_test_pred.actual, gbm_test_pred.predicted))

rmse