import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
#Loading the Data

train = pd.read_csv('../input/train.csv')
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
import statsmodels.api as sm

import statsmodels.formula.api as smf
train.info()
train.columns
s1 = set(train.columns)

s2 = set(['SalePrice'])
features = "+".join((set(s1)-s2))

features
train = train.rename(columns ={'1stFlrSF':'firstFlrSF','2ndFlrSF':'iindFlrSF','3SsnPorch':'iiiSsnPorch'})
lm=smf.ols('SalePrice~firstFlrSF+MasVnrType+GarageFinish+KitchenAbvGr+WoodDeckSF+LandContour+LandSlope+GarageCars+Street+Exterior1st+iindFlrSF+SaleCondition+Electrical+LotConfig+HeatingQC+PavedDrive+LotArea+BsmtUnfSF+RoofMatl+TotRmsAbvGrd+BsmtFullBath+ExterQual+BedroomAbvGr+EnclosedPorch+BsmtQual+BsmtFinSF2+GarageCond+HouseStyle+GrLivArea+PoolArea+Utilities+BsmtExposure+HalfBath+Condition1+YrSold+MasVnrArea+BldgType+MSZoning+Fireplaces+FireplaceQu+BsmtFinType1+YearBuilt+BsmtHalfBath+Heating+SaleType+BsmtCond+MSSubClass+ScreenPorch+OpenPorchSF+FullBath+BsmtFinSF1+MoSold+LowQualFinSF+GarageType+Exterior2nd+iiiSsnPorch+TotalBsmtSF+ExterCond+Neighborhood+OverallQual+GarageArea+LotShape+MiscVal+YearRemodAdd+OverallCond+BsmtFinType2+Condition2+CentralAir+LotFrontage+Functional+RoofStyle+GarageYrBlt+KitchenQual+Foundation+GarageQual',data = train).fit()
lm.summary()
imc = pd.DataFrame(lm.pvalues)

imc
best_features = imc[imc[0] <= 0.05].index

best_features
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
train['intercept'] = lm.params[0]
for i in range(18):

    print (vif(train[['firstFlrSF', 'WoodDeckSF', 'GarageCars', 'iindFlrSF', 'LotArea',

       'BsmtUnfSF', 'GrLivArea', 'PoolArea', 'Fireplaces', 'YearBuilt',

       'ScreenPorch', 'LowQualFinSF', 'TotalBsmtSF', 'OverallQual',

       'GarageArea', 'YearRemodAdd', 'OverallCond','intercept']].as_matrix(), i))
train_a = train[ ['GarageFinish','Exterior1st','SaleCondition', 'LotConfig', 'RoofMatl', 'ExterQual', 'BsmtQual',  'GarageCond',

        'BsmtExposure', 'Condition1','BldgType', 'MSZoning', 'SaleType','GarageType', 'Exterior2nd','Neighborhood', 'Condition2',

       'Functional', 'GarageYrBlt', 'KitchenQual','Foundation', 'GarageQual', 'WoodDeckSF', 'LotArea',

       'BsmtUnfSF', 'Fireplaces', 'YearBuilt','ScreenPorch', 'LowQualFinSF', 'TotalBsmtSF', 'OverallQual',

       'YearRemodAdd', 'OverallCond','SalePrice']]
best_train = train_a

best_train.info()
from sklearn.ensemble import RandomForestRegressor
train_d = pd.get_dummies(train)
numeric = train._get_numeric_data()

category = train.drop(numeric.columns,axis = 1)
train_dx = train_d.drop(["SalePrice"],axis = 1)

train_dy = train_d.SalePrice
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(

        train_dx,

        train_dy,

        test_size=0.20,

        random_state=123)
radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100 )

radm_clf.fit( X_train, Y_train )
indices = np.argsort(radm_clf.feature_importances_)[::-1]

feature_rank = pd.DataFrame( columns = ['rank', 'feature', 'importance'] )

for f in range(X_train.shape[1]):

    feature_rank.loc[f] = [f+1,

                         X_train.columns[indices[f]],

                         radm_clf.feature_importances_[indices[f]]]

f, ax = plt.subplots(figsize=(10,100))

sns.barplot( y = 'feature', x = 'importance', data = feature_rank, color = 'Yellow')

plt.show()
ff = feature_rank.head(30)

ff
list(ff.feature)
final_cols = train_d[['OverallQual','GrLivArea','GarageCars', 'TotalBsmtSF', 'BsmtFinSF1', 'firstFlrSF',

 'GarageArea', 'LotArea', 'YearBuilt', 'OpenPorchSF', 'FullBath', 'LotFrontage', 'BsmtUnfSF', 'YearRemodAdd',

 'OverallCond','iindFlrSF','MasVnrArea','GarageType_Detchd','WoodDeckSF','MoSold','BsmtQual_Gd','TotRmsAbvGrd',

 'Neighborhood_Edwards','KitchenAbvGr','MSZoning_RM','MSSubClass','BsmtQual_Ex','GarageType_Attchd',

'ExterQual_Ex','KitchenQual_Gd']]
data_x = final_cols

data_y = train.SalePrice

final_data = pd.concat([data_x,data_y],axis = 1)
feats = "+".join(data_x)

feats
import statsmodels.api as sm

import statsmodels.formula.api as smf
final_data = final_data.rename(columns ={'1stFlrSF':'firstFlrSF','2ndFlrSF':'iindFlrSF'})
lm=smf.ols('SalePrice~OverallQual+GrLivArea+GarageCars+TotalBsmtSF+BsmtFinSF1+firstFlrSF+GarageArea+LotArea+YearBuilt+OpenPorchSF+FullBath+LotFrontage+BsmtUnfSF+YearRemodAdd+OverallCond+iindFlrSF+MasVnrArea+GarageType_Detchd+WoodDeckSF+MoSold+BsmtQual_Gd+TotRmsAbvGrd+Neighborhood_Edwards+KitchenAbvGr+MSZoning_RM+MSSubClass+BsmtQual_Ex+GarageType_Attchd+ExterQual_Ex+KitchenQual_Gd',final_data).fit()
lm.summary()
lm.pvalues
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
final_data['intercept'] = lm.params[0]
final_data.columns
for i in range(31):

    print (vif(final_data[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'BsmtFinSF1',

       'firstFlrSF', 'GarageArea', 'LotArea', 'YearBuilt', 'OpenPorchSF',

       'FullBath', 'LotFrontage', 'BsmtUnfSF', 'YearRemodAdd', 'OverallCond',

       'iindFlrSF', 'MasVnrArea', 'GarageType_Detchd', 'WoodDeckSF', 'MoSold',

       'BsmtQual_Gd', 'TotRmsAbvGrd', 'Neighborhood_Edwards', 'KitchenAbvGr',

       'MSZoning_RM', 'MSSubClass', 'BsmtQual_Ex', 'GarageType_Attchd',

       'ExterQual_Ex', 'KitchenQual_Gd','intercept']].as_matrix(), i))
final_data = final_data.rename(columns ={'firstFlrSF':'1stFlrSF','iindFlrSF':'2ndFlrSF'})
final_data1 = final_data.drop(['GrLivArea', 'GarageCars', 'BsmtFinSF1', 'TotalBsmtSF',

       '1stFlrSF', 'GarageArea','YearBuilt','BsmtUnfSF','2ndFlrSF'],axis = 1)
import scipy.stats as stats
train.Neighborhood.value_counts()
nb1 = train.SalePrice[train.Neighborhood=='NAmes']

nb2 = train.SalePrice[train.Neighborhood=='CollgCr']

nb3 = train.SalePrice[train.Neighborhood=='Edwards']

nb4 = train.SalePrice[train.Neighborhood=='Somerst']

nb5 = train.SalePrice[train.Neighborhood=='Gilbert']

nb6 = train.SalePrice[train.Neighborhood=='NridgHt']

nb7 = train.SalePrice[train.Neighborhood=='Sawyer']

nb8 = train.SalePrice[train.Neighborhood=='NWAmes']

nb9 = train.SalePrice[train.Neighborhood=='SawyerW']

nb10 = train.SalePrice[train.Neighborhood=='BrkSide']

nb11 = train.SalePrice[train.Neighborhood=='Crawfor']

nb12= train.SalePrice[train.Neighborhood=='Mitchel']

nb13 = train.SalePrice[train.Neighborhood=='NoRidge']

nb14 = train.SalePrice[train.Neighborhood=='Timber']

nb15 = train.SalePrice[train.Neighborhood=='IDOTRR']

nb16 = train.SalePrice[train.Neighborhood=='ClearCr']

nb17 = train.SalePrice[train.Neighborhood=='StoneBr']

nb18 = train.SalePrice[train.Neighborhood=='SWISU']

nb19 = train.SalePrice[train.Neighborhood=='Blmngtn']

nb20 = train.SalePrice[train.Neighborhood=='MeadowV']

nb21 = train.SalePrice[train.Neighborhood=='BrDale']

nb22 = train.SalePrice[train.Neighborhood=='Veenker']

nb23 = train.SalePrice[train.Neighborhood=='NPkVill']

nb24 = train.SalePrice[train.Neighborhood=='Blueste']
stats.f_oneway(nb1,nb2,nb3,nb4,nb5,nb6,nb7,nb8,nb9,nb10,nb11,nb12,nb13,nb14,nb15,nb16,nb17,nb18,nb19,nb20,nb21,nb22,nb23,nb24)
train.GarageQual.value_counts()
gq1 = train.SalePrice[train.GarageQual=='TA']

gq2 = train.SalePrice[train.GarageQual=='NG']

gq3 = train.SalePrice[train.GarageQual=='Fa']

gq4 = train.SalePrice[train.GarageQual=='Gd']

gq5 = train.SalePrice[train.GarageQual=='Ex']

gq6 = train.SalePrice[train.GarageQual=='Po']
stats.f_oneway(gq1,gq2,gq3,gq4,gq5)
train.GarageCond.value_counts()
gc1 = train.SalePrice[train.GarageQual=='TA']

gc2 = train.SalePrice[train.GarageQual=='NG']

gc3 = train.SalePrice[train.GarageQual=='Fa']

gc4 = train.SalePrice[train.GarageQual=='Gd']

gc5 = train.SalePrice[train.GarageQual=='Po']

gc6 = train.SalePrice[train.GarageQual=='Ex']
stats.f_oneway(gc1,gc2,gc3,gc4,gc5)
train.BsmtExposure.value_counts()
be1 = train.SalePrice[train.BsmtExposure=="No"]

be2 = train.SalePrice[train.BsmtExposure=="Av"]

be3 = train.SalePrice[train.BsmtExposure=="Gd"]

be4 = train.SalePrice[train.BsmtExposure=="Mn"]

be5 = train.SalePrice[train.BsmtExposure=="NB"]
stats.f_oneway(be1,be2,be3,be4,be5)
test_data = pd.read_csv('../input/test.csv')
test1 = test_data[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 

       'MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'Condition1',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual',

       'Foundation', 'BsmtQual', 'BsmtExposure', 'CentralAir', 'FireplaceQu',

       'GarageFinish', 'GarageQual', 'GarageCond']]
nulls1 = test1.isnull().sum().sort_values(ascending = False)

nulls1
test1['FireplaceQu']=test1['FireplaceQu'].fillna('NF')

test1['GarageCond']=test1['GarageCond'].fillna('NG')

test1['GarageFinish']=test1['GarageFinish'].fillna('NG')

test1['GarageQual']=test1['GarageQual'].fillna('NG')

test1['BsmtExposure']=test1['BsmtExposure'].fillna('NB')

test1['BsmtQual'] = test1['BsmtQual'].fillna('NB')

test1['MasVnrArea'] = test1['MasVnrArea'].fillna(test1['MasVnrArea'].mean())

test1['MSZoning'] = test1['MSZoning'].fillna('RL')

test1['BsmtFinSF1'] = test1['BsmtFinSF1'].fillna(test1['BsmtFinSF1'].mean())

test1['TotalBsmtSF'] = test1['TotalBsmtSF'].fillna(test1['TotalBsmtSF'].mean())
test2 = test1._get_numeric_data()
def var_summary(x):

    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max()], 

                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])



test2.apply(lambda x: var_summary(x)).T

test3 = test1.drop(test2.columns,axis = 1)
test2['OverallQual']= test2['OverallQual'].clip_upper(test2['OverallQual'].quantile(0.99))

test2['OverallQual']= test2['OverallQual'].clip_lower(test2['OverallQual'].quantile(0.01))

test2['YearBuilt']= test2['YearBuilt'].clip_upper(test2['YearBuilt'].quantile(0.99))

test2['YearBuilt']= test2['YearBuilt'].clip_lower(test2['YearBuilt'].quantile(0.01))

test2['MasVnrArea']= test2['MasVnrArea'].clip_upper(test2['MasVnrArea'].quantile(0.99))

test2['BsmtFinSF1']= test2['BsmtFinSF1'].clip_upper(test2['BsmtFinSF1'].quantile(0.99))

test2['TotalBsmtSF']= test2['TotalBsmtSF'].clip_upper(test2['TotalBsmtSF'].quantile(0.99))

test2['TotalBsmtSF']= test2['TotalBsmtSF'].clip_upper(test2['TotalBsmtSF'].quantile(0.99))

test2['TotRmsAbvGrd']= test2['TotRmsAbvGrd'].clip_upper(test2['TotRmsAbvGrd'].quantile(0.99))
finaltest = pd.concat([test2,test3],axis = 1)
finaltest1 = pd.get_dummies(finaltest)
finaltest.columns
final_data1.head()
train1 =final_data1.sample(n = 730 ,random_state = 123)

train2 = final_data1.drop(train1.index)
train1x = train1.drop(['intercept','SalePrice'], axis = 1)

train1y = train1.SalePrice
train2x = train2.drop(['SalePrice','intercept'],axis = 1)

train2y = train2.SalePrice
best_train = pd.get_dummies(best_train)
train_s1 = best_train.sample(n = 730 ,random_state = 123)

train_s2 = best_train.drop(train_s1.index)                             
train_s1x = train_s1.drop(['SalePrice'], axis = 1)

train_s1y = train_s1.SalePrice
train_s2x = train_s2.drop(['SalePrice'],axis = 1)

train_s2y = train_s2.SalePrice
from sklearn.linear_model import LinearRegression
X_train = train1x

Y_train = train1y
linreg = LinearRegression()

linreg.fit(X_train, Y_train)
X_train , X_test, Y_train, Y_test = train_test_split(

        train2x,

        train2y,

        test_size=0.20,

        random_state=123)
y_pred = linreg.predict(X_test)
from sklearn import metrics
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))

rmse
metrics.r2_score(Y_test, y_pred)
from sklearn import metrics

from sklearn.tree import DecisionTreeRegressor

from sklearn.grid_search import GridSearchCV
X_train = train1x 

Y_train = train1y
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
tree_final = DecisionTreeRegressor(max_depth=8)

tree_final.fit(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(

        train2x,

        train2y,

        test_size=0.20,

        random_state=123)
tree_test_pred = pd.DataFrame({'actual': Y_test, 'predicted': tree_final.predict(X_test)})
tree_test_pred.sample(10)
metrics.r2_score(Y_test, tree_test_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))

rmse
from sklearn.ensemble import RandomForestRegressor
X_train = train1x

Y_train = train1y
depth_list = list(range(1,20))

for depth in depth_list:

    dt_obj = RandomForestRegressor(max_depth=depth)

    dt_obj.fit(X_train, Y_train)

    print ('depth:', depth, 'R_Squared:', metrics.r2_score(Y_test, dt_obj.predict(X_test)))
radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100)

radm_clf.fit( X_train, Y_train )
X_train, X_test, Y_train, Y_test = train_test_split(

        train2x,

        train2y,

        test_size=0.20,

        random_state=123)
radm_test_pred = pd.DataFrame( { 'actual':  Y_test,

                            'predicted': radm_clf.predict( X_test ) } )
metrics.r2_score( radm_test_pred.actual, radm_test_pred.predicted )
rmse = np.sqrt(metrics.mean_squared_error(radm_test_pred.actual, radm_test_pred.predicted))

rmse
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics

import matplotlib.pyplot as plt 

import seaborn as sns
param_bag = {'n_estimators': list(range(100, 801, 100)),

             }
from sklearn.grid_search import GridSearchCV

bag_cl = GridSearchCV(estimator=BaggingRegressor(),

                  param_grid=param_bag,

                  cv=5,

                  verbose=True, n_jobs=-1)
bag_cl.get_params()
X_train = train1x

Y_train = train1y
bag_cl.fit(X_train, Y_train)
bag_cl.best_params_
bagclm = BaggingRegressor(oob_score=True, n_estimators=600)

bagclm.fit(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(

        train2x,

        train2y,

        test_size=0.20,

        random_state=123)
y_pred = pd.DataFrame( { 'actual':  Y_test,

                            'predicted': bagclm.predict( X_test) } )
bagclm.estimators_features_
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
X_train = train1x

Y_train = train1y
ada.fit(X_train, Y_train)
ada.best_params_
ada_clf = AdaBoostRegressor(learning_rate=0.1, n_estimators=800)
ada_clf.fit(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(

        train2x,

        train2y,

        test_size=0.20,

        random_state=123)
ada_test_pred = pd.DataFrame({'actual': Y_test,

                            'predicted': ada_clf.predict(X_test)})
metrics.r2_score(ada_test_pred.actual, ada_test_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(Y_test, y_pred.predicted))

rmse
param_test1 = {'n_estimators': [100, 200, 400, 600, 800],

              'max_depth': list(range(1,10))}

gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,

                                                               max_features='sqrt',subsample=0.8, random_state=10), 

                        param_grid = param_test1, scoring='r2',n_jobs=4,iid=False, cv=5)
X_train = train1x

Y_train = train1y
gsearch1.fit(X_train, Y_train)
gsearch1.best_params_
gbm = GradientBoostingRegressor(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,max_depth=1, n_estimators=200,

                                                               max_features='sqrt',subsample=0.8, random_state=10)
gbm.fit(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(

        train2x,

        train2y,

        test_size=0.20,

        random_state=123)
gbm_test_pred = pd.DataFrame({'actual': Y_test,

                            'predicted': gbm.predict(X_test)})
metrics.r2_score(gbm_test_pred.actual, gbm_test_pred.predicted)
rmse = np.sqrt(metrics.mean_squared_error(gbm_test_pred.actual, gbm_test_pred.predicted))

rmse