# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
url_data = '../input/train.csv'
url_test = '../input/test.csv'
train_data = pd.read_csv(url_data)
test_data = pd.read_csv(url_test)
train_data.head()
train_ID = train_data['Id']
test_ID = test_data['Id']
train_data.drop(['Id'], axis = 1, inplace=True)
test_data.drop(['Id'], axis = 1, inplace=True)
print("The shape of train_data is {}".format(train_data.shape))
print("The shape of test_data is {}".format(test_data.shape))
#train_data['SalePrice']
#all_data = pd.concat()
import seaborn as sb
sb.scatterplot(x=train_data['LotArea'], y = train_data['SalePrice'])
train_data = train_data.drop(train_data[train_data['LotArea']>150000].index)
sb.distplot(train_data['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
sb.distplot(train_data['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(train_data['SalePrice'], plot=plt)
y_train = train_data.SalePrice.values
n_train = train_data.shape[0]
all_data = pd.concat([train_data, test_data], axis = 0, sort=False)
all_data.drop(['SalePrice'], axis = 1, inplace=True)
#all_data.head()
y_train.shape
n = all_data.shape[0]
na = (all_data.isnull().sum()/n).sort_values(ascending=False).head(20)
plt.subplots(figsize=(22,9))
sb.barplot(x= na.index, y = na.values)
all_data.corr().style.background_gradient().set_precision(1)
#all_data = all_data.drop(['GarageYrBlt','TotalBsmtSF','TotRmsAbvGrd','GarageCars'], axis=1)
(all_data.isnull().sum()/n).sort_values(ascending=False).head(20)
colnone = ['Alley','FireplaceQu','Fence','PoolQC']
#All Basement and Garage related variables: NA => None
colnone.extend(['BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2'])
colnone.extend(['GarageQual','GarageCond','GarageFinish','GarageType','GarageArea'])
colnone.extend(['MasVnrType','MiscFeature'])
colnone.extend(['GarageYrBlt','GarageCars'])

for coli in colnone:
    all_data[coli].fillna('None',inplace=True)

#Missing values for numeric data could be zero
#Here, missing data for Basement area could be zero.
col0 = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']
col0.extend(['TotalBsmtSF'])
col0.extend(['MasVnrArea','Fireplaces','PoolArea','MiscVal'])
for coli in col0:
    all_data[coli].fillna(0, inplace=True)
    
#Median Imputing
colmedian = ['LotFrontage','LotArea','YearBuilt','1stFlrSF','WoodDeckSF']
colmedian.extend(['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch'])

for coli in colmedian:
    all_data[coli].fillna(all_data[coli].median(),inplace=True)

#YearRemodAdd same as YearBuilt if na
all_data['YearRemodAdd'].fillna(all_data['YearBuilt'], inplace=True)
all_data['Functional'].fillna('Typ',inplace=True)
#2ndFlrSF => depends on if the house is 1 storied or 2
(all_data.isnull().sum()).sort_values(ascending=False).head(10)
sb.boxplot(x='MSZoning', y='GrLivArea', data=all_data, palette='hls')
colmode = ['Electrical','MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','SaleType']
for coli in colmode:
    all_data[coli].fillna(all_data[coli].mode()[0],inplace=True)
    
#all_data.dropna(inplace=True)
(all_data.isnull().sum()).sort_values(ascending=False).head(10)
colcat = ['MSSubClass','YearBuilt','YearRemodAdd','YrSold','MoSold']
for coli in colcat:
    all_data[coli] = all_data[coli].astype(str)
all_data.dtypes
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
#numerics = [ 'float16', 'float32', 'float64']

#numeric = all_data.select_dtypes(include=numerics)

#for c in numeric.columns:
 #   all_data[c] = np.log1p(all_data[c])
    
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.8]
print("There are {} skewed numerical features to transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
for feat in skewed_features:
    all_data[feat] = np.log1p(all_data[feat])
all_data = pd.get_dummies(all_data)
print(all_data.shape)
train = all_data[:n_train]
test = all_data[n_train:]
train.shape
def rms_score(y_pred, y_train):
    score = np.square((y_pred) - (y_train))
    return np.sqrt(score.sum())
from sklearn.decomposition import PCA
pca_list = [100, 150, 200, 300]
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

lr = Lasso(alpha = 1e-5, normalize = True, max_iter=1e5)
for i in pca_list:
    pca = PCA(n_components = i)
    pca.fit(train)
    train_pca = pca.transform(train)
    lr.fit(train_pca, y_train)
    #print(rms_score(lr.predict(train_pca), y_train))
    print(cross_val_score(lr, train_pca, y_train, cv = 5).mean())
#lr.fit(train, y_train)
#lr.score(train, y_train)
#y_pred = lr.predict(test)
#print('Cross Val Score: ', cross_val_score(lr, train, y_train, cv = 5))
alpha_list = [1e-6, 1e-5, 1e-4, 1e-3]
pca = PCA(n_components = 150)
pca.fit(train)
train_pca = pca.transform(train)
for i in alpha_list:
    lr_ = Lasso(i, normalize = True, max_iter=1e5)
    print(cross_val_score(lr_, train_pca, y_train, cv = 5).mean())    
#pca.explained_variance_
lr = Lasso(1e-5, normalize = True, max_iter = 1e5)
lr.fit(train_pca, y_train)
train.shape
print('Lasso Regression- Cross Val Scores:', cross_val_score(lr, train_pca, y_train, cv = 5))
print('The RMSE for Lasso Regression: ',rms_score(lr.predict(train_pca), y_train))
from sklearn.linear_model import Ridge
rd = Ridge(alpha = 1e-05, normalize = True, random_state = 2)
rd.fit(train_pca, y_train)
print('Ridge Regression CV scores:', cross_val_score(rd, train_pca, y_train, cv=5))
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(train_pca, y_train)
print('Linear Regression CV scores:', cross_val_score(lin, train_pca, y_train, cv=5))
from sklearn.svm import SVR
svr = SVR(kernel = 'linear')
svr.fit(train_pca, y_train)
print('SVR CV Scores:', cross_val_score(svr, train_pca, y_train, cv=5))
from sklearn.linear_model import ElasticNet
enet = ElasticNet(alpha = 1e-5, normalize = True, random_state = 2)
enet.fit(train_pca, y_train)
print('E-Net CV Scores:', cross_val_score(enet, train_pca, y_train, cv=5))
'''from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 20, max_depth = 50, random_state = 2)
rf.fit(train_pca, y_train)
print('Random Forest- CV Scores:',cross_val_score(rf, train_pca, y_train, cv = 5))
print('RMS:',rms_score(rf.predict(train_pca), y_train))'''
from sklearn.ensemble import GradientBoostingRegressor
#rates = [0.06, 0.08,  0.11, 0.13]
#for ri in rates:
#    gbm = GradientBoostingRegressor(loss = 'ls', learning_rate = ri, n_estimators = 150, random_state=2)
#    gbm.fit(X = train_pca, y = y_train)
#    print('GBM- CV Scores:', cross_val_score(gbm, train_pca, y_train, cv = 5).mean(),'for rate',ri)
    #print('RMS:', rms_score(gbm.predict(train_pca), y_train))
gbm = GradientBoostingRegressor(loss = 'ls', learning_rate = 0.11, n_estimators = 250, random_state=2)
gbm.fit(train_pca, y_train)
#print('GBM- CV Scores:', cross_val_score(gbm, train_pca, y_train, cv = 5))
print('RMS:', rms_score(gbm.predict(train_pca), y_train))
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(n_estimators = 100, learning_rate = 1., random_state = 2)
ada.fit(train_pca, y_train)
#print('AdaBoost:', cross_val_score(ada, train_pca, y_train, cv = 5))
print('RMS:', rms_score(ada.predict(train_pca), y_train))
test = pca.transform(test)
from mlxtend.regressor import StackingRegressor
stregr = StackingRegressor(regressors=[lr, rd, lin, svr, enet], 
                           meta_regressor=lr)
#y_lr = lr.predict(test)
#y_rf = rf.predict(test)
#y_gbm = gbm.predict(test)
#y_ada = ada.predict(test)
stregr.fit(train_pca, y_train)
y_pred = stregr.predict(test)
print('Stacked CV scores:', cross_val_score(stregr, train_pca, y_train, cv=5))
#y_pred = y_lr*1.0 + y_gbm*0.0 + y_ada*0.0
#y_pred = (y_ada*(1/0.612) + y_gbm*(1/0.280) + y_rf*(1/0.280) + y_lr*(1/0.384))/(1/0.612 + 1/0.28 + 1/0.28 + 1/0.384)
#test = pca.transform(test)
#y_pred = lr.predict(test)
y_pred = np.expm1(y_pred)
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = y_pred
sub.to_csv('submission.csv', index=False)
