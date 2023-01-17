# Loading libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.metrics import make_scorer, fbeta_score, accuracy_score

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

from lightgbm import LGBMRegressor
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading train_data

train = pd.read_csv('/kaggle/input/house-prices-dataset/train.csv')

pd.set_option('display.max_columns', 82)

train.head()
# Loading test_data

test = pd.read_csv('/kaggle/input/house-prices-dataset/test.csv')

test.head()
print('The train data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))

print('The test data has {} rows and {} columns'.format(test.shape[0],test.shape[1]))
train.info()
# Checking for the null values

train.isnull().sum()
train.columns[train.isnull().any()]
# Missing values percentage

miss=train.isnull().sum()/len(train)

miss=miss[miss>0]

miss.sort_values(inplace=True)

miss
# Numerical variables

num=train.select_dtypes(include=np.number)

print(num.columns)

print('No. of Numerical variables are {}'.format(len(num.columns)))
# Categorical variables

cat_data=train.select_dtypes(exclude=np.number)

print(cat_data.columns)

print('No. of Categorical variables are {}'.format(len(cat_data.columns)))
num=num.drop('Id',axis=1)
# Checking the distribution of target variable

sns.distplot(train['SalePrice'])

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,10))

matrix=np.triu(num.corr())

sns.heatmap(num.corr(),mask=matrix,cmap='coolwarm')

plt.show()
corr=num.corr()

print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n') #top 15 values

print ('----------------------')

print (corr['SalePrice'].sort_values(ascending=False)[-5:]) #last 5 values
plt.figure(figsize=(16,6))

corr['SalePrice'].sort_values(ascending=False)[1:].plot(kind='bar')
train['OverallQual'].unique()
#let's check the mean price per quality and plot it.

pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median) # median because target is right skewed.

pivot.sort_values(by='SalePrice')
pivot.plot(kind='bar')

plt.show()
#GrLivArea variable

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])

plt.show()
fig, ax = plt.subplots(ncols=2, nrows = 2, figsize=(12,10))

sns.scatterplot(x='GarageCars',y='SalePrice', data=train, ax=ax[0,0])

sns.scatterplot(x='GarageArea', y='SalePrice', data=train, ax=ax[0,1])

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train, ax=ax[1,0])

sns.scatterplot(x='YearBuilt', y='SalePrice', data=train, ax=ax[1,1])

plt.show()
sns.scatterplot(x='LotFrontage', y='SalePrice', data=train)

plt.show()
import scipy.stats as stats
cat = [f for f in train.columns if train.dtypes[f] == 'object']

def anova(frame):

    anv = pd.DataFrame()

    anv['features'] = cat

    pvals = []

    for c in cat:

        samples = []

        for cls in frame[c].unique():

            s = frame[frame[c] == cls]['SalePrice'].values

            samples.append(s)

        pval = stats.f_oneway(*samples)[1]

        pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')
cat_data['SalePrice'] = train.SalePrice.values

k = anova(cat_data) 

k['disparity'] = np.log(1/k['pval'].values) 

print(k)
plt.figure(figsize=(10,6))

sns.barplot(data=k, x = 'features', y='disparity') 

plt.xticks(rotation=90) 

plt.show()
#dropping id column from both train and test datsets.

train=train.drop('Id',axis=1)

test=test.drop('Id',axis=1)
#dropping outliers from GrLivArea

train=train[train['GrLivArea']<4000]

train.reset_index(drop=True,inplace=True)
#dropping outliers from TotalBsmtSF

train.drop(train[train['TotalBsmtSF']>4000].index,inplace=True)
#dropping outliers from LotFrontage

train.drop(train[train['LotFrontage']>250].index,inplace=True)

train.reset_index(drop=True,inplace=True)
#Now combining test and tarin datasets

train['DataType']='train'

test['DataType']='test'

test['SalePrice']=np.nan
data=pd.concat([train,test],sort=False)

data.info()
print(data.columns[data.isnull().any()])
# Missing values percentage

miss=data.isnull().sum()/len(data)

miss=miss[miss>0]

miss.sort_values(inplace=True)

miss
miss_col=data[['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath',

       'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu',

       'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',

       'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType']]

for i in miss_col.columns:

    print(miss_col[i].value_counts())
data['MSSubClass'] = data['MSSubClass'].apply(str)

data.groupby('MSSubClass')['MSZoning']
data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# imputing LotFrontage by the median of neighborhood

data['LotFrontage']=data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median))
data=data.drop(['Alley','Utilities'],axis=1)
data['Exterior1st']=data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd']=data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['MasVnrArea']=data['MasVnrArea'].fillna(0)
data['MasVnrType']=data.groupby('MasVnrArea')['MasVnrType'].transform(lambda x:x.fillna(x.mode()[0]))
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')
for col in ('BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'):

    data[col] = data[col].fillna(0)
data['Electrical']=data['Electrical'].fillna(data['Electrical'].mode()[0])
for col in ('BsmtHalfBath','BsmtFullBath'):

    data[col] = data[col].fillna(0)
for col in ('KitchenQual', 'Functional', 'FireplaceQu'):

    data[col]=data[col].fillna(data[col].mode()[0])

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    data[col] = data[col].fillna(0)



for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data[col] = data[col].fillna('None')
data=data.drop(['PoolQC', 'Fence', 'MiscFeature'],axis=1)
data['SaleType']=data['SaleType'].fillna(data['SaleType'].mode()[0])
data.columns[data.isnull().any()]
data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +

                                 data['1stFlrSF'] + data['2ndFlrSF'])



data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +

                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))



data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +

                              data['EnclosedPorch'] + data['ScreenPorch'] +

                              data['WoodDeckSF'])
data=data[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt',

           'YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','YrSold','OverallCond','MSSubClass',

           'EnclosedPorch','KitchenAbvGr','Neighborhood','ExterQual','KitchenQual','Foundation','HeatingQC','SaleCondition',

           'Exterior1st','Exterior2nd','SaleType','MSZoning','HouseStyle','LotShape','CentralAir','PavedDrive','RoofStyle',

           'haspool','has2ndfloor','hasgarage','hasbsmt','hasfireplace','Total_sqr_footage','Total_Bathrooms','Total_porch_sf','DataType']]
cat_cols=data.select_dtypes(['object']).columns

cat_cols
cat_cols=cat_cols[:-1]

cat_cols
# Doing dummification

for col in cat_cols:

    freqs=data[col].value_counts()

    k=freqs.index[freqs>20][:-1]

    for cat in k:

        name=col+'_'+cat

        data[name]=(data[col]==cat).astype(int)

    del data[col]

    print(col)
data.shape
data_train=data[data['DataType']=='train']

del data_train['DataType']

data_test=data[data['DataType']=='test']

data_test.drop(['SalePrice','DataType'],axis=1,inplace=True)
print('Train Shape',data_train.shape)

print('Test Shape',data_test.shape)
del data
data_train['SalePrice']=np.log(data_train['SalePrice'])
X=data_train.drop('SalePrice',axis=1)

y=data_train['SalePrice']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=1)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
dt=DecisionTreeRegressor()

parameters = {'max_depth':range(1,10)},{'min_samples_split': (0.1,0.2,0.3,0.4,0.5)},

{'min_samples_leaf': range(1,10)},{'min_weight_fraction_leaf': (0.0,0.1,0.2)}

grid_obj = GridSearchCV(dt, param_grid = parameters)

grid_fit = grid_obj.fit(X_train,y_train)



best_dt = grid_fit.best_estimator_

best_dt
dt1=DecisionTreeRegressor(criterion='mse', max_depth=6, max_features=None,

                      max_leaf_nodes=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=2, min_weight_fraction_leaf=0.0,

                      presort=False, random_state=None, splitter='best')

dt1.fit(X_train,y_train)

print('Decision Tree Training Score :',dt1.score(X_train,y_train))

print('Decision Tree Testing Score :',dt1.score(X_test,y_test))
y_pred=np.floor(np.exp(dt1.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
score = []

for k in range(1,100):   # running for different K values to know which yields the max accuracy. 

    clf = KNeighborsRegressor(n_neighbors = k,  weights = 'distance', p=1)

    clf.fit(X_train, y_train)

    score.append(clf.score(X_test, y_test ))
k_max = score.index(max(score))+1

print( "At K = {}, Max Accuracy = {}".format(k_max, max(score)*100))
knn=KNeighborsRegressor(n_neighbors=7,weights='distance')

knn.fit(X_train,y_train)

print('KNN Training Score :',knn.score(X_train,y_train))

print('KNN Testing Score :',knn.score(X_test,y_test))
rfg=RandomForestRegressor(random_state=1)
estimators = np.arange(10, 200, 2)

scores = []

for n in estimators:

    rfg.set_params(n_estimators=n)

    rfg.fit((X_train), y_train)

    scores.append(rfg.score((X_test), y_test))

print(scores)
estimators[scores.index(max(scores))]
param_dist = {'n_estimators': [48,64],'max_depth': [2, 3, 4,10],'bootstrap': [True, False],

              'max_features': ['auto', 'sqrt', 'log2', None],

              'criterion': ['mse', 'mae']}



cv_rf = GridSearchCV(rfg, cv = 5 ,param_grid=param_dist, n_jobs = 3)

cv_rf.fit(X_train,y_train)

print('RF Training Score :',cv_rf.score(X_train,y_train))

print('RF Testing Score :',cv_rf.score(X_test,y_test))
y_pred=np.floor(np.exp(cv_rf.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in learning_rates:

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate = learning_rate, max_depth = 2, random_state = 1)

    gb.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)

    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, y_train)))

    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_test, y_test)))

    print()
gb = GradientBoostingRegressor(n_estimators=100, learning_rate = 0.1, max_depth = 2, random_state = 1)

gb.fit(X_train, y_train)

print('RF Training Score :',gb.score(X_train,y_train))

print('RF Testing Score :',gb.score(X_test,y_test))
y_pred=np.floor(np.exp(gb.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
from xgboost import XGBRegressor
xgb=XGBRegressor(max_depth=5)

xgb.fit(X_train,y_train)

print('XGB Training Score :',xgb.score(X_train,y_train))

print('XGB Testing Score :',xgb.score(X_test,y_test))
y_pred=np.floor(np.exp(xgb.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
lambdas=np.linspace(0.001,2,100)

params={'alpha':lambdas}

ls=Lasso(fit_intercept=True)
grid_search=GridSearchCV(ls,param_grid=params,cv=10,scoring='neg_mean_squared_error')

grid_search.fit(X_train,y_train)



lasso=grid_search.best_estimator_

lasso
lasso.fit(X_train,y_train)
print('Lasso Training Score :',lasso.score(X_train,y_train))

print('Lasso Testing Score :',lasso.score(X_test,y_test))
y_pred=np.floor(np.exp(lasso.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
rg=Ridge(fit_intercept=True)

grid_search=GridSearchCV(rg,param_grid=params,cv=10,scoring='neg_mean_squared_error')

grid_search.fit(X_train,y_train)



ridge=grid_search.best_estimator_

ridge
ridge.fit(X_train,y_train)
print('Ridge Training Score :',ridge.score(X_train,y_train))

print('Ridge Testing Score :',ridge.score(X_test,y_test))
y_pred=np.floor(np.exp(ridge.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
rcv=RidgeCV(alphas=np.arange(0.001,1,0.001),store_cv_values=True)

rcv.fit(X_train,y_train)

print('RidgeCV Training Score :',rcv.score(X_train,y_train))

print('RidgeCV Testing Score :',rcv.score(X_test,y_test))
lightgbm = LGBMRegressor(objective='regression', num_leaves=6,learning_rate=0.01, n_estimators=7000,max_bin=200, 

                         bagging_fraction=0.8,bagging_freq=4, bagging_seed=8,feature_fraction=0.2,feature_fraction_seed=8,

                         min_sum_hessian_in_leaf = 11,verbose=-1,random_state=42)

lightgbm.fit(X_train,y_train)
print('LGBM Training Score :',lightgbm.score(X_train,y_train))

print('LGBM Testing Score :',lightgbm.score(X_test,y_test))
y_pred=np.floor(np.exp(lightgbm.predict(data_test)))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
from mlxtend.regressor import StackingCVRegressor
stregr = StackingCVRegressor(regressors=(ridge,lasso,lightgbm,xgb,gb), 

                           meta_regressor=lasso,use_features_in_secondary=True)

stregr.fit(X_train,y_train)
print('StackGen Training Score :',stregr.score(np.array(X_train),np.array(y_train)))

print('StackGen Testing Score :',stregr.score(np.array(X_test),np.array(y_test)))
y_pred=np.floor(np.exp(stregr.predict(np.array(data_test))))
#sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':y_pred})

#sub.to_csv('submission.csv',index=False)
def blended_predictions(data_test):

    return ((0.2 * ridge.predict(data_test)) + \

            (0.3 * lasso.predict(data_test)) + \

            (0.05 * lightgbm.predict(data_test)) + \

            (0.05 * xgb.predict(data_test)) + \

            (0.05 * gb.predict(data_test)) + \

            (0.05 * cv_rf.predict(data_test)) + \

            (0.3 * stregr.predict(np.array(data_test))))
np.exp(blended_predictions(X))
sub = pd.DataFrame({'Id':np.arange(1461,2920), 'SalePrice':np.floor(np.exp(blended_predictions(data_test)))})

sub.to_csv('submission.csv',index=False)
submission = pd.read_csv("submission.csv")

submission.shape
q1 = submission['SalePrice'].quantile(0.0045)

q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission1.csv", index=False)
submission['SalePrice'].head()
#ScAle Predictions



#submission['SalePrice'] *= 1.001619

#submission.to_csv("submission2.csv", index=False)