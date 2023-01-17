import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import preprocessing
import seaborn as sns
from scipy.stats import skew
import warnings
warnings.filterwarnings("ignore")
import os

def checknan(ds):
    dsm = pd.DataFrame(ds.isnull().sum()/ds.shape[0], columns=['nan%'])
    return (dsm[dsm['nan%']>0]['nan%'].sort_values(ascending = False))

def countnan(ds):
    dsm = pd.DataFrame(ds.isnull().sum(), columns=['nan'])
    return (dsm[dsm['nan']>0]['nan'].sort_values(ascending = False))
#dftest = pd.read_csv('test.csv')
#dftrain = pd.read_csv('train.csv')
#dftest.shape, dftrain.shape
dftrain = pd.read_csv("../input/train.csv")
dftest = pd.read_csv("../input/test.csv")
corr = dftrain.corr()
corr2 = dftrain[corr.SalePrice.sort_values(ascending=False).head(11).index]
sns.pairplot(corr2)
corr2.columns
plt.scatter(dftrain['GrLivArea'], dftrain['SalePrice'])
dftrain[(dftrain.SalePrice<300000) & (dftrain.GrLivArea>4000)]
dftrain.drop([523, 1298], axis=0, inplace=True)
plt.scatter(dftrain['GrLivArea'], dftrain['SalePrice'])
corr = dftrain.corr()
corr2 = dftrain[corr.SalePrice.sort_values(ascending=False).head(11).index]
sns.pairplot(corr2)
corr2.columns
def bp(feature):
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    ax1 = plt.boxplot((dftrain[feature]))
    plt.xlabel(feature)
    ax2 = fig.add_subplot(122)
    ax2 = plt.boxplot(np.log1p(dftrain[feature]))
    plt.xlabel(feature+' log')
    plt.show
for featr in corr2.columns:
    bp(featr)
'''took a look but decided apply log to all numeric features.. later on this notebook'''
'''reset dataframe to start point'''
df = dftrain
df = df.append(dftest, sort=True)
df.shape
checknan(df)
'''check missing in a df'''
nan_ck = pd.DataFrame(df.isnull().sum(), columns = ['Nan_sum'])
nan_ck = nan_ck.drop('SalePrice')
nan_ck['Nan_cnt'] = pd.DataFrame(df.isnull().count())
nan_ck['Nan%'] = nan_ck['Nan_sum'] / nan_ck['Nan_cnt']
nan_ck = nan_ck[nan_ck['Nan%'] != 0].sort_values(['Nan%'], ascending = False)
nan_ck.head(80)
'''low missing data'''
lowmd = nan_ck[(nan_ck['Nan_sum'] <= 4) & (nan_ck['Nan_sum'] > 0)]
'''mid missing data'''
midmd = nan_ck[(nan_ck['Nan_sum'] <= 486) & (nan_ck['Nan_sum'] > 4)]
'''hi missing data'''
himd = nan_ck[(nan_ck['Nan_sum'] > 486)]
print('himd:\n',list(himd.index),'\n\nlowmd:\n', list(lowmd.index), '\n\nmidmd:\n', list(midmd.index))
df[himd.index].info()
df['PoolQC'].value_counts(), \
df['MiscFeature'].value_counts(), \
df['Alley'].value_counts(), \
df['Fence'].value_counts(), \
df['FireplaceQu'].value_counts()
df[himd.index] = df[himd.index].fillna('NA')
df[lowmd.index].info()
df[lowmd.index].describe()
df[lowmd.index]=df[lowmd.index].fillna(df[lowmd.index].median())
df[lowmd.index].describe()
lowmd1 = df[checknan(df[lowmd.index]).index]
lowmd1.head()
print (lowmd1['MSZoning'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Functional'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Utilities'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['SaleType'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Electrical'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['KitchenQual'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Exterior1st'].value_counts().head(2),'\n', '-' * 25)
print (lowmd1['Exterior2nd'].value_counts().head(2))
df['MSZoning'] = df['MSZoning'].fillna('RL')
df['Functional'] = df['Functional'].fillna('Typ')
df['Utilities'] = df['Utilities'].fillna('AllPub')
df['SaleType'] = df['SaleType'].fillna('WD')
df['Electrical'] = df['Electrical'].fillna('SBrkr')
df['KitchenQual'] = df['KitchenQual'].fillna('TA')
df['Exterior1st'] = df['Exterior1st'].fillna('VinylSd')
df['Exterior2nd'] = df['Exterior2nd'].fillna('VinylSd')
df[midmd.index].info()
checknan(df[midmd.index].select_dtypes(include=np.number))
print( 'normal >', df['LotFrontage'].skew(), ' -- log >', np.log1p(df['LotFrontage']).skew())
print( 'normal >', df['GarageYrBlt'].skew(), ' -- log >', np.log1p(df['GarageYrBlt']).skew())
print( 'normal >', df['MasVnrArea'].skew(), ' -- log >', np.log1p(df['MasVnrArea']).skew())
fig = plt.figure(figsize=(20,15))

ax1 = fig.add_subplot(321)
ax1.hist(df[df['LotFrontage']>=0]['LotFrontage'], bins=18)
plt.xlabel('LotFrontage')
ax2 = fig.add_subplot(322)
ax2.hist(np.log1p(df[df['LotFrontage']>=0]['LotFrontage']), bins=18)
plt.xlabel('LotFrontage Log')

ax3 = fig.add_subplot(323)
ax3.hist(df[df['GarageYrBlt']>=0]['GarageYrBlt'], bins=18)
plt.xlabel('GarageYrBlt')
ax4 = fig.add_subplot(324)
ax4.hist(np.log1p(df[df['GarageYrBlt']>=0]['GarageYrBlt']), bins=18)
plt.xlabel('GarageYrBlt Log')

ax5 = fig.add_subplot(325)
ax5.hist(df[df['MasVnrArea']>=0]['MasVnrArea'], bins=18)
plt.xlabel('MasVnrArea')
ax6 = fig.add_subplot(326)
ax6.hist(np.log1p(df[df['MasVnrArea']>=0]['MasVnrArea']), bins=18)
plt.xlabel('MasVnrArea Log')

plt.show()
df['MasVnrArea'].value_counts().head(10) # set to 0
df['GarageYrBlt'].value_counts().head(10) # set to 2005
'''apply log on to LotFrontage and fillna = median
    fill MasVnrArea NA = 0.0
    fill GarageYrBlt NA = 2005.0'''
df['LotFrontage'] = df['LotFrontage'].fillna(df[df['LotFrontage']>0]['LotFrontage'].median())
#df['LotFrontage'] = np.log1p(df['LotFrontage'])
df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(2005.0)
checknan(df[midmd.index].select_dtypes(include=np.number))
checknan(df[midmd.index].select_dtypes(include=['object']))
lista = list(checknan(df[midmd.index].select_dtypes(include=['object'])).index)
print('-' * 60)
for n in lista:
    print('',n,'\n','='*len(n)) 
    print(df[n].value_counts()/df[n].count(),'\n','-'*80)
'''
 'GarageFinish' = Unf
 'GarageQual' = TA,
 'GarageCond' = TA,
 'GarageType' = Attchd,
 'BsmtCond' = TA,
 'BsmtExposure' = No,
 'BsmtQual' = TA,
 'BsmtFinType2' = Unf,
 'BsmtFinType1' = Unf,
 'MasVnrType' = None
 '''
lista2 = ['Unf','TA','TA','Attchd','TA','No','TA','Unf','Unf','None']
listadict = dict(zip(lista, lista2))
listadict
for n in listadict:
    df[n] = df[n].fillna(listadict[n])
''' apply log to full df:
    SalePrice,
    GrLivArea, 
    1stFlrSF, 
    GarageArea, 
    TotRmsAbvGrd
'''
'''apply log to all numeric features'''
#for featr in ['GrLivArea', 
#              '1stFlrSF', 
#              'GarageArea', 
#              'TotRmsAbvGrd']:
#    df[featr] = np.log1p(df[featr])
numerics = list(df.drop(['SalePrice', 'Id'], axis=1).select_dtypes(include=np.number).columns)
'''implement squares and cubic features to numerics and and apply log...'''
for featr in numerics:
    df[featr+'2'] =  np.log1p(df[featr] ** 2)
    df[featr+'3'] =  np.log1p(df[featr] ** 3)
    df[featr] = np.log1p(df[featr])
all = df
all = pd.get_dummies(all,drop_first=True)
X_train = all[:dftrain.shape[0]]
Y_train = X_train[['Id','SalePrice']]
X_test = all[dftrain.shape[0]:]
X_train.drop(['Id','SalePrice'], axis=1, inplace=True)
X_test.drop(['Id','SalePrice'], axis=1, inplace=True)
X_train.shape, Y_train.shape, X_test.shape

from sklearn.model_selection import cross_val_score, KFold, learning_curve
import xgboost as xgb
from xgboost import XGBClassifier,XGBRegressor
#from sklearn.grid_search import GridSearchCV 
from sklearn import metrics
seed = 45
n_folds = 5
kfold = KFold(n_folds, shuffle=True, random_state=seed).get_n_splits(X_train)
def error(actual, predicted):
    actual = np.log(actual)
    predicted = np.log(predicted)
    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))

def get_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, 
                                    X_train, 
                                    y=np.log(Y_train.SalePrice), 
                                    scoring="neg_mean_squared_error", 
                                    cv=kfold))
    return rmse.mean()
xgbmod = XGBRegressor() 
'''w/o fine tuning'''
xgbmod.fit(X_train, np.log(Y_train.SalePrice))
yhat = np.exp(xgbmod.predict(X_train))
print('rmse:', get_rmse(xgbmod))
import shap
shap_values = shap.TreeExplainer(xgbmod).shap_values(X_train)

global_shap_vals = np.abs(shap_values).mean(0)[:-1]
variables_values = pd.DataFrame(list(zip(X_train.columns,global_shap_vals)))
variables_values.rename(columns={0:'variable',1:'shap_value'},inplace=True)
variables_values.sort_values(by=['shap_value'],ascending=False,inplace=True)
top_n = variables_values.head(12)

pos=range(0,-top_n.shape[0],-1)
plt.barh(pos, top_n['shap_value'], color="#007fff")
plt.yticks(pos, top_n['variable'])
plt.xlabel("mean SHAP value magnitude (do not change in log odds)")
plt.gcf().set_size_inches(8, 4)
plt.gca()
plt.show()
shap.summary_plot(shap_values, X_train)
'''remove shap_0 features to new prediction'''
top_n = variables_values
remove_featr = list(top_n[top_n.shap_value == 0]['variable'])
X_train.drop(remove_featr, axis=1, inplace = True)
X_test.drop(remove_featr, axis=1, inplace = True)
xgbmod = XGBRegressor(colsample_bylevel=1, colsample_bytree=1, learning_rate=0.03,max_delta_step=0, 
                      max_depth=6,min_child_weight=6,n_estimators=450,subsample= 0.5)
'''w/ fine tuning'''                      

xgbmod.fit(X_train, np.log(Y_train.SalePrice))
yhat = np.exp(xgbmod.predict(X_train))
print('rmse:', get_rmse(xgbmod))
#yhat = np.exp(xgbmod.predict(X_test))
#        #yhat = np.exp(yhat)
#Y_test = all[dftrain.shape[0]:]['Id']
#yhat = pd.DataFrame(yhat, columns = ['SalePrice'])
#Y_test = pd.DataFrame (Y_test)
#Y_test['SalePrice'] = yhat.SalePrice
#Y_test.to_csv('subm6.csv', index= False)

#'''gridsearch better params'''
#
#xgb_reg = XGBRegressor()
#parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
#              'colsample_bylevel' : [1],
#              'learning_rate': [0.03,0.04], #so called `eta` value
#              'max_depth': [6],
#              'min_child_weight': [6],
#              'max_delta_step' : [0],
#              'reg_lambda': [1],
#              'subsample': [0.5, 0.6],
#              'colsample_bytree': [ 1],
#              'n_estimators': [450, 550]}
#
#xgb_grid_reg = GridSearchCV(xgb_reg,
#                        parameters,
#                        cv = 2,
#                        n_jobs = 5,
#                        verbose=True)
#
#xgb_grid_reg.fit(X_train,np.log1p(Y_train.SalePrice))
#
#print(xgb_grid_reg.best_score_)
#print(xgb_grid_reg.best_params_)
#0.9071702277585866
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.03, 
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 7, 'n_estimators': 400, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.5}
#
#0.9093014037566548
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.04, 
# 'max_delta_step': 0, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 400, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.6}
#
#0.9075638827570375
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.03, 
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 6, 'n_estimators': 500, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.5}
#
#0.9077751136520033
#{'colsample_bylevel': 1, 'colsample_bytree': 1, 'learning_rate': 0.03, 
# 'max_delta_step': 0, 'max_depth': 6, 'min_child_weight': 6, 'n_estimators': 450, 
# 'nthread': 4, 'reg_lambda': 1, 'subsample': 0.5}