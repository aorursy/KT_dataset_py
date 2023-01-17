# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option("display.max_columns",100)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load train and test data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

full = pd.concat([train,test], ignore_index=True)

train_id = train.Id

test_id = test.Id
full.head()
# get features that has missing values

features_missing_cnt = full.isnull().sum()

missing_features = {col:cnt for col,cnt in zip(features_missing_cnt.index,features_missing_cnt.values) if cnt!=0}

missing_features
# Sample basement records

full[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].head()
# missing basement record count

full[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].isnull().sum()
full[full.TotalBsmtSF.isnull()|full.TotalBsmtSF==0][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].drop_duplicates()
# missing basement record count

full[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].isnull().sum()# Impute default values for zero or missing basement areas

_ = full.set_value((full.TotalBsmtSF.isnull()|(full.TotalBsmtSF==0)),

                   ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',

                    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'],

                   ['NA','NA','NA','NA',0,'NA',0,0,0,0,0]

                  )

_ = train.set_value((train.TotalBsmtSF.isnull()|(train.TotalBsmtSF==0)),

                   ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',

                    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'],

                   ['NA','NA','NA','NA',0,'NA',0,0,0,0,0]

                  )

_ = test.set_value((test.TotalBsmtSF.isnull()|(test.TotalBsmtSF==0)),

                   ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',

                    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'],

                   ['NA','NA','NA','NA',0,'NA',0,0,0,0,0]

                  )
# missing basement record count

full[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].isnull().sum()
full[full.BsmtQual.isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
full[(full.BsmtExposure=='No')&(full.BsmtCond=='Fa')][['BsmtQual','BsmtCond','BsmtExposure']].BsmtQual.value_counts()
full[(full.BsmtExposure=='No')&(full.BsmtCond=='TA')][['BsmtQual','BsmtCond','BsmtExposure']].BsmtQual.value_counts()
_ = full.set_value(full.BsmtQual.isnull(),'BsmtQual','TA')

_ = train.set_value(train.BsmtQual.isnull(),'BsmtQual','TA')

_ = test.set_value(test.BsmtQual.isnull(),'BsmtQual','TA')
full[full.BsmtCond.isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
full[full.BsmtQual=='Gd'].BsmtCond.value_counts()
full[full.BsmtQual=='TA'].BsmtCond.value_counts()
_ = full.set_value(full.BsmtCond.isnull(),'BsmtCond','TA')

_ = train.set_value(train.BsmtCond.isnull(),'BsmtCond','TA')

_ = test.set_value(test.BsmtCond.isnull(),'BsmtCond','TA')
full[full.BsmtExposure.isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
full[(full.BsmtQual=='Gd')&(full.BsmtCond=='TA')&(full.BsmtFinType1=='Unf')&(full.BsmtFinType2=='Unf')].BsmtExposure.value_counts()
_ = full.set_value(full.BsmtExposure.isnull(),'BsmtExposure','No')

_ = train.set_value(train.BsmtExposure.isnull(),'BsmtExposure','No')

_ = test.set_value(test.BsmtExposure.isnull(),'BsmtExposure','No')
full[full.BsmtFinType2.isnull()][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
full[(full.BsmtFinSF2>=450)&(full.BsmtFinSF2<=550)&(full.TotalBsmtSF>=2000)&(full.TotalBsmtSF<=4000)][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
full[(full.BsmtQual=='Gd')&(full.BsmtCond=='TA')&(full.BsmtExposure=='No')&(full.BsmtFinType1=='GLQ')].BsmtFinType2.value_counts()
_ = full.set_value(full.BsmtFinType2.isnull(),'BsmtFinType2','Unf')

_ = train.set_value(train.BsmtFinType2.isnull(),'BsmtFinType2','Unf')

_ = test.set_value(test.BsmtFinType2.isnull(),'BsmtFinType2','Unf')
# missing basement record count

full[['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',

      'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']].isnull().sum()
# sample garage records

full[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].head()
# garage missing count

full[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].isnull().sum()
full[(full.GarageArea.isnull())|(full.GarageArea==0)][['GarageType','GarageYrBlt','GarageFinish','GarageCars',

                                                       'GarageArea','GarageQual','GarageCond']].drop_duplicates()
# fill garage features with default values for Null or zero garage area



_ = full.set_value((full.GarageArea.isnull())|(full.GarageArea==0),

                   ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'],

                   ['NA',0,'NA',0,0,'NA','NA']

                  )

_ = train.set_value((train.GarageArea.isnull())|(train.GarageArea==0),

                   ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'],

                   ['NA',0,'NA',0,0,'NA','NA']

                  )

_ = test.set_value((test.GarageArea.isnull())|(test.GarageArea==0),

                   ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond'],

                   ['NA',0,'NA',0,0,'NA','NA']

                  )
# garage missing count

full[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].isnull().sum()
full[full.GarageYrBlt.isnull()][['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']]
full[(full.GarageType=='Detchd')&(full.GarageCars==1)&(full.GarageArea==360)][['GarageType','GarageYrBlt','GarageFinish',

                                                                               'GarageCars','GarageArea','GarageQual',

                                                                               'GarageCond']]
_ = full.set_value(full.GarageYrBlt.isnull(),

                   ['GarageYrBlt','GarageFinish','GarageQual','GarageCond'],

                   [1970,'Unf','TA','TA']

                  )

_ = train.set_value(train.GarageYrBlt.isnull(),

                   ['GarageYrBlt','GarageFinish','GarageQual','GarageCond'],

                   [1970,'Unf','TA','TA']

                  )

_ = test.set_value(test.GarageYrBlt.isnull(),

                   ['GarageYrBlt','GarageFinish','GarageQual','GarageCond'],

                   [1970,'Unf','TA','TA']

                  )
# garage missing count

full[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']].isnull().sum()
full.Alley.value_counts(dropna=False)
#_ = full.drop('Alley', axis=1, inplace=True)

#_ = train.drop('Alley', axis=1, inplace=True)

#_ = test.drop('Alley', axis=1, inplace=True)



_ = full.set_value(full.Alley.isnull(),'Alley','NA')

_ = train.set_value(train.Alley.isnull(),'Alley','NA')

_ = test.set_value(test.Alley.isnull(),'Alley','NA')
train[train.Electrical.isnull()].SalePrice
_ = train[['SalePrice','Electrical']].boxplot(by=['Electrical'], figsize=(10,4))

_ = plt.plot(np.arange(100),[167500]*100, c='k')
_ = full.set_value(full.Electrical.isnull(),'Electrical','SBrkr')

_ = train.set_value(train.Electrical.isnull(),'Electrical','SBrkr')
full[['Exterior1st','Exterior2nd','ExterQual','ExterCond']].head()
full[full.Exterior1st.isnull()][['Exterior1st','Exterior2nd','ExterQual','ExterCond','RoofStyle','RoofMatl']]
full[(full.ExterQual=='TA')&(full.ExterCond=='TA')&(full.RoofStyle=='Flat')&(full.RoofMatl=='Tar&Grv')][['Exterior1st','Exterior2nd','ExterQual','ExterCond','RoofStyle','RoofMatl']]
_ = full.set_value(full.Exterior1st.isnull(),['Exterior1st','Exterior2nd'],'Plywood')

_ = test.set_value(test.Exterior1st.isnull(),['Exterior1st','Exterior2nd'],'Plywood')
full.Fence.value_counts(dropna=False)
#_ = full.drop('Fence', axis=1, inplace=True)

#_ = train.drop('Fence', axis=1, inplace=True)

#_ = test.drop('Fence', axis=1, inplace=True)



_ = full.set_value(full.Fence.isnull(),'Fence','NA')

_ = train.set_value(train.Fence.isnull(),'Fence','NA')

_ = test.set_value(test.Fence.isnull(),'Fence','NA')
full.FireplaceQu.value_counts(dropna=False)
full[full.FireplaceQu.isnull()][['FireplaceQu','Fireplaces']].drop_duplicates()
_ = full.set_value(full.FireplaceQu.isnull(), 'FireplaceQu', 'NA')

_ = train.set_value(train.FireplaceQu.isnull(), 'FireplaceQu', 'NA')

_ = test.set_value(test.FireplaceQu.isnull(), 'FireplaceQu', 'NA')
full.Functional.value_counts()
_ = full.set_value(full.Functional.isnull(),['Functional'],'Typ')

_ = test.set_value(test.Functional.isnull(),['Functional'],'Typ')
full[full.KitchenQual.isnull()][['KitchenAbvGr','KitchenQual']]
full[full.KitchenAbvGr==1].KitchenQual.value_counts()
_ = full.set_value(full.KitchenQual.isnull(),['KitchenQual'],'TA')

_ = test.set_value(test.KitchenQual.isnull(),['KitchenQual'],'TA')
full[['LotFrontage','LotArea','LotShape','LotConfig']].head()
full[full.LotFrontage.isnull()][['LotFrontage','LotArea','LotShape','LotConfig']].head()
#full[full.LotFrontage.isnull()].LotFrontage = full[['LotFrontage','LotShape','LotConfig']].groupby(['LotShape','LotConfig']).apply(lambda x : x.fillna(np.mean(x).astype(int)))



lot_lookup = full[['LotFrontage','LotShape','LotConfig']].groupby(['LotShape','LotConfig']).agg(np.mean).astype(int).reset_index()

LotFrontage_lookup = lot_lookup.pivot('LotShape','LotConfig','LotFrontage')

LotFrontage_lookup = LotFrontage_lookup.fillna(0)

LotFrontage_lookup
lot_fet = full[full.LotFrontage.isnull()][['LotShape','LotConfig']].drop_duplicates()

for LotShape,LotConfig in zip(lot_fet.LotShape,lot_fet.LotConfig):

    val = LotFrontage_lookup.loc[LotShape][LotConfig]

    full.set_value((full.LotShape==LotShape)&(full.LotConfig==LotConfig)&(full.LotFrontage.isnull()),'LotFrontage',val)

    train.set_value((train.LotShape==LotShape)&(train.LotConfig==LotConfig)&(train.LotFrontage.isnull()),'LotFrontage',val)

    test.set_value((test.LotShape==LotShape)&(test.LotConfig==LotConfig)&(test.LotFrontage.isnull()),'LotFrontage',val)
full[full.MSZoning.isnull()][['MSZoning','Neighborhood','Condition1','Condition2']].drop_duplicates()
full[(full.Neighborhood=='IDOTRR')&(full.Condition1=='Norm')&(full.Condition2=='Norm')].MSZoning.value_counts()
full[(full.Neighborhood=='Mitchel')&(full.Condition1=='Artery')&(full.Condition2=='Norm')].MSZoning.value_counts()
_ = full.set_value((full.MSZoning.isnull())&(full.Neighborhood=='IDOTRR'),'MSZoning','RM')

_ = test.set_value((test.MSZoning.isnull())&(test.Neighborhood=='IDOTRR'),'MSZoning','RM')



_ = full.set_value((full.MSZoning.isnull())&(full.Neighborhood=='Mitchel'),'MSZoning','RL')

_ = test.set_value((test.MSZoning.isnull())&(test.Neighborhood=='Mitchel'),'MSZoning','RL')
full[(full.MasVnrType.isnull())|(full.MasVnrArea.isnull())][['MasVnrType','MasVnrArea']].drop_duplicates()
full[full.MasVnrArea==198].MasVnrType.value_counts()
_ = full.set_value(full.MasVnrArea.isnull(),['MasVnrType','MasVnrArea'],['None',0])

_ = train.set_value(train.MasVnrArea.isnull(),['MasVnrType','MasVnrArea'],['None',0])

_ = test.set_value(test.MasVnrArea.isnull(),['MasVnrType','MasVnrArea'],['None',0])



_ = full.set_value(full.MasVnrType.isnull(),'MasVnrType','Stone')

_ = train.set_value(train.MasVnrType.isnull(),'MasVnrType','Stone')

_ = test.set_value(test.MasVnrType.isnull(),'MasVnrType','Stone')
full.MiscFeature.value_counts(dropna=False)
#_ = full.drop('MiscFeature', axis=1, inplace=True)

#_ = train.drop('MiscFeature', axis=1, inplace=True)

#_ = test.drop('MiscFeature', axis=1, inplace=True)

_ = full.set_value(full.MiscFeature.isnull(),'MiscFeature','NA')

_ = train.set_value(train.MiscFeature.isnull(),'MiscFeature','NA')

_ = test.set_value(test.MiscFeature.isnull(),'MiscFeature','NA')
full[full.PoolQC.isnull()][['PoolQC','PoolArea']].drop_duplicates()
full[(full.PoolArea>=300)&(full.PoolArea<=600)][['PoolQC','PoolArea']].drop_duplicates()
_ = full.set_value((full.PoolQC.isnull())&(full.PoolArea==0),'PoolQC','NA')

_ = train.set_value((train.PoolQC.isnull())&(train.PoolArea==0),'PoolQC','NA')

_ = test.set_value((test.PoolQC.isnull())&(test.PoolArea==0),'PoolQC','NA')



_ = full.set_value(full.PoolQC.isnull(),'PoolQC','Gd')

_ = train.set_value(train.PoolQC.isnull(),'PoolQC','Gd')

_ = test.set_value(test.PoolQC.isnull(),'PoolQC','Gd')
full[full.SaleType.isnull()][['SaleType','SaleCondition']]
full[full.SaleCondition=='Normal'].SaleType.value_counts()
_ = full.set_value(full.SaleType.isnull(),'SaleType','WD')

_ = train.set_value(train.SaleType.isnull(),'SaleType','WD')

_ = test.set_value(test.SaleType.isnull(),'SaleType','WD')
full.Utilities.value_counts()
_ = full.set_value(full.Utilities.isnull(),'Utilities','AllPub')

_ = train.set_value(train.Utilities.isnull(),'Utilities','AllPub')

_ = test.set_value(test.Utilities.isnull(),'Utilities','AllPub')
full.isnull().sum().sum(), train.isnull().sum().sum(), test.isnull().sum().sum()
_ = train[['Id','SalePrice']].plot(kind='scatter', x='Id',y='SalePrice', figsize=(10,4))
_ = full[['Id','LotArea']].plot(kind='scatter', x='Id',y='LotArea', figsize=(10,4))

_ = plt.plot(full.Id,[100000]*len(full.Id))

train.LotArea.min(),train.LotArea.max()



#_ = full[['Id','TotalArea']].plot(kind='scatter', x='Id',y='TotalArea', figsize=(10,4))

#_ = plt.plot(full.Id,[100000]*len(full.Id))

#train.TotalArea.min(),train.TotalArea.max()
_ = full.drop((full[full.LotArea>=100000]).index, axis=0, inplace=True)

_ = train.drop((train[train.LotArea>=100000]).index, axis=0, inplace=True)



#_ = full.drop((full[full.TotalArea>=100000]).index, axis=0, inplace=True)

#_ = train.drop((train[train.TotalArea>=100000]).index, axis=0, inplace=True)
cont_features=['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF',

               '2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',

               '3SsnPorch','ScreenPorch','PoolArea','MiscVal','SalePrice']
train_log = train.copy()

test_log = test.copy()



for col in cont_features:

    if col=='SalePrice':

        None

        train_log[col] = np.log1p(train_log[col])

    else:

        train_log[col] = np.log1p(train_log[col])

        test_log[col] = np.log1p(test_log[col])
train_log[cont_features].head()
train_log.shape, test_log.shape
full = pd.concat([train_log,test_log], ignore_index=True)
full.drop(cont_features, axis=1).head()
full_dummy = pd.get_dummies(full)
full_dummy.head()
full_dummy.shape
full_dummy.drop(cont_features, axis=1).head()
X_train = full_dummy[full_dummy.SalePrice.notnull()].drop('SalePrice', axis=1)

y_train = full_dummy[full_dummy.SalePrice.notnull()][['SalePrice']]

X_test = full_dummy[full_dummy.SalePrice.isnull()].drop('SalePrice', axis=1)
X_train.head()
y_train.head()
X_train.shape, y_train.shape, X_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_train_scaled
#from sklearn.svm import SVR

#from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import GridSearchCV



#param = {'C':[.1,1,10,100,200,500] ,'gamma':[.0001,.001,.01,.1]}

#svr_grid = GridSearchCV(SVR(), param_grid=param ,cv=5).fit(X_train_scaled ,y_train.values.ravel())



#print("grid search score : ",svr_grid.score(X_train_scaled, y_train.values.ravel()))

#print("grid search best score : ",svr_grid.best_score_)

#print("grid search best parameter : ",svr_grid.best_params_)
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score



svr = SVR(C=200, gamma=.0001)

svr_score = cross_val_score(svr, X_train_scaled, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error')

print("SVR RMSE : ", svr_score.mean())
#svr_reg = SVR(C=200, gamma=.0001).fit(X_train_scaled,y_train.values.ravel())

#y_train_pred_svr = pd.DataFrame({'Id':X_test.Id.values.ravel(),

#                             'SalePrice': np.expm1(svr_reg.predict(X_test_scaled))},

#                            index=X_test.index

#                           )

#y_train_pred_svr.to_csv("SVR_submission.csv",index=False)
#from sklearn.model_selection import cross_val_score

#from sklearn.linear_model import LinearRegression



#lin_reg = LinearRegression()

#lin_scores = cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

#print("Linear regression RMSE : ",lin_scores.mean())
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge



alpha = [.0001,.001,.01,.1,1,10,100,1000]

ridge_score = []

ridge_score_imp_features = []



for a in alpha:

    ridge_reg = Ridge(alpha=a)

    ridge_score.append(cross_val_score(ridge_reg, X_train, y_train, cv=5))

    

ridge_score = np.mean(ridge_score, axis=1)



_ = plt.plot(np.arange(len(alpha)),ridge_score, label='all features')

_ = plt.xticks(np.arange(len(alpha)),alpha)

_ = plt.legend(loc=3)

_ = plt.xlabel("alpha")

_ = plt.ylabel('Train Accuracy')
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge



ridge_reg = Ridge(alpha=10)

ridge_scores = cross_val_score(ridge_reg, X_train, y_train, cv=5)

ridge_rmse = cross_val_score(ridge_reg, X_train, y_train, cv=5, scoring="neg_mean_squared_error")

print("Ridge regression mean score : ",ridge_scores.mean())

print("Ridge regression RMSE : ",ridge_rmse.mean())
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



alpha = [.0001,.001,.01,.1,1,10,100,1000]

lasso_score = []



param = {'alpha':[.0001,.001,.01,.1,1,10,100,1000]}



for a in alpha:

    lasso = Lasso(alpha=a)

    lasso_score.append(cross_val_score(lasso,X_train, y_train, cv=5))



lasso_score = np.mean(lasso_score, axis=1)



_ = plt.plot(np.arange(len(alpha)),lasso_score, label='all features')

_ = plt.xticks(np.arange(len(alpha)),alpha)

_ = plt.legend(loc=3)

_ = plt.xlabel("alpha")

_ = plt.ylabel('Train Accuracy')
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Lasso



lasso_reg = Lasso(alpha=0.0007, max_iter=10000)

lasso_scores = cross_val_score(lasso_reg, X_train, y_train, cv=5)

lasso_RMSE = cross_val_score(lasso_reg, X_train, y_train, cv=5, scoring="neg_mean_squared_error")

print("Lasso regression mean score : ",lasso_scores.mean())

print("Lasso regression RMSE : ",lasso_RMSE.mean())
lasso_reg = Lasso(alpha=.0007, max_iter=10000).fit(X_train,y_train)

#y_train_pred_lasso = pd.DataFrame({'Id':X_test.Id.values.ravel(),

#                             'SalePrice': np.expm1(lasso_reg.predict(X_test))},

#                            index=X_test.index

#                           )

#y_train_pred_lasso.to_csv("Lasso_submission.csv",index=False)
#from sklearn.tree import DecisionTreeRegressor

#from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import GridSearchCV



#param ={'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'min_samples_leaf':[1, 5,10,20,50,100,200,400]}



#dtree_grid = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid=param, cv=5).fit(X_train, y_train)



#print("Best parameter : ", dtree_grid.best_params_)

#print("Best Score : ",dtree_grid.best_score_)

#print(dtree_grid.best_estimator_)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



dtree = DecisionTreeRegressor(max_depth=11, min_samples_leaf=10)

dtree_score = cross_val_score(dtree, X_train, y_train, cv=5)

dtree_RMSE = cross_val_score(dtree, X_train, y_train, cv=5, scoring='neg_mean_squared_error')



print("Decision tree regression mean score : ",dtree_score.mean())

print("Decision tree regression RMSE : ",dtree_RMSE.mean())
#from sklearn.ensemble import RandomForestRegressor

#from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import GridSearchCV



#param = {'max_depth':[1,3,5,7,9], 'n_estimators':[10,100,200,400]}





#rtree_grid = GridSearchCV(RandomForestRegressor(random_state=0), param_grid=param, cv=5).fit(X_train, y_train.values.ravel())



#print("Best parameter : ", rtree_grid.best_params_)

#print("Best Score : ",rtree_grid.best_score_)

#print(rtree_grid.best_estimator_)
#from sklearn.ensemble import RandomForestRegressor

#from sklearn.model_selection import cross_val_score



#rtree = RandomForestRegressor(max_depth=9, n_estimators=400)

#r_tree_scores = cross_val_score(rtree, X_train, y_train.values.ravel(), cv=5)

#r_tree_RMSE = cross_val_score(rtree, X_train, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error')



#print("Random forest regression mean score : ",r_tree_scores.mean())

#print("Random forest regression RMSE : ",r_tree_RMSE.mean())
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

rtree = RandomForestRegressor(max_depth=9, n_estimators=400).fit(X_train, y_train.values.ravel())

pred_lasso_rtree =np.expm1(rtree.predict(X_train))

print("R2 score",r2_score(np.expm1(y_train.values.ravel()),pred_lasso_rtree))

print("mean_squared_error :",mean_squared_error(np.expm1(y_train.values.ravel()),pred_lasso_rtree))
from sklearn.metrics import r2_score, mean_squared_error

pred_lasso_rtree = np.expm1(lasso_reg.predict(X_train))

print("R2 score",r2_score(np.expm1(y_train.values.ravel()),pred_lasso_rtree))

print("mean_squared_error :",mean_squared_error(np.expm1(y_train.values.ravel()),pred_lasso_rtree))
from sklearn.metrics import r2_score, mean_squared_error

pred_lasso_rtree = np.expm1(rtree.predict(X_train))*.5 + np.expm1(lasso_reg.predict(X_train))*.5

print("R2 score",r2_score(np.expm1(y_train.values.ravel()),pred_lasso_rtree))

print("mean_squared_error :",mean_squared_error(np.expm1(y_train.values.ravel()),pred_lasso_rtree))


y_test_pred_rtree = pd.DataFrame({'Id':test.Id.values.ravel(),

                             'SalePrice': (np.expm1(rtree.predict(X_test))*.5 + np.expm1(lasso_reg.predict(X_test))*.5)

                           },                            

                            index=X_test.index

                           )

y_test_pred_rtree.to_csv("Rtree_lasso_submission.csv",index=False)