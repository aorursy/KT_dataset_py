#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = pd.concat([train.drop('SalePrice', axis = 1), test])
SalePrice = train['SalePrice']
plt.subplot(211)
sns.barplot(x = 'YrSold', y = 'SalePrice',data = train)
plt.subplot(212)
sns.barplot(x = 'MoSold', y = 'SalePrice',data = train)

combine.YrSold.value_counts()
combine.MoSold.value_counts()
train.groupby('BsmtFinType1')['BsmtFinSF1'].describe()
train[train.BsmtFinType1.isin(['ALQ','BLQ','GLQ','LwQ','Rec'])]['BsmtFinSF1'].describe()
train[['BsmtFinType1','BsmtFinSF1','BsmtFinType2', 'BsmtFinSF2', 'BsmtQual',
       'BsmtCond', 'BsmtExposure','BsmtUnfSF', 'TotalBsmtSF']][(train.BsmtFinSF1 < 20) & (train.BsmtFinSF1 >0)]
plt.figure(figsize =[6,5])
plt.subplot(211)
sns.distplot(train[train.TotalBsmtSF !=0]['TotalBsmtSF'], kde= False, color = 'b')
plt.subplot(212)
plt.scatter(train[train.TotalBsmtSF!=0]['TotalBsmtSF'], train[train.TotalBsmtSF !=0]['SalePrice'])
train[train['BsmtFinType2'].isin(['ALQ','BLQ','GLQ','LwQ','Rec'])]['BsmtFinSF2'].describe()
train[train.BsmtUnfSF !=0]['BsmtUnfSF'].describe()
train[train.TotalBsmtSF !=0]['TotalBsmtSF'].describe()
combine['HasBasement'] = np.where(combine['TotalBsmtSF'] >0, 1,0)
combine['TotalBsmtSF_cat']  = np.where(combine.TotalBsmtSF >1309, 4, (np.where(combine.TotalBsmtSF >1004,3,(np.where(combine.TotalBsmtSF>810,2, (np.where(combine.TotalBsmtSF>104,1,0)))))))
combine['BsmtFinSF1_cat'] = np.where(combine.BsmtFinSF1 >867, 4,(np.where(combine.BsmtFinSF1>604,3,(np.where(combine.BsmtFinSF1>371,2,np.where(combine.BsmtFinSF1>1,1,0))))))
#create new feature combining SF and quality of the Basement:
combine['BsmtFinType1_qual'] = combine['BsmtFinType1'].astype(str) + combine['BsmtFinSF1_cat'].astype(str)
combine['BsmtOverall_qual'] = combine['BsmtQual'].astype(str) + combine['TotalBsmtSF_cat'].astype(str)

combine['BsmtFinType1_qual'].value_counts()
combine.LotArea.describe()
combine['LotArea_cat'] = np.where(combine.LotArea >11570, 4, (np.where(combine.LotArea >9453, 3,(np.where(combine.LotArea>7478,2,1)))))

#Categorize Lot Frontage
combine.LotFrontage.describe()
combine['LotFrontage_cat'] = np.where(combine.LotFrontage > 80, 4,(np.where(combine.LotFrontage >68, 3,(np.where(combine.LotFrontage>59,2,1)))))
#See the year built feature
combine.YearBuilt.value_counts().head(10)
combine['Age'] = 2010- combine.YearBuilt
sns.boxplot(x = 'Age', data = combine)
combine.Age.describe()
#Create feature Age cat
combine['Age_cat'] = np.where(combine.Age >56,4,(np.where(combine.Age >37,3,(np.where(combine.Age >9,2,1)))))

#more about year built and remodelling
cond = (combine['YearRemodAdd'] - combine['YearBuilt']>0)
a = combine['YearRemodAdd']- combine['YearBuilt']
print(a[cond].describe())
print(a.describe())
#We can see that around 28 year after the house is built it will be remodel
sns.distplot(a[cond], kde = False, color = 'b')
cond1 =(combine['YearRemodAdd']- combine['YearBuilt'] > 1)
cond2 =(combine['YearRemodAdd']- combine['YearBuilt'] > 2)
plt.subplot(211)
sns.distplot(a[cond1], kde = False, color = 'b')
plt.subplot(212)
sns.distplot(a[cond2], kde = False, color = 'b')
#now see the description of the data:
a[cond1].describe()
Age_train= 2010- train['YearBuilt']
Age_train_Remod = 2010 - train['YearRemodAdd']
cond2 = (train['YearBuilt'] == train['YearRemodAdd'])
plt.figure(figsize= [5,20])
plt.subplot(311)
plt.scatter(Age_train, train['SalePrice'])
plt.subplot(312)
plt.scatter(Age_train_Remod, train['SalePrice'])
plt.subplot(313)
plt.scatter(Age_train_Remod[cond2], train['SalePrice'][cond2])
combine['OldHouse'] = 2010 - combine['YearRemodAdd'] >30
combine['NewHouse'] = np.where(2010- combine['YearRemodAdd'] >30, 0,(np.where(combine['YearRemodAdd'] !=combine['YearBuilt'],1,2)))
#combine.iloc[:len(train)] = train
sns.barplot(combine.iloc[:len(train)]['NewHouse'], SalePrice)
plt.xlabel('Level of New House')
combine['MasVnrArea'][combine.MasVnrArea !=0].describe()
combine['MasVnrArea_cat'] = np.where(combine.MasVnrArea >327,4,(np.where(combine.MasVnrArea >202,3,(np.where(combine.MasVnrArea >120,2,(np.where(combine.MasVnrArea>0,1,0)))))))
#To omit the case of no 2nd floor, I create the a1 DF
plt.figure(figsize = [10,5])
ax1 = plt.subplot(211)
sns.boxplot(x = '1stFlrSF', data = combine)
plt.subplot(212,sharex = ax1)
a1 = combine[combine['2ndFlrSF'] !=0]
ax2 = sns.boxplot(x = '2ndFlrSF', data = a1)
print(combine['1stFlrSF'].describe())
print('-------')
print(a1['1stFlrSF'].describe())
combine['LowQualFinSF_rate'] = combine['LowQualFinSF']/(combine['1stFlrSF'] + combine['2ndFlrSF'])
sns.distplot(combine['LowQualFinSF_rate'], kde = False, color = 'b')
combine['LowQualFinSF_rate'].describe()
#Most of the house doesn't have low quality SF. Let's see the range
sns.distplot(combine['LowQualFinSF_rate'][combine.LowQualFinSF_rate !=0], kde = False, color = 'b')
combine['LowQualFinSF_rate'][combine.LowQualFinSF_rate !=0].describe()
#Just 40 house has Low quality SF. 50% of them has the SF less than 10%. so I will create the feature HasLowQualFinSF
combine['HasLowQualFinSF']  = combine['LowQualFinSF'] > 0

#Let's check again the assumption of HasLowQualSF will drag down the House Price
sns.barplot(combine.iloc[:len(train)]['HasLowQualFinSF'], SalePrice)
#To be expanded later: I can test the significance test be z-test
combine[['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea']][combine['LowQualFinSF']!=0].sample(3)
#I have the assumption that people prefer the higher 1stFlrSF percentage. Let do it a check
a2 = train['1stFlrSF']/train['GrLivArea']
plt.scatter(a2, train['SalePrice'])
print(combine['BsmtFullBath'].value_counts())
print(combine['BsmtHalfBath'].value_counts())
print(combine['FullBath'].value_counts())
print(combine['HalfBath'].value_counts())

#number of fullbath, halfbath
FullBath = combine['BsmtFullBath'] + combine['FullBath']
HalfBath = combine['BsmtHalfBath'] + combine['HalfBath']
TotalBath = FullBath + HalfBath
plt.figure(figsize = [12,10])
sns.set_style('whitegrid')
plt.subplot(311)
sns.barplot(FullBath.iloc[:len(train)], SalePrice)
plt.xlabel('Number of Full Bath')
plt.subplot(312)
sns.barplot(HalfBath.iloc[:len(train)], SalePrice)
plt.xlabel('Number of Half Bath')
plt.subplot(313)
sns.barplot(TotalBath.iloc[:len(train)], SalePrice)
plt.xlabel('Total number of Bath')
#check the assumption
pd.crosstab(TotalBath, combine['NewHouse']).T
#now create the category of Number of Bathroom
combine['TotalBath'] = TotalBath
combine['TotalBath_cat'] = np.where(combine.TotalBath > 3, 4, (np.where(combine.TotalBath ==3,3,(np.where(combine.TotalBath==2,2,(np.where(combine.TotalBath ==1,1,0)))))))
sns.barplot(combine['TotalBath_cat'].iloc[:len(train)], SalePrice)
#look at the BedroomAbvGr feature
combine.BedroomAbvGr.value_counts()
combine.BedroomAbvGr.isnull().sum()
combine.KitchenAbvGr.value_counts()
sns.barplot(combine['KitchenAbvGr'].iloc[:len(train)], SalePrice)
pd.crosstab(combine['KitchenAbvGr'], combine['NewHouse'])
#Let's take a look at the KitchenQual feature:
sns.barplot(combine['KitchenQual'].iloc[:len(train)],SalePrice, order=['Ex','Gd','TA','Fa'])
#Look at the TotRmsAbvGrd (Total Room above grade)
sns.barplot(combine['TotRmsAbvGrd'].iloc[:len(train)], SalePrice)
plt.xlabel('Total Room above Grade')
combine['TotRmsAbvGrd_cat'] = np.where(combine.TotRmsAbvGrd >=9,9, combine.TotRmsAbvGrd)
sns.barplot(combine['TotRmsAbvGrd_cat'].iloc[:len(train)], SalePrice)
train[['YearBuilt','YearRemodAdd','TotRmsAbvGrd','SalePrice']][train.TotRmsAbvGrd.isin(['11','12','14'])]
#explore functionality
sns.barplot(train['Functional'], SalePrice)
print(combine.Functional.describe())
#Create category Functional_cat
combine['Functional_cat'] = np.where(combine.Functional =='Typ',1,0)
sns.barplot(combine['Functional_cat'].iloc[:len(train)],SalePrice)
combine.Fireplaces.value_counts()
sns.barplot(train['Fireplaces'], SalePrice)
combine['Fireplaces_cat'] = np.where(combine['Fireplaces']>1,2,(np.where(combine['Fireplaces']==1,1,0)))
sns.barplot(combine['Fireplaces_cat'].iloc[:len(train)],SalePrice)
#take a look at the FireplaceQu
sns.barplot(combine['FireplaceQu'].iloc[:len(train)], SalePrice, order=['Ex','Gd','TA','Fa','Po'])
Fire_na_number = train['FireplaceQu'].isnull().sum()
int(Fire_na_number)
print('The number of NA value in FireplaceQu feature is {}'.format(Fire_na_number))
combine['FireplaceQu']  = combine['FireplaceQu'].fillna(0)
#take a look again
sns.barplot(combine['FireplaceQu'].iloc[:len(train)], SalePrice, order = ['Ex','Gd','TA','Po',0])
combine[['GarageType','GarageYrBlt','GarageCars','GarageFinish','GarageArea','GarageQual','GarageCond']].sample(5)
print(combine['GarageType'].value_counts())
print('--------')
print(combine['GarageType'].isnull().sum())
combine['GarageType'] = combine['GarageType'].fillna(0)
sns.barplot(combine['GarageType'].iloc[:len(train)], SalePrice)
combine['GarageAge']  = 2010 -combine['GarageYrBlt']
plt.scatter(combine['GarageAge'].iloc[:len(train)], SalePrice)
combine['GarageAge'].describe()
combine[combine['GarageAge']<0][['GarageYrBlt','YearBuilt','GarageCond','GarageQual']]
combine['GarageYrBlt'].loc[1132]
#Imputing (the first to be the same with 1930 but the second is with 2007)
combine['GarageYrBlt'].loc[1132] = 2007
combine['GarageYrBlt'].iloc[1132] = 1930
#check again
combine['GarageYrBlt'].loc[1132]

combine['GarageAge'] = 2010 - combine['GarageYrBlt']
#take a look at the GarageAge and SalePrice
plt.scatter(combine['GarageAge'].iloc[:len(train)], SalePrice)
combine['GarageAge'].describe()
combine['GarageAge_cat'] = np.where(combine.GarageAge> 50,3,(np.where(combine.GarageAge>31,2,(np.where(combine.GarageAge> 8,1,0)))))
sns.barplot(combine.GarageAge_cat.iloc[:len(train)], SalePrice)
combine['GarageAge'] = combine['GarageAge'].fillna(-1)
combine['GarageAge'].isnull().sum()
combine['GarageAge_cat'] = np.where(combine.GarageAge> 50,3,(np.where(combine.GarageAge>31,2,(np.where(combine.GarageAge> 8,1,(np.where(combine.GarageAge >=0,0,4)))))))
sns.barplot(combine.GarageAge_cat.iloc[:len(train)], SalePrice)
#Take a look at the WoodDeck SF
combine.WoodDeckSF.isnull().sum()
plt.scatter(train['WoodDeckSF'], SalePrice)
combine['WoodDeckSF'][combine.WoodDeckSF !=0].describe()
combine['WoodDeckSF_cat'] = np.where(combine.WoodDeckSF >240, 4, (np.where(combine.WoodDeckSF >171,3,(np.where(combine.WoodDeckSF>121,2,(np.where(combine.WoodDeckSF>0,1,0)))))))
sns.barplot(combine['WoodDeckSF_cat'].iloc[:len(train)], SalePrice)
#Take a look at the OpenPorchSF
print(combine['OpenPorchSF'].isnull().sum())
print((combine['OpenPorchSF'] ==0).sum())
#devide into 5 feature as other case
combine['OpenPorchSF'][combine.OpenPorchSF!=0].describe()
combine['OpenPorchSF_cat'] =np.where(combine.OpenPorchSF >110,4,(np.where(combine.OpenPorchSF>63,3,(np.where(combine.OpenPorchSF>39,2,(np.where(combine.OpenPorchSF>0,1,0)))))))
#take a look the the relation with SalePrice
sns.barplot(combine['OpenPorchSF_cat'].iloc[:len(train)],SalePrice)
combine[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']].sample(5)
combine['PorchSF'] = combine['OpenPorchSF'] + combine['EnclosedPorch'] + combine['3SsnPorch'] + combine['ScreenPorch']
(combine['PorchSF'] == 0).sum()
combine['PorchSF'][combine.PorchSF != 0].describe()
combine['PorchSF_cat'] = np.where(combine.PorchSF >180,4, (np.where(combine.PorchSF>96,3,(np.where(combine.PorchSF>48,2,(np.where(combine.PorchSF>0,1,0)))))))
sns.barplot(combine['PorchSF_cat'].iloc[:len(train)], SalePrice)
#Take a look at the Pool related features
sns.barplot(train['PoolQC'], SalePrice)
combine['PoolQC'].notnull().sum()
(combine['PoolArea']!=0).sum()
#a look at the Fence feature
print(combine['Fence'].isnull().sum())
print(combine['Fence'].notnull().sum())
combine['Fence'].value_counts()
#Imputing Null
combine['Fence'] = combine['Fence'].fillna(0)
sns.barplot(combine['Fence'].iloc[:len(train)], SalePrice)
#MiscFeature and MiscVal I will let it under progress
for i in combine.columns:
    print(i)
#Fill in some other missing value
combine[['BsmtExposure','BsmtQual','BsmtCond','GarageCond','GarageQual','GarageFinish']] = combine[['BsmtExposure','BsmtQual','BsmtCond','GarageCond','GarageQual','GarageFinish']].fillna(0)
combine['HasPool'] = np.where(combine.PoolQC.notnull == True, 1,0)
combine['HasMiscFeature'] = np.where(combine.MiscFeature.notnull == True,1,0)
combine['Alley']  = combine['Alley'].fillna(0)
combine['MasVnrType']= combine['MasVnrType'].fillna(0)
combine_model = combine.drop(['Id','LotFrontage','LotArea','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','TotalBath','KitchenAbvGr','TotRmsAbvGrd','Functional','Fireplaces','GarageAge','GarageYrBlt','PoolQC','MiscFeature'], axis = 1)
combine_model.columns
#check the missing value again
combine_model.isnull().sum().sort_values(ascending = False).head(20)
combine_model.isnull().sum().sort_values(ascending = False).head(20).index
#Count the number of value in each category
#len(combine_model.MSSubClass.value_counts().index) = the number of categories in each feature
for col in combine_model.columns:
    combine_model[col] = combine_model[col].astype("category")
    combine_model[col].cat.categories = np.arange(len(combine_model[col].value_counts().index))
    combine_model[col] =combine_model[col].astype("int")
#get the train and test dataframe
train = combine_model.iloc[:len(train)]
test = combine_model.iloc[len(train):]
train['SalePrice'] = SalePrice
#train test split
training, testing = train_test_split(train, test_size=0.2, random_state=0)


cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
       'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
       'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       'GrLivArea', 'BedroomAbvGr', 'KitchenQual', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'Fence', 'MiscVal', 'MoSold', 'YrSold',
       'SaleType', 'SaleCondition', 'HasBasement', 'TotalBsmtSF_cat',
       'BsmtFinSF1_cat', 'BsmtFinType1_qual', 'BsmtOverall_qual',
       'LotArea_cat', 'LotFrontage_cat', 'Age', 'Age_cat', 'OldHouse',
       'NewHouse', 'MasVnrArea_cat', 'LowQualFinSF_rate', 'HasLowQualFinSF',
       'Functional_cat', 'Fireplaces_cat', 'GarageAge_cat', 'WoodDeckSF_cat',
       'OpenPorchSF_cat', 'PorchSF', 'PorchSF_cat', 'TotalBath_cat',
       'TotRmsAbvGrd_cat', 'HasPool', 'HasMiscFeature']
tcols = np.append(['SalePrice'], cols)
df = training.loc[:,tcols].dropna()
X = df.loc[:, cols]
y = np.ravel(df.loc[:, ['SalePrice']])
#import
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#model
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)
    rmse= np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#Lasso Regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

#Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#Gradient Boosting
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
#XGboost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
#LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)
print("\nLasso score: {:.1f} ({:.1f})\n".format(score.mean(), score.std()))
print(cross_val_score(lasso, X.values, y, scoring= "neg_mean_squared_error").mean())
lasso.fit(X,y)
SalePrice_pred_X = lasso.predict(X.values)
dif = SalePrice_pred_X - y
a = np.sqrt(dif**2)
(a/y).sum()/a.shape[0]

score = rmsle_cv(ENet)
print("\nEnet score: {} ({})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
clf = lasso  
clf.fit(X,y)
df2 = test.loc[:,cols].fillna(method='pad')
SalePrice_pred = clf.predict(df2)
submit = pd.DataFrame({'Id' : combine['Id'].iloc[len(train):],
                       'SalePrice': SalePrice_pred})
submit.to_csv('submit_Lasso.csv', index= False)







