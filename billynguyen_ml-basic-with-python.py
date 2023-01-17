import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

r_submission = pd.read_csv("../input/submission-glm/submission_glmnet.csv")
train.head()
# set up aesthetic design

plt.style.use('seaborn')

sns.set_style('whitegrid')



# create NA plot for train data

plt.subplots(0,0,figsize = (15,3)) # positioning for 1st plot

train.isnull().mean().sort_values(ascending = False).plot.bar(color = 'blue')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('Missing values average per columns in TRAIN data', fontsize = 20)

plt.show()



# create NA plot for test data

plt.subplots(1,0,figsize = (15,3))

test.isnull().mean().sort_values(ascending = False).plot.bar(color = 'blue')

plt.axhline(y = 0.1, color = 'r', linestyle = '-')

plt.title('Missing values average per columns in TEST data', fontsize = 20)

plt.show()
train['MSSubClass'] = train['MSSubClass'].astype(object)

test['MSSubClass'] = test['MSSubClass'].astype(object)
categorical_cols = [col for col in train.columns if train[col].dtypes == "object"]

print("number of categorical columns is: ")

print(len(categorical_cols))
# combine train and test to pre-processing

df = train.drop(['SalePrice'], axis = 1)

df = df.append(test)

df = df.drop(['Id'], axis = 1)

df.shape
# fill NA value by missing

for col in categorical_cols:

    df[col] = df[col].fillna("missing")
# Ex > Gd > TA > Fa > Po

ord_cols_1 = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC', 'KitchenQual',

             'FireplaceQu','GarageQual','GarageCond']



mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,'missing':0}



for col in ord_cols_1:

    df[col] = df[col].map(mapping)
# Gd > Av > Mn > No > NA (missing)

ord_cols_2 = ['BsmtExposure']



mapping = {'Gd':4, 'Av': 3, 'Mn': 2, 'No': 1, 'missing': 0}



for col in ord_cols_2:

    df[col] = df[col].map(mapping)     
# GLQ > ALQ > BLQ > Rec > LwQ > Unf > NA (missing)

ord_cols_3 = ['BsmtFinType1', 'BsmtFinType2']



mapping = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'missing': 0}



for col in ord_cols_3:

    df[col] = df[col].map(mapping)   
# Functional

ord_cols_4 = ['Functional']



mapping = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3,

          'Sev': 2, 'Sal': 1}



for col in ord_cols_4:

    df[col] = df[col].map(mapping)   
# garageFinish

ord_cols_5 = ['GarageFinish']



mapping = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'missing': 0}



for col in ord_cols_5:

    df[col] = df[col].map(mapping)   
# fence

mapping = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'missing': 0}



df['Fence'] = df['Fence'].map(mapping)



# LotShape

mapping = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1, 'missing': 4}

df['LotShape'] = df['LotShape'].map(mapping)



# LandContour

mapping = {'Lvl': 4, 'Bnk': 3, 'HLS':2,'Low':1,'missing': 4}

df['LandContour'] = df['LandContour'].map(mapping)



# Utilities

mapping = {'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1,'missing':4}

df['Utilities'] = df['Utilities'].map(mapping)

# LandSlope

mapping = {'Gtl':3,'Mod':2,'Sev':1}

df['LandSlope'] = df['LandSlope'].map(mapping)



# Heating

mapping = {'Wall':'other','OthW':'other','Floor':'other','Grav':'other'}

df['Heating'] = df['Heating'].map(mapping)



# Electrical

mapping = {'SBrkr':5,'FuseA':4,'FuseF':3,'FuseP':2,'Mix':1}

df['Electrical'] = df['Electrical'].map(mapping)



# PavedDrive

mapping = {'Y':3,'P':2,'N':1}



#MiscFeature

mapping = {'TenC':'high','Elev':'high'}
# binary categorical variables

df['CentralAir'] = df['CentralAir'].replace("N","0")

df['CentralAir'] = df['CentralAir'].replace("Y","1")

df['CentralAir'] = df['CentralAir'].astype(int)
# check again categorical cols

print("number of categorical cols before is: ")

print(len(categorical_cols))



categorical_cols = [col for col in df.columns if df[col].dtypes == "object"]

print("new number of categorical columns is: ")

print(len(categorical_cols))
# one hot encoder for remain categorical variables

cat_df = df.loc[:, categorical_cols]

cat_df = cat_df.drop(['Neighborhood','Condition2',

                     'Exterior1st','Exterior2nd',

                      'PoolQC','Alley'

                     ], axis=1)

cat_df = pd.get_dummies(cat_df)



print(cat_df.shape)

cat_df.head()
# explore continuous variables

continuous_variables = [col for col in df.columns if df[col].dtype != "object"]

print("the number of continuous variables is: ")

print(len(continuous_variables))
plt.subplots(1,0,figsize = (15,3))

df.loc[:,continuous_variables].isnull().mean().sort_values(ascending = False).plot.bar(color = 'blue')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('Missing values average per columns in FULL data', fontsize = 20)

plt.show()
num = train.select_dtypes(exclude = 'object')

numcorr = num.corr()

f, ax = plt.subplots(figsize = (19,1)) # set figure size

sns.heatmap(numcorr.sort_values(by = 'SalePrice', ascending = False).head(1),annot = True, fmt = ".2f")

plt.show()
plt.subplots(1,0,figsize = (15,3))

numcorr['SalePrice'].sort_values(ascending = False).to_frame().plot.bar(color = 'blue')

plt.axhline(y = 0.5, color = 'r', linestyle = '-')

plt.title('Corrplot vs SalePrice')

plt.show()
Num=numcorr['SalePrice'].sort_values(ascending=False).to_frame()

cm = sns.light_palette("orange", as_cmap=True)

s = Num.style.background_gradient(cmap=cm)

s
df = df.drop(['MoSold',

              #'3SsnPorch','BsmtFinSF2','BsmtHalfBath',

             #'MiscVal',

              'LowQualFinSF','YrSold'], axis = 1)
fig = plt.figure(figsize = (15,10))

ax1 = plt.subplot2grid((2,2),(0,0))

ax1.set_xlim([0,7000])

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'], color = ('yellowgreen'))

plt.axvline(x = 4600, color = 'r', linestyle = '-')

plt.title('GrLivArea - SalePrice', fontsize = 15, weight = 'bold')



ax1 = plt.subplot2grid((2,2),(0,1))

plt.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'], color = ('red'))

plt.axvline(x = 4600, color = 'r', linestyle = '-')

plt.title('TotalBsmtSF - SalePrice', fontsize = 15, weight = 'bold')



ax1 = plt.subplot2grid((2,2),(1,0))

plt.scatter(x = train['GarageArea'], y = train['SalePrice'], color = ('deepskyblue'))

plt.axvline(x = 4600, color = 'r', linestyle = '-')

plt.title('GarageArea - SalePrice', fontsize = 15, weight = 'bold')



ax1 = plt.subplot2grid((2,2),(1,1))

plt.scatter(x = train['MasVnrArea'], y = train['SalePrice'], color = ('gold'))

plt.axvline(x = 4600, color = 'r', linestyle = '-')

plt.title('MasVnrArea - SalePrice', fontsize = 15, weight = 'bold')
continuous_variables = [col for col in df.columns if df[col].dtype != "object"]
# specific replacement NA value

df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1980)



# mass replacement NA value by median

for col in continuous_variables:

    df[col] = df[col].fillna(0)
# combine df

df = df.drop(categorical_cols, axis = 1)

df = pd.concat([df, cat_df], axis = 1, join = 'inner')

print(df.shape)

df.head()
#df['Age_1'] = df['YrSold'] - df['YearBuilt']

#df['Age_2'] = df['YrSold'] - df['YearRemodAdd']

df['year_qual'] = df['YearBuilt']*df['OverallQual']

df['year_r_qual'] = df['YearRemodAdd']*df['OverallQual']

df['qual_bsmt'] = df['OverallQual']*df['TotalBsmtSF']

df['livarea_qual'] = df['OverallQual']*df['GrLivArea']

df['qual_bath'] = df['OverallQual']*df['FullBath']

#df['qual_ext'] = df['OverallQual']*df['ExterCond']

df['garage_qual'] = df['GarageArea']*df['GarageQual']



df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea'] + df['GarageArea']

df['Bathrooms'] = df['FullBath'] + df['HalfBath']*0.5

df['YearAverage'] = (df['YearRemodAdd'] + df['YearBuilt'])/2
log_cols = ['livarea_qual','qual_bsmt',

            'LotArea',

           'MasVnrArea', 'TotalBsmtSF',

           'GarageArea','GrLivArea',

            ]

for col in log_cols:

    df[col] = np.log1p(df[col])
# overview df shape

print("overview df shape: ")

print(df.shape)



# split back to train and test set

train_x = df.iloc[0:len(train),]

test_x = df.iloc[len(train): len(df),]



train_y = np.log1p(train['SalePrice'])

#train_y = train['SalePrice']



#train_x_1 = train_x.loc[:,con_cols_1]

#train_x_2 = train_x.drop(con_cols_1, axis = 1)

# drop outlier

#train_x =  train_x.drop([523,1298,297,

#                        898,688,1181,473,1349,632,712,153,1078,120,0,80,88

#                        ], axis = 0)

#train_y = train_y.drop([523,1298,297,

#                       898,688,1181,473,1349,632,712,153,1078,120,0,80,88

#                       ], axis = 0)



# overview train_x and test_x shape

print(train_x.shape)

#print(train_x_2.shape)

print(test_x.shape)
plt.hist(train_y)

plt.xlabel("log1p of Sale Price")

plt.ylabel("frequency")

plt.title("Log transformation on Sale Price")
from sklearn.metrics import mean_absolute_error as mae

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV, KFold, train_test_split, StratifiedKFold



from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

import lightgbm as lgb
def rmsle(y_true, y_pred):

    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False
# train_x, train_y, test_x

kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)



params = {'num_leaves':[10,30,50],

         'n_estimators': [100,300,500],

          #'metric': ['l2', 'l1'],

          'learning_rate': [0.01,0.1],

          'min_child_samples': [20,40,60],

          'feature_fraction': [0.8,0.9],

          'bagging_fraction': [0.8,0.9],

          'bagging_freq': [1,3]

         }



fit_lgb = lgb.LGBMRegressor(random_state=123,verbose=0)



gbm = GridSearchCV(fit_lgb, params, cv=5,verbose=0)

gbm.fit(train_x, train_y)



print('Best parameters found by grid search are:', gbm.best_params_)

print('Best score: ',gbm.best_score_)
sample_submission['SalePrice'] = np.exp(gbm.predict(test_x))-1

sample_submission.to_csv('submission_lgb.csv',index=False)
best_xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.07,

                 max_depth=5,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42,

                                     random_state=123)

best_xgb_model.fit(train_x,train_y)
sample_submission['SalePrice'] = np.exp(best_xgb_model.predict(test_x))-1

sample_submission.to_csv('submission_xgb.csv',index=False)
from sklearn.ensemble import VotingRegressor

voting = VotingRegressor([('lightgbm',gbm.best_estimator_),('xgboost',best_xgb_model)])



voting.fit(train_x,train_y)
sample_submission['SalePrice'] = np.exp(voting.predict(test_x))-1

sample_submission.to_csv('submission_ensemble.csv',index=False)