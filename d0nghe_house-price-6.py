# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso,Ridge,ElasticNet

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.set_style("white")

sns.set_color_codes(palette='deep')

f, ax = plt.subplots(figsize=(8, 7))

#Check the new distribution 

sns.distplot(train['SalePrice'], color="b")

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")

sns.despine(trim=True, left=True)

plt.show()
train
test.shape
data = pd.concat([train,test])
corr = data.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (20,20))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.show()
data.info()
data.index = range(2919)

data.drop(['Id','Utilities'],axis=1,inplace=True)  



data['MoSold'] = data['MoSold'].astype(str)

data['YrSold'] = data['YrSold'].astype(str)
l2 = ['LotFrontage','MasVnrArea']

l3 = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea']



for item in l2 :

    data[item]= data[item].fillna(data[item].mean())

for item in l3 :

    data[item]= data[item].fillna(0)
data['FireplaceQu'] = data['FireplaceQu'].fillna('None')

data['MiscFeature'] = data['MiscFeature'].fillna('None')

data['Alley']=data['Alley'].fillna('None')
l1 = data['MSZoning'].unique().tolist()

for i in range(len(l1)-1):

    j = np.random.randint(0,4)

    data['MSZoning'] = data['MSZoning'].fillna(l1[j])  
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna('None')

data['Exterior2nd'] = data['Exterior2nd'].fillna('None')

data['MasVnrType'] = data['MasVnrType'].fillna('None') 



data['BsmtQual'] = data['BsmtQual'].fillna('None')      #  no Bsmt

data['BsmtCond'] = data['BsmtCond'].fillna('None')

data['BsmtExposure'] = data['BsmtExposure'].fillna('None')

data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')

data['Fence'] = data['Fence'].fillna('None')

data['Functional'] = data['Functional'].fillna('Typical')

data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')



data['GarageType'] = data['GarageType'].fillna('None')  #no Garage

data['GarageFinish'] = data['GarageFinish'].fillna('None')

data['GarageQual'] = data['GarageQual'].fillna('None')

data['GarageCond'] = data['GarageCond'].fillna('None')



data['PoolQC'] = data['PoolQC'].fillna('None')     # no Pool
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 'HouseStyle',

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1','Neighborhood', 'SaleCondition',

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope','Condition1','Condition2',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond','GarageType','SaleType','BldgType')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lb = LabelEncoder() 

    lb.fit(list(data[c].values)) 

    data[c] = lb.transform(list(data[c].values))
data = pd.get_dummies(data)
data.shape
data['Total_Bathrooms'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

data['GarageScale'] = data['GarageCars'] * data['GarageArea']                                   

data['GarageOrdinal'] = data['GarageQual'] + data['GarageCond']+data['GarageType']                 

data['GarageState'] = data['GarageFinish'] + data['GarageYrBlt'] + data['PavedDrive']              

data['Porchtotal'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']  

data['Extertotal'] = data['ExterQual'] + data['ExterCond']                                       

data['KitchenCombined'] = data['KitchenQual'] * data['KitchenAbvGr']                               

data['FireplaceCombined'] = data['FireplaceQu'] * data['Fireplaces']                               

data['BsmtOrdinal'] = data['BsmtQual'] + data['BsmtCond']                                          

data['BsmtFinishedAll'] = data['BsmtFinSF1'] + data['BsmtFinSF2']                                  

data['AllFlrSF'] = data['1stFlrSF'] + data['2ndFlrSF']                                             

data['OverallCombined'] = data['OverallQual'] + data['OverallCond']                                

data['TotalSF'] = data['AllFlrSF'] + data['TotalBsmtSF']                                           

data['YrBltAndRemod'] = data["YearRemodAdd"] + data['YearBuilt']                                   

data['roomtotal'] = data['BedroomAbvGr'] + data['KitchenAbvGr']+ data['TotRmsAbvGrd']              

data['SaleState'] = data['SaleType'] + data['SaleCondition']                                

data['HouseComment'] = data['BldgType']  + data['HouseStyle'] + data['OverallCond']                

data['Condition'] = data['Condition1'] + data['Condition2'] +  data['Neighborhood']               
#Log features：Gaussian

def logs(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   

        res.columns.values[m] = l + '_log'

        m += 1

    return res



log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','Total_Bathrooms'

               ,'GarageScale','GarageOrdinal','Porchtotal','Extertotal','KitchenCombined','FireplaceCombined','BsmtOrdinal',

                'BsmtFinishedAll','AllFlrSF','OverallCombined','TotalSF','YrBltAndRemod','roomtotal','SaleState','HouseComment','Condition']

data = logs(data, log_features)
#square_features

def square(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(np.square(res[l])).values)   

        res.columns.values[m] = l + '_square'

        m += 1

    return res



square_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','Total_Bathrooms'

               ,'GarageScale','GarageOrdinal','Porchtotal','Extertotal','KitchenCombined','FireplaceCombined','BsmtOrdinal',

                'BsmtFinishedAll','AllFlrSF','OverallCombined','TotalSF','YrBltAndRemod','roomtotal','SaleState','HouseComment','Condition']

data = square(data, square_features)
# # value**1.5

# def s_r(res, ls):

#     m = res.shape[1]

#     for l in ls:

#         res = res.assign(newcol=pd.Series(np.power(res[l],1.5)).values)   

#         res.columns.values[m] = l + '_c'

#         m += 1

#     return res



# square_root_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

#                  'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

#                  'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

#                  'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

#                  'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','Total_Bathrooms'

#                ,'GarageScale','GarageOrdinal','Porchtotal','Extertotal','KitchenCombined','FireplaceCombined',

#                'BsmtOrdinal','BsmtFinishedAll','AllFlrSF','OverallCombined','TotalSF','YrBltAndRemod']

# data = s_r(data, square_root_features)
data.shape
train_data = data[:1460]

test_data = data[1460:].drop(['SalePrice'],axis = 1)
y = np.log1p(train_data['SalePrice'])    #smoothing

x = train_data.drop(['SalePrice'],axis=1)
xtrain ,xtest ,ytrain ,ytest = train_test_split(x,y,test_size = 0.3, random_state = 4200 )
la = Lasso(alpha=0.1,max_iter=500)

la.fit(x,y)

pred1 = np.expm1(la.predict(test_data))
R = Ridge(alpha=1.0,max_iter=500)

R.fit(x,y)

pred2 = np.expm1(R.predict(test_data))
xgbr = XGBRegressor(  booster='gbtree'

                    , colsample_bylevel=1

                    , colsample_bynode=1

                    , colsample_bytree=0.6

                    , gamma=0

                    , importance_type='gain'

                    , learning_rate=0.01

                    , max_delta_step=0

                    , max_depth= 3 

                    , min_child_weight=1.5

                    , n_estimators=5400

                    , n_jobs=-1

                    , nthread=None

                    , objective='reg:squarederror'

                    , reg_alpha=0.3

                    , reg_lambda=0.7

                    , scale_pos_weight=1

                    , silent=None

                    , subsample=0.6

                    , verbosity=1)

xgbr.fit(x,y)

pred3 = np.expm1(xgbr.predict(test_data))
lgmb = LGBMRegressor(objective='regression',

                    boosting_type='gbdt',

                    num_leaves= 5,

                    max_depth = 3,

                    learning_rate=0.01,

                    n_estimators=6300,

                    subsample_for_bin=100,

                    class_weight=None,

                    min_split_gain=0.0,

                    min_child_weight=1.3,

                    min_child_samples=3,

                    subsample=0.1,

                    subsample_freq=0,

                    colsample_bytree=0.6,

                    reg_alpha=0.1,

                    reg_lambda=0.6,

                    random_state=None,

                    n_jobs= -1,

                    silent=True,

                    importance_type='gain')

lgmb.fit(x,y)

pred4 = np.expm1(lgmb.predict(test_data))
gbrt = GradientBoostingRegressor(alpha=0.5, ccp_alpha=0.0,

                                 criterion='friedman_mse',

                                 init=None, learning_rate=0.01,

                                 loss='huber',  max_depth=3,

                                 max_leaf_nodes=5,

                                 min_impurity_decrease=0.0,

                                 min_impurity_split=None,

                                 min_samples_leaf=10,

                                 min_samples_split=3,

                                 min_weight_fraction_leaf=0.0,

                                 n_estimators=3350,

                                 n_iter_no_change=None,

                                 random_state=5,subsample=0.6,

                                 tol=0.0001,

                                 validation_fraction=0.1,

                                 verbose=0)

gbrt.fit(x,y)

pred5 = np.expm1(gbrt.predict(test_data))
E = ElasticNet(alpha=0.1,l1_ratio=0.1)

E.fit(x,y)

pred6 = np.expm1(E.predict(test_data))
#0.12056

#pred =  0.01* pred1 + 0.25* pred2  + 0.35*pred3 + 0.35*pred4 +0.01* pred5+0.03* pred6    



#0.11902

pred =  0.0* pred1 + 0.2* pred2  + 0.37*pred3 + 0.38*pred4 +0.00* pred5+0.05* pred6    
pred = pd.DataFrame({"id": test['Id'], "SalePrice": pred})

pred.to_csv('sample_submission.csv',index=False)