import numpy as np 

import pandas as pd

import seaborn as sns

from sklearn import *



import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(10)

#test.head()
sns.distplot(train.SalePrice)
train.YrSold.unique()
sns.heatmap(train.corr(),vmax=.8, linewidths=0.01,

            square=True,cmap='magma',linecolor="black");
#Correlations

train.corr()['SalePrice'].sort_values(ascending=False)
# ilişkili özelliklerin listesi görselleştireceğiz ve aykırı noktaları temizleyeceğiz

features = ['OverallQual','YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF',

                    'GrLivArea','FullBath','TotRmsAbvGrd','GarageCars','GarageArea']



sns.heatmap(train[features].corr(),annot = True,linewidths = 0.5);

##Plot Reg GrLivArea

sns.regplot('GrLivArea','SalePrice',data=train,color = 'blue');
#4000'e kadar GrLivArea alacagiz. 4000 ten buyuk olanlari setinin modu üzerinde çok az etkisi var

train = train.drop(train[(train['GrLivArea']>4000)].index)

sns.regplot('GrLivArea','SalePrice',data=train,color = 'red');
sns.regplot('GarageArea','SalePrice',data=train,color='red');
#The error seems to have constant variance till GarageArea=1000 but we after that it's dipersed and it can create huge problem in analysis. So we'll remove the outliers from here.

train = train[train['GarageArea']<1200]

sns.regplot('GarageArea','SalePrice',data=train,color='red');
#'1stFlrSf' ve '2ndFlrSF' Birleştirdim SalePrice ile her ikisinden de tek başına daha iyi bir ilişkisi var.

sns.regplot(train['1stFlrSF'] + train['2ndFlrSF'],train['SalePrice'],color='red');
#Bu plot bir evin Genel Kalitesinin artması nedeniyle medyan Satış Fiyatının arttığını göstermektedir.

sns.regplot(train['OverallQual'], train['SalePrice']);
train['OverallQual'].value_counts().plot();
##Data Preprocessing and Cleaning

train['log_SalePrice']=np.log(train['SalePrice']+1)

saleprices=train[['SalePrice','log_SalePrice']]



saleprices.head(5)
train=train.drop(columns=['SalePrice','log_SalePrice'])
print(test.shape)

print(train.shape)
all_data = pd.concat((train, test))

print(all_data.shape)

all_data.head()
#Checking  NaN values in Data

#Find and count missing values

null_data = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))

null_data

#null_data.plot.bar()
numeric=train.select_dtypes(include=[np.number])

categoric=train.select_dtypes(include=[np.object])

print('Numeric :\n',num_feat.dtypes,'\n')

print('Categoric :\n',cat_feat.dtypes)
#Change the mıssıng values with 'None'

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    

    all_data[col] = all_data[col].fillna('None')

for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

#Combining similar features to make new features

#All the above three feature define area of the house and we can easily combine these to form TotalSF - Total Area in square feet

all_data['TotalSF']=all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data['No2ndFlr']=(all_data['2ndFlrSF']==0)

all_data['NoBsmt']=(all_data['TotalBsmtSF']==0)

sns.regplot(train['TotalBsmtSF']+train['1stFlrSF']+train['2ndFlrSF'],saleprices['SalePrice'],color='blue');
#The BsmtFullBath ,FullBath, BsmtHalfBath can be combined for a TotalBath similar to Total

all_data['TotalBath']=all_data['BsmtFullBath'] + all_data['FullBath'] + all_data['BsmtHalfBath'] + all_data['HalfBath']
#Combining YearBuilt and YearRemodAdd

all_data['YrBltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']
sns.heatmap(all_data.corr(),linewidths = 0.1);
#These features are not much related to the SalePrice so we'll drop them.

all_data=all_data.drop(columns=['Street','Utilities','Condition2','RoofMatl',

                                'Heating','PoolArea','PoolQC','MiscVal','MiscFeature'])
# treat some numeric values as str which are infact a categorical variables

all_data['MSSubClass']=all_data['MSSubClass'].astype(str)

all_data['MoSold']=all_data['MoSold'].astype(str)

all_data['YrSold']=all_data['YrSold'].astype(str)
#These features might look better without 0 data

all_data['NoLowQual']=(all_data['LowQualFinSF']==0)

all_data['NoOpenPorch']=(all_data['OpenPorchSF']==0)

all_data['NoWoodDeck']=(all_data['WoodDeckSF']==0)

all_data['NoGarage']=(all_data['GarageArea']==0)
Basement = ['BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFinType1',

            'BsmtFinType2', 'BsmtQual', 'BsmtUnfSF','TotalBsmtSF']

Bsmt=all_data[Basement]

Bsmt.head()
Bsmt=Bsmt.replace(to_replace='Po', value=1)

Bsmt=Bsmt.replace(to_replace='Fa', value=2)

Bsmt=Bsmt.replace(to_replace='TA', value=3)

Bsmt=Bsmt.replace(to_replace='Gd', value=4)

Bsmt=Bsmt.replace(to_replace='Ex', value=5)

Bsmt=Bsmt.replace(to_replace='None', value=0)



Bsmt=Bsmt.replace(to_replace='No', value=1)

Bsmt=Bsmt.replace(to_replace='Mn', value=2)

Bsmt=Bsmt.replace(to_replace='Av', value=3)

Bsmt=Bsmt.replace(to_replace='Gd', value=4)



Bsmt=Bsmt.replace(to_replace='Unf', value=1)

Bsmt=Bsmt.replace(to_replace='LwQ', value=2)

Bsmt=Bsmt.replace(to_replace='Rec', value=3)

Bsmt=Bsmt.replace(to_replace='BLQ', value=4)

Bsmt=Bsmt.replace(to_replace='ALQ', value=5)

Bsmt=Bsmt.replace(to_replace='GLQ', value=6)

Bsmt.head()
Bsmt['BsmtScore']= Bsmt['BsmtQual']  * Bsmt['BsmtCond'] * Bsmt['TotalBsmtSF']

all_data['BsmtScore']=Bsmt['BsmtScore']

Bsmt['BsmtFin'] = (Bsmt['BsmtFinSF1'] * Bsmt['BsmtFinType1']) + (Bsmt['BsmtFinSF2'] * Bsmt['BsmtFinType2'])

all_data['BsmtFinScore']=Bsmt['BsmtFin']

all_data['BsmtDNF']=(all_data['BsmtFinScore']==0)
lot=['LotFrontage', 'LotArea','LotConfig','LotShape']

Lot=all_data[lot]

Lot.head()
garage=['GarageArea','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType','GarageYrBlt']

Garage=all_data[garage]



Garage=Garage.replace(to_replace='Po', value=1)

Garage=Garage.replace(to_replace='Fa', value=2)

Garage=Garage.replace(to_replace='TA', value=3)

Garage=Garage.replace(to_replace='Gd', value=4)

Garage=Garage.replace(to_replace='Ex', value=5)

Garage=Garage.replace(to_replace='None', value=0)



Garage=Garage.replace(to_replace='Unf', value=1)

Garage=Garage.replace(to_replace='RFn', value=2)

Garage=Garage.replace(to_replace='Fin', value=3)



Garage=Garage.replace(to_replace='CarPort', value=1)

Garage=Garage.replace(to_replace='Basment', value=4)

Garage=Garage.replace(to_replace='Detchd', value=2)

Garage=Garage.replace(to_replace='2Types', value=3)

Garage=Garage.replace(to_replace='Basement', value=5)

Garage=Garage.replace(to_replace='Attchd', value=6)

Garage=Garage.replace(to_replace='BuiltIn', value=7)

Garage.head()
non_numeric=all_data.select_dtypes(exclude=[np.number, bool])

non_numeric.head()
def onehot(col_list):

    global all_data

    while len(col_list) !=0:

        col=col_list.pop(0)

        data_encoded=pd.get_dummies(all_data[col], prefix=col)

        all_data=pd.merge(all_data, data_encoded, on='Id')

        all_data=all_data.drop(columns=col)

    print(all_data.shape)
def log_transform(col_list):

    transformed_col=[]

    while len(col_list)!=0:

        col=col_list.pop(0)

        if all_data[col].skew() > 0.5:

            all_data[col]=np.log(all_data[col]+1)

            transformed_col.append(col)

        else:

            pass

    

print(all_data.shape)
numeric=all_data.select_dtypes(include=np.number)

log_transform(list(numeric))

print(train.shape)

print(test.shape)
#Extracting Train and Test Data again

train=all_data[:len(train)]

test=all_data[len(train):]

print(train.shape)

print(test.shape)
#Modelleme
feature_names=list(all_data)

X_train = train[feature_names]

X_test = test[feature_names]

y_train = saleprices['log_SalePrice']
from sklearn.ensemble import RandomForestRegressor
RandomForestReg = RandomForestRegressor(bootstrap=False, criterion='mse', max_depth=None,

           max_features=60, max_leaf_nodes=None, min_impurity_decrease=0.0,

           min_impurity_split=None, min_samples_leaf=1,

           min_samples_split=2, min_weight_fraction_leaf=0.0,

           n_estimators=70, n_jobs=1, oob_score=False, random_state=42,

           verbose=0, warm_start=False)
RandomForestReg.fit(X_train, y_train)

regr_Predictions=np.exp(RandomForestReg.predict(X_test))-1
submission=pd.read_csv("C:\\data\\sample_submission.csv")

submission['SalePrice']= ensemble

submission.to_csv('submission.csv',index=True)
