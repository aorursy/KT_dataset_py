from sklearn.preprocessing import LabelEncoder

from scipy.stats import norm, skew 

from scipy.special import boxcox1p

import matplotlib.pylab as plt

import seaborn as sns

import pandas as pd

pd.set_option('display.max_columns', 100) 

import numpy as np

import warnings

warnings.filterwarnings("ignore")

import os
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



train.shape,test.shape
num_feat = [col for col in train.columns if train[col].dtype != 'object']

print("Number of numeric feature in data :",len(num_feat))
def num_analysis(i=False, feat_name=False):

    

    if (i):

        feat = num_feat[i]

    

    elif (feat_name):

        feat = feat_name

        

    print("Feature: {}  & Correlation with target: {}".format(feat,train[feat].corr(train['SalePrice'])))

            

    fig=plt.figure(figsize=(20,4))



    ax=fig.add_subplot(1,3,1)

    ax.hist(train[feat])

    ax.set_title("train", fontsize = 20)

    ax.set_xlabel(feat,fontsize=20)

    ax.tick_params(labelsize=13)

    

    ax=fig.add_subplot(1,3,2)

    ax.scatter(train[feat],train['SalePrice'])

    ax.set_title(feat+" vs SalesPrice", fontsize = 20)

    ax.set_xlabel(feat,fontsize=20)

    ax.set_ylabel("SalesPrice",fontsize=20)

    ax.tick_params(labelsize=13)

    

    ax=fig.add_subplot(1,3,3)

    ax.hist(test[feat])

    ax.set_title("test", fontsize = 20)

    ax.set_xlabel(feat,fontsize=20)

    ax.tick_params(labelsize=13)

    

    plt.show()
k=1

num_analysis(i = k)
train['MSSubClass'] = train['MSSubClass'].apply(str)

test['MSSubClass'] = test['MSSubClass'].apply(str)



lbl = LabelEncoder() 

lbl.fit(list(train['MSSubClass'].values) + list(test['MSSubClass'].values)) 



train['MSSubClass'] = lbl.transform(list(train['MSSubClass'].values))

test['MSSubClass'] = lbl.transform(list(test['MSSubClass'].values))



num_analysis(feat_name='MSSubClass')
k= k +1

num_analysis(i = k)
dff = train.loc[(train['LotFrontage'] >= 300) & (train['SalePrice'] < 200000)]

dff
k = k+1

num_analysis(i = k)
from IPython.display import Image

Image("../input/skew-img/skew_kurt.png")
print("Skewness: %f" % train['LotArea'].skew())

print("Kurtosis: %f" % train['LotArea'].kurt())
train["LotArea"] = np.log1p(train["LotArea"])

test["LotArea"] = np.log1p(test["LotArea"])

num_analysis(i = k)
k = k + 1

num_analysis(i =k)
k = k+ 1

num_analysis(i=k)
train.loc[(train['OverallCond'] == 6) & (train['SalePrice'] > 600000)]
train['cond*qual'] = (train['OverallCond'] * train['OverallQual']) / 100.0

test['cond*qual'] = (test['OverallCond'] * test['OverallQual']) / 100.0



num_analysis(feat_name='cond*qual')
k = k+ 1

num_analysis(i=k)
train['home_age_when_sold'] = train['YrSold'] - train['YearBuilt']

test['home_age_when_sold'] = test['YrSold'] - test['YearBuilt']
num_analysis(feat_name='home_age_when_sold')
k = k +1 

num_analysis(i=k)
train['after_remodel_home_age_when_sold'] = train['YrSold'] - train['YearRemodAdd']

test['after_remodel_home_age_when_sold'] = test['YrSold'] - test['YearRemodAdd']
num_analysis(feat_name='after_remodel_home_age_when_sold')
k = k + 1

num_analysis(i=k)
print("Skewness: %f" % train['MasVnrArea'].skew())

print("Kurtosis: %f" % train['MasVnrArea'].kurt())
train["MasVnrArea"] = np.log1p(train["MasVnrArea"])

test["MasVnrArea"] = np.log1p(test["MasVnrArea"])

num_analysis(i = k)
k = k +1

num_analysis(i=k)
train.loc[(train['BsmtFinSF1'] > 4500) & (train['SalePrice'] < 200000)]
k = k +1

num_analysis(i=k)
print("Number of home which has BsmtFinSF2 in train : {}%" .format(((train['BsmtFinSF2']!=0).sum() / train.shape[0])*100))

print("Number of home which has BsmtFinSF2 in test  : {}%" .format(((test['BsmtFinSF2']!=0).sum() / test.shape[0])*100))
train['BsmtFinSF1+BsmtFinSF2'] = train['BsmtFinSF1'] + train['BsmtFinSF2']

test['BsmtFinSF1+BsmtFinSF2'] = test['BsmtFinSF1'] + test['BsmtFinSF2']



num_analysis(feat_name='BsmtFinSF1+BsmtFinSF2')
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
train.loc[(train['TotalBsmtSF']>=6000 ) &(train['SalePrice'] < 200000)]
k = k +1

num_analysis(i=k)
train.loc[(train['1stFlrSF'] >= 4000) & (train['SalePrice'] < 200000)]
k = k +1

num_analysis(i=k)
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']



num_analysis(feat_name='TotalSF')
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
train.loc[(train['GrLivArea'] >= 5000) & (train['SalePrice'] < 200000)]
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))

test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))



num_analysis(feat_name='Total_Bathrooms')
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
train.loc[(train['KitchenAbvGr'] == 3) & (train['SalePrice'] < 150000)]
k = k +1

num_analysis(i=k)
train.loc[(train['TotRmsAbvGrd'] == 14) & (train['SalePrice'] <= 200000)]
k = k +1

num_analysis(i=k)
train.loc[(train['Fireplaces'] == 3)]
print("Missing value count in Fireplaces  :",train['Fireplaces'].isna().sum())

print("Missing value count in FireplaceQu :",train['FireplaceQu'].isna().sum())
train["FireplaceQu"] = train["FireplaceQu"].fillna("NA")

test["FireplaceQu"] = test["FireplaceQu"].fillna("NA")
def FireplaceQu_encode(x):

    if x=='Ex':

        return 5

    elif x=='Gd':

        return 4

    elif x == 'TA':

        return 3

    elif x=='Fa':

        return 2

    elif x=='Po':

        return 1

    elif x=='NA':

        return 0
train['FireplaceQu'] = train['FireplaceQu'].apply(lambda x: FireplaceQu_encode(x))

test['FireplaceQu'] = test['FireplaceQu'].apply(lambda x: FireplaceQu_encode(x))
num_analysis(feat_name='FireplaceQu')
train['FirePlace*FireplaceQu'] = train['Fireplaces']*train['FireplaceQu']

test['FirePlace*FireplaceQu'] = test['Fireplaces']*test['FireplaceQu']
num_analysis(feat_name='FirePlace*FireplaceQu')
k = k +1

num_analysis(i=k)
test.loc[(test['GarageYrBlt'] >= 2100)]
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)
train['total_porch_area'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']

test['total_porch_area'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']



num_analysis(feat_name='total_porch_area')
k = k +1

num_analysis(i=k)
print("Number of house which has Pool in train :",(train['PoolArea'] != 0).sum())

print("Number of house which has Pool in test  :",(test['PoolArea'] != 0).sum())
k = k +1

num_analysis(i=k)
test.loc[test['MiscVal'] >= 15000]
k = k +1

num_analysis(i=k)
k = k +1

num_analysis(i=k)