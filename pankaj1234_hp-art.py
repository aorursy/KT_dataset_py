# Submissions for https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# House Prices: Advanced Regression Techniques



#Standard Toolbox

import sys

import numpy as np 

import pandas as pd

import os

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Evaluation 

from sklearn import preprocessing

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from sklearn.neighbors import LocalOutlierFactor

from sklearn.model_selection import train_test_split



# Stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from scipy import stats



from sklearn.linear_model import RidgeCV

from sklearn.svm import SVC





# Preset data display

pd.options.display.max_seq_items = 5000

pd.options.display.max_rows = 5000

pd.set_option('display.max_columns', 500)



import warnings

warnings.filterwarnings(action="ignore")
def displot(c):

    fig, ax = plt.subplots(figsize=(18,5), ncols=2, nrows=1)

    ax1 = sns.distplot(c, fit=norm, kde=False, ax=ax[0])

    ax2 = stats.probplot(c, plot=ax[1])

    mu, sigma = norm.fit(c)

    ax1.legend(['Dist. ($\mu=$ {:.0f}, $\sigma=$ {:.0f} , skw= {:.3f} , kurt= {:.3f} )'.format(mu, sigma, c.skew(), c.kurt())],

            loc='best')
def OHE(df, cols):

    return pd.get_dummies(df, columns=cols)



def Encoding_BE(df, col, th):

    #c = col+'_Code'

    c = col

    vc = df[col].value_counts().to_frame()

    include = vc[vc[col] > th]

    df[c] = df[col]

    for v in df[c].unique():

        i = np.where(df[c]==v)[0]

        if v in include.index:

            m= len(list(include.index)) - list(include.index).index(v)

            df.loc[i,c]= m

        else:

            df.loc[i,c]= 0

    i = np.where(df[c].isnull())[0]

    df.loc[i,c]= 0

    return df



def Encoding_BE_Rank(df, col, rank):

    #c = col+'_Code'

    c = col

    df[c] = df[col]

    for v in df[c].unique():

        i = np.where(df[c]==v)[0]

        if v in rank:

            m = rank.index(v)

            df.loc[i,c]= m +1

    return df

def Normalize01(df, col):

    #df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())

    #df[col] = df[col]

    rscaler = MinMaxScaler()

    df_col = rscaler.fit_transform(np.array(df[col]).reshape(-1,1))

    df[col] = df_col

    return df





def Impute_Missing(c,t,m=False,r=True):

    total_rec = c.shape[0]

    null_count = c.isna().sum()

    p_null = null_count/total_rec * 100

    t = pd.get_dummies(t)

    m_index = c[(c.isna() | c.isnull())].index

    Y = c.drop(m_index)

    X = t.drop(m_index)

    X_m = np.array(t.iloc[m_index])

    if r:

        if m:

            m_regressor = RidgeCV(alphas=[1e-3, 1e-2],cv=5).fit(X, Y)

            c[m_index] = m_regressor.predict(X_m).astype(str(c.dtypes)).reshape(len(m_index),)

    else:

        if m:

            m_classifier = SVC(kernel='rbf',gamma=1, C=0.5).fit(X, Y)

            c[m_index] = m_classifier.predict(X_m).reshape(len(m_index),)

    return c



def outliers(c, t, top=5):

    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.05)

    x_=np.array(c).reshape(-1,1)

    preds = lof.fit_predict(x_)

    lof_scr = lof.negative_outlier_factor_

    out_idx = pd.Series(lof_scr).sort_values()[:top].index

    return out_idx



def reduceSkew(c):

    #displot(c)

    #c_trn = c[0:trn_i]

    #c_tst = c[trn_i:]

    #sk_c_trn = boxcox1p(c_trn, boxcox_normmax(c_trn+1))

    #sk_c_tst = boxcox1p(c_tst, boxcox_normmax(c_tst+1))

    #nc = pd.concat([sk_c_trn, sk_c_tst], ignore_index=True)

    if skew(c)>0.75:

        print('Column: ' + c.name +' skew:' +  str(skew(c)))

        return boxcox1p(c, boxcox_normmax(c+1))

    else:

        return c
testdata = pd.read_csv(os.path.join("../input","test.csv"))

traindata = pd.read_csv(os.path.join("../input","train.csv"))



trn_i = traindata.shape[0]

train_target = traindata.SalePrice

log_trn_target = np.log(train_target)

trainID = traindata.Id

testID = testdata.Id

testdata['SalePrice'] = 0

traindata = traindata.drop(['Id','SalePrice'], axis=1)

testdata = testdata.drop(['Id','SalePrice'], axis=1)







outs = outliers(traindata['LotArea'], train_target, top=40)

traindata = traindata.drop(outs)

train_target = train_target.drop(outs)

log_trn_target =log_trn_target.drop(outs)

trn_i = traindata.shape[0]

dfull = pd.concat([traindata,testdata], ignore_index=True) # merge for dataframe for EDA

print(traindata.shape, testdata.shape, dfull.shape)
plt.scatter(dfull['LotArea'],dfull['GrLivArea'])
plt.boxplot(dfull["LotArea"])
displot(dfull['LotArea'])
dfull['LotArea'].sort_values()
columns_to_drop=[]

OHE_Cols=[]



#Process Columns

# 1) ID

#columns_to_drop.append('Id')

# 2) MSSubClass - Convert to Category; append to OHE list

dfull['MSSubClass']= dfull['MSSubClass'].astype('category') 

OHE_Cols.append('MSSubClass')

#dfull['MSSubClass'] = reduceSkew(dfull['MSSubClass'])

# 3) MSZoning -  Append to OHE List

OHE_Cols.append('MSZoning')

# 4) LotFrontage + LotArea

#dfull['LotFrontage'].fillna(dfull['LotFrontage'].mean(), inplace=True)

dfull['LotFrontage'] = Impute_Missing(c=dfull['LotFrontage'],t=dfull.loc[:,['LotArea','Neighborhood', 'GrLivArea']], r=True, m=True)

#dfull['LotFrontRatio'] = dfull['LotFrontage']/dfull['LotArea']

#dfull['LotFrontRatio'] = reduceSkew(dfull['LotFrontRatio'])

dfull= Normalize01(dfull, 'LotFrontage')

dfull= Normalize01(dfull, 'LotArea')

dfull['LotArea'] = reduceSkew(dfull['LotArea'])

dfull['LotFrontage'] = reduceSkew(dfull['LotFrontage'])

#columns_to_drop.append('LotFrontage')

#columns_to_drop.append('LotArea') # double check

# 6) Street

OHE_Cols.append('Street')

# 7) Alley

    #dfull['Alley'].fillna('None', inplace=True)

dfull['Alley'] = dfull['Alley'].apply(lambda x: 0 if pd.isnull(x) else 1)

columns_to_drop.append('Alley')

# 8) LotShape

OHE_Cols.append('LotShape')

# 9) LandContour

OHE_Cols.append('LandContour')

# 10) Utilities

columns_to_drop.append('Utilities')

# 11) LotConfig

OHE_Cols.append('LotConfig')

# 12) LandSlope

OHE_Cols.append('LandSlope')

# 13) 'Neighborhood'

OHE_Cols.append('Neighborhood')

# 14) Condition1

#dfull = Encoding_BE(dfull,'Condition1', 20)

#dfull = Normalize01(dfull, 'Condition1')

OHE_Cols.append('Condition1')

# 14) Condition2

    #New Column to track houses with two different prime conditions for location

dfull['If2Conditions'] = ((dfull['Condition1'] != 'Norm') & (dfull['Condition2'] != 'Norm') & (dfull['Condition2'] != dfull['Condition2'])).astype('int16')

columns_to_drop.append('Condition2')

# 15) BldgType

OHE_Cols.append('BldgType')

# 16) HouseStyle

OHE_Cols.append('HouseStyle')

# 17) & 18) OverallQual * OverallCond

dfull['OverallQuCo']= (dfull['OverallQual'] * dfull['OverallCond'])

dfull = Normalize01(dfull, 'OverallQuCo')

dfull['OverallQuCo'] = reduceSkew(dfull['OverallQuCo'])

columns_to_drop.append('OverallQual')

columns_to_drop.append('OverallCond')

# 19 - 20) YearBuilt & YearRemodAdd- > deducted New column 'Rennovate'

dfull['Rennovate'] = ((dfull['YearRemodAdd'].astype('int16') - dfull['YearBuilt'].astype('int16'))>1).astype('int16')

dfull= Normalize01(dfull, 'Rennovate')

dfull['Rennovate'] = reduceSkew(dfull['Rennovate'])

dfull['YearOld'] = dfull['YearBuilt'].max() - dfull['YearBuilt'].astype('int16')

dfull= Normalize01(dfull, 'YearOld')

dfull['YearOld'] = reduceSkew(dfull['YearOld'])

columns_to_drop.append('YearBuilt')

columns_to_drop.append('YearRemodAdd')

# 21) RoofStyle

OHE_Cols.append('RoofStyle')

# 22) RoofMatl

OHE_Cols.append('RoofMatl')

# 23) & Exterior1st

OHE_Cols.append('Exterior1st')

# 24) Exterior2nd

dfull['Exterior'] = ((dfull['Exterior1st'] != dfull['Exterior2nd'])).astype('int32')

columns_to_drop.append('Exterior2nd')

# 25) MasVnrType

dfull['MasVnrType'].loc[dfull['MasVnrType'].isna()==True] = 'None'

#dfull['MasVnrType']= Impute_Missing(c = dfull['MasVnrType'], t = dfull.loc[:,['Exterior1st','Exterior2nd']], m = True, r = False)

OHE_Cols.append('MasVnrType')

# 26) MasVnrArea

dfull['MasVnrArea'].loc[dfull['MasVnrType'].isna()==True] = 0

#dfull['MasVnrArea'].fillna((dfull['MasVnrArea'].loc[dfull['MasVnrArea']>0]).mean(), inplace=True)

dfull= Normalize01(dfull, 'MasVnrArea')

dfull['MasVnrArea'] = reduceSkew(dfull['MasVnrArea'])

# 27) & 28) ExterQaCo =  ExterQual * ExterCond

dfull = Encoding_BE_Rank(dfull,'ExterQual', ['None','Po','Fa','TA','Gd','Ex'])

dfull = Encoding_BE_Rank(dfull,'ExterCond', ['None','Po','Fa','TA','Gd','Ex'])

dfull['ExterQaCo'] = (dfull['ExterQual'] * dfull['ExterCond'])

dfull= Normalize01(dfull, 'ExterQaCo')

dfull['ExterQaCo'] = reduceSkew(dfull['ExterQaCo'])

columns_to_drop.append('ExterQual')

columns_to_drop.append('ExterCond')

# 29) Foundation 

OHE_Cols.append('Foundation')

# 30) BsmtQual 30) BsmtQual  32) BsmtExposure 

dfull['BsmtQual'].fillna('None', inplace=True)

dfull['BsmtCond'].fillna('None', inplace=True)

dfull['BsmtExposure'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'BsmtExposure', ['None','No','Mn','Av','Gd'])

dfull['BsmtExposure'] = reduceSkew(dfull['BsmtExposure'])

dfull= Normalize01(dfull, 'BsmtExposure')

dfull = Encoding_BE_Rank(dfull,'BsmtCond', ['None','Po','Fa','TA','Gd','Ex'])

dfull = Encoding_BE_Rank(dfull,'BsmtQual', ['None','Po','Fa','TA','Gd','Ex'])

dfull['BsmtQC'] = (dfull['BsmtQual'] * dfull['BsmtCond'])**(0.5)

dfull['BsmtQC'] = reduceSkew(dfull['BsmtQC'])

dfull= Normalize01(dfull, 'BsmtQC')

columns_to_drop.append('BsmtQual')

columns_to_drop.append('BsmtCond')

# 33) & 35) BsmtFinType1 & BsmtFinType2

dfull['BsmtFinType1'].fillna('None', inplace=True)

dfull['BsmtFinType2'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'BsmtFinType2', ['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'])

dfull = Encoding_BE_Rank(dfull,'BsmtFinType1', ['None','Unf','LwQ','Rec','BLQ','ALQ','GLQ'])

#dfull['BsmtFin'] = dfull['BsmtFinType1'] + dfull['BsmtFinType2']

#dfull['BsmtFin'] = reduceSkew(dfull['BsmtFin'])

#dfull= Normalize01(dfull, 'BsmtFin')

columns_to_drop.append('BsmtFinType1')

columns_to_drop.append('BsmtFinType2')

# 34) BsmtFinSF1

columns_to_drop.append('BsmtFinSF1')

# 36) BsmtFinSF2

columns_to_drop.append('BsmtFinSF2')

# 37) BsmtUnfSF

dfull['BsmtUnfSF'].fillna(0.0, inplace=True) # this is for house at index 2120

columns_to_drop.append('BsmtUnfSF')

# 38) TotalBsmtSF

dfull['TotalBsmtSF'].fillna(0, inplace=True) # this is for house at index 2120

dfull['BsmtFinRatio'] = dfull.loc[:,['TotalBsmtSF','BsmtUnfSF']].apply(lambda x: x['BsmtUnfSF']/x['TotalBsmtSF'] if x['TotalBsmtSF']!=0 else 0, axis=1) # new column

dfull['BsmtFinRatio'] = reduceSkew(dfull['BsmtFinRatio'])

dfull= Normalize01(dfull, 'BsmtFinRatio')

columns_to_drop.append('TotalBsmtSF')

# 39) Heating

OHE_Cols.append('Heating')

# 40) HeatingQC

dfull = Encoding_BE_Rank(dfull,'HeatingQC', ['None','Po','Fa','TA','Gd','Ex'])

dfull['HeatingQC'] = reduceSkew(dfull['HeatingQC'])

dfull= Normalize01(dfull, 'HeatingQC')

# 41) CentralAir

dfull['CentralAir'] = dfull['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)

dfull['CentralAir'] = reduceSkew(dfull['CentralAir'])

dfull= Normalize01(dfull, 'CentralAir')

# 42)Electrical

dfull['Electrical'].fillna('SBrkr', inplace=True) #applying Majority class for #1379

dfull = Encoding_BE_Rank(dfull,'Electrical', ['None','Mix','FuseP','FuseF','FuseA','SBrkr'])

dfull['Electrical'] = reduceSkew(dfull['Electrical'])

dfull= Normalize01(dfull, 'Electrical')

# 43) 1stFlrSF

columns_to_drop.append('1stFlrSF')

#dfull= Normalize01(dfull, '1stFlrSF')

# 44) 2ndFlrSF

dfull['2ndFlrSF']= dfull['2ndFlrSF'].apply(lambda x: 1 if x > 10 else 0) # check if 2nd floor exist

# 45) LowQualFinSF

dfull['LowQualFinSF']= dfull['LowQualFinSF'].apply(lambda x: 1 if x > 10 else 0) # check if LowQualFinSF exist for the house

# 46) GrLivArea

dfull['GrLivArea'] = reduceSkew(dfull['GrLivArea'])

dfull= Normalize01(dfull, 'GrLivArea')

# 47) BsmtFullBath 48) BsmtHalfBath

dfull['BsmtFullBath'].fillna(0.0, inplace=True)

dfull['BsmtHalfBath'].fillna(0.0, inplace=True)

dfull['BsmtTotBath'] = dfull['BsmtFullBath'] + 0.5*dfull['BsmtHalfBath']

dfull['BsmtTotBath'] = reduceSkew(dfull['BsmtTotBath'])

dfull= Normalize01(dfull, 'BsmtTotBath')

columns_to_drop.append('BsmtFullBath')

columns_to_drop.append('BsmtHalfBath')

# 49) FullBath 50) HalfBath

dfull['FullBath'].fillna(0.0, inplace=True)

dfull['HalfBath'].fillna(0.0, inplace=True)

dfull['TotBath'] = dfull['FullBath'] + 0.5* dfull['HalfBath']

dfull['TotBath'] = reduceSkew(dfull['TotBath'])

dfull= Normalize01(dfull, 'TotBath')

columns_to_drop.append('FullBath')

columns_to_drop.append('HalfBath')

# 51) BedroomAbvGr

dfull['BedroomAbvGr'] = reduceSkew(dfull['BedroomAbvGr'])

dfull= Normalize01(dfull, 'BedroomAbvGr')

# 52) KitchenAbvGr

dfull['KitchenAbvGr'] = reduceSkew(dfull['KitchenAbvGr'])

dfull= Normalize01(dfull, 'KitchenAbvGr')

# 53) KitchenQual

dfull['KitchenQual'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'KitchenQual', ['None','Po','Fa','TA','Gd','Ex'])

dfull= Normalize01(dfull, 'KitchenQual')

# 54) TotRmsAbvGrd

dfull['TotRmsAbvGrd'] = reduceSkew(dfull['TotRmsAbvGrd'])

dfull= Normalize01(dfull, 'TotRmsAbvGrd')

columns_to_drop.append('TotRmsAbvGrd')

# 55) Functional

#dfull['Functional']=dfull['Functional'].apply(lambda x: 1 if x == 'Typ' else 0)

dfull['Functional'].fillna('Typ', inplace=True)

dfull = Encoding_BE_Rank(dfull,'Functional', ['None','Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal'])

dfull= Normalize01(dfull, 'Functional')

# 56) Fireplaces 57) FireplaceQu

dfull['FireplaceQu'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'FireplaceQu', ['None','Po','Fa','TA','Gd','Ex'])

dfull['FireplaceInfo'] = (dfull['FireplaceQu'] * dfull['Fireplaces'])

columns_to_drop.append('Fireplaces')

columns_to_drop.append('FireplaceQu')

# 58) GarageType

dfull['GarageType'].fillna('None', inplace=True)

OHE_Cols.append('GarageType')

# 59) GarageYrBlt - deducted New column 'Rennovate'

dfull['GarageYrBlt'].fillna(0, inplace=True)

dfull['GarageSmYr'] = dfull['GarageYrBlt'].astype('int16')- dfull['YearBuilt'].astype('int16')

dfull['GarageSmYr'] = dfull['GarageSmYr'].apply(lambda x: 0 if x <= 0 else 1) ## Need to check again 

#OHE_Cols.append('GarageSmYr')

#dfull['GarageSmYr'] = reduceSkew(dfull['GarageSmYr'])

#dfull= Normalize01(dfull, 'GarageSmYr')

columns_to_drop.append('GarageYrBlt')

#columns_to_drop.append('GarageSmYr')

# 60) GarageFinish

dfull['GarageFinish'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'GarageFinish', ['None','Unf','RFn','Fin'])

dfull= Normalize01(dfull, 'GarageFinish')

# 61) GarageCars & 62) GarageArea

dfull['GarageCars'].fillna(0, inplace=True)

dfull['GarageArea'].fillna(0, inplace=True)

dfull['GarageAreaCar'] = dfull.loc[:,['GarageCars','GarageArea']].apply(lambda x: x['GarageArea']/x['GarageCars'] if x['GarageCars']!=0 else 0, axis=1)

dfull['GarageAreaCar'] = reduceSkew(dfull['GarageAreaCar'])

columns_to_drop.append('GarageCars')

columns_to_drop.append('GarageArea')

dfull= Normalize01(dfull, 'GarageAreaCar')

# 63) GarageQual 64) GarageCond

dfull['GarageQual'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'GarageQual', ['None','Po','Fa','TA','Gd','Ex'])

dfull['GarageCond'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'GarageCond', ['None','Po','Fa','TA','Gd','Ex'])

dfull['GarageQuCo']= (dfull['GarageQual'] * dfull['GarageCond'])

dfull['GarageQuCo'] = reduceSkew(dfull['GarageQuCo'])

columns_to_drop.append('GarageQual')

columns_to_drop.append('GarageCond')

dfull = Normalize01(dfull, 'GarageQuCo')

# 65) PavedDrive

dfull = Encoding_BE_Rank(dfull,'PavedDrive', ['N','P','Y'])

dfull['PavedDrive'] = reduceSkew(dfull['PavedDrive'])

dfull = Normalize01(dfull, 'PavedDrive')

# 66) WoodDeckSF

dfull['WoodDeckSF'] = dfull['WoodDeckSF'].apply(lambda x: 1 if x> 1 else 0)

dfull['WoodDeckSF'] = reduceSkew(dfull['WoodDeckSF'])

# 67) OpenPorchSF

dfull['OpenPorchSF'] = dfull['OpenPorchSF'].apply(lambda x: 1 if x> 1 else 0)

dfull['OpenPorchSF'] = reduceSkew(dfull['OpenPorchSF'])

# 68) EnclosedPorch

dfull['EnclosedPorch'] = dfull['EnclosedPorch'].apply(lambda x: 1 if x> 1 else 0)

#dfull['EnclosedPorch'] = reduceSkew(dfull['EnclosedPorch'])

# 69) 3SsnPorch

dfull['3SsnPorch'] = dfull['3SsnPorch'].apply(lambda x: 1 if x> 1 else 0)

#dfull['3SsnPorch'] = reduceSkew(dfull['3SsnPorch'])

# 70) ScreenPorch

dfull['ScreenPorch'] = dfull['ScreenPorch'].apply(lambda x: 1 if x>1 else 0)

#dfull['ScreenPorch'] = reduceSkew(dfull['ScreenPorch'])

# 71) PoolArea 72) PoolQC

dfull['PoolArea']=dfull['PoolArea'].apply(lambda x: 1 if x>0 else 0)

dfull['PoolQC'].fillna('None', inplace=True)

dfull = Encoding_BE_Rank(dfull,'PoolQC', ['None','Fa','TA','Gd','Ex'])

dfull['PoolInfo'] = (dfull['PoolArea'] * dfull['PoolQC'])

#dfull['PoolInfo'] = reduceSkew(dfull['PoolInfo'])

dfull = Normalize01(dfull, 'PoolInfo')

columns_to_drop.append('PoolQC')

columns_to_drop.append('PoolArea')

# 73) Fence

dict_fence = {0:"NA", 1:"MnWw", 2:"GdWo", 3:"MnPrv", 4:"GdPrv"}

dfull['Fence']=dfull['Fence'].apply(lambda x: 0 if pd.isnull(x) else 1)

# 74) MiscFeature

dfull['MiscFeature']=dfull['MiscFeature'].apply(lambda x: 0 if pd.isnull(x) else 1)

# 75) MiscVal

columns_to_drop.append('MiscVal')

# 76) MoSold

#dfull['MoSold']=dfull['MoSold'].apply(lambda x: 'Q1' if x in [1,2,3] else ('Q2' if x in [4,5,6] else ( 'Q3' if x in [7,8,9] else 'Q4')) )

dfull['MoSold']=dfull['MoSold'].apply(lambda x: 'H1' if x in [1,2,3,4,5,6] else 'H2')

OHE_Cols.append('MoSold')

# 77) YrSold

OHE_Cols.append('YrSold')

# 78) SaleType

#dfull['SaleType'] = dfull['SaleType'].apply(lambda x: 1 if x in ['WD','New','CWD'] else 0)

OHE_Cols.append('SaleType')

# 79) SaleCondition

OHE_Cols.append('SaleCondition')

# 80) SalePrice

#dfull['SalePrice'] = dfull['SalePrice'].astype('float')



#for One Hot Encoding (OHE)

dfull = OHE(dfull,OHE_Cols)

#Drop unused columns

dfull =dfull.drop(columns_to_drop, axis=1)
print("Columns to be dropped for training and predictions")

print(columns_to_drop)

print("Columns for One Hot Encoding")

print(OHE_Cols)
print(dfull.shape)

#print(dfull.head())
corrmat = dfull.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.75,vmin =.5)
# Select upper triangle of correlation matrix

upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]



dfull =dfull.drop(to_drop, axis=1)
dfull.head()
testdata = dfull.loc[trn_i:,:]

traindata = dfull.loc[0:(trn_i-1), :]

testdata = testdata.reset_index(drop=True)

traindata= traindata.reset_index(drop=True)

print(traindata.shape)

print(testdata.shape)
traindata_y = log_trn_target

traindata_x = traindata

#testdata_y = testdata.pop("SalePrice")

testdata_x = testdata

traindata_x.head(3)
#train_x, val_x, train_y,val_y = train_test_split(traindata_x,traindata_y, train_size=0.7,test_size=0.3, random_state = 123456)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

def get_predictions(model, params, x, y, xtest):

    gs = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=10, verbose=False)

    gs.fit(x,y)

    print(gs.best_params_)

    print(gs.best_score_)

    bestmodel = gs.best_estimator_

    bestmodel.fit(x,y)

    prediction = bestmodel.predict(xtest)

    return prediction



submissions = pd.read_csv("../input/sample_submission.csv")

rfm = RandomForestRegressor( random_state=123)

params = {'n_estimators':[1000]}

#rfm_pred = np.exp(get_predictions(rfm, params, traindata_x, traindata_y,testdata_x))

#submissions["SalePrice"] = np.exp(get_predictions(lgbmr, params, traindata_x, traindata_y,testdata_x))
from lightgbm import LGBMRegressor

lgbmr = LGBMRegressor(random_state=123)

params = {

    'num_leaves': [25],

    'max_depth': [7], 

    'bagging_fraction': [0.63],

    'feature_fraction': [0.55],

    'min_data_in_leaf': [23],

    'learning_rate': [0.01],

    'n_estimators':[1500]}



submissions["SalePrice"] = np.exp(get_predictions(lgbmr, params, traindata_x, traindata_y,testdata_x))

#submissions["SalePrice"] = (rfm_pred + pred_lgbm)/2

submissions.to_csv("submission.csv", index=False)