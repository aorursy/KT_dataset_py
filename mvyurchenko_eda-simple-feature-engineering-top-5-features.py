import warnings

warnings.filterwarnings('ignore')



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor





from sklearn.svm import SVR



from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler



import lightgbm as lgb



from sklearn.decomposition import PCA

from sklearn.preprocessing import scale
data = pd.read_csv("../input/train.csv", index_col = "Id")
data.head()
# let's look at data briefly info

data['SalePrice'].describe()
# Now disturbtion of target variable

plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

plt.title("Disturbtion of target")

sns.distplot(data['SalePrice']);



# Now disturbtion of log(target) variable

plt.subplot(1, 2, 2)

plt.title("Disturbtion of log(target)")

sns.distplot(data['SalePrice'].apply(np.log));
#skewness and kurtosis

print("Skewness: %f" % data['SalePrice'].skew())

print("Kurtosis: %f" % data['SalePrice'].kurt())
# Create a pd.Series with log(SalePrice) 

log_target = data['SalePrice'].apply(np.log)
plt.figure(figsize=(20, 5))

columns_float  = list(data.loc[:, data.dtypes == np.float64].columns)



for num, var in enumerate(columns_float):

    plt.subplot(1, len(columns_float), num + 1)

    sns.regplot(x=data[var], y = log_target);
plt.figure(figsize=(40, 40))

columns_int = list(data.loc[:, data.dtypes == np.int64].columns)

columns_plot = [ ]



for num, var in enumerate(columns_int):

    if len(data[var].unique()) > 15:

        columns_plot.append(var)



for num, var in enumerate(columns_plot):

    plt.subplot(4, 5, num + 1)

    df = pd.concat([log_target, data[var]], axis=1)

    sns.regplot(x=data[var], y = log_target);
#box plot overallqual/saleprice

plt.figure(figsize=(20, 20))





cat_columns = ['YearBuilt', 'OverallQual']

for num, var in enumerate(cat_columns):

    plt.subplot(2, 1, num + 1)

    fig = sns.boxplot(x=var, y=log_target, data=data)

    plt.xticks(rotation=90);
# As suggested by many participants, we remove several outliers

data.drop(data[(data['OverallQual']<5) & (data['SalePrice']>200000)].index, inplace=True)

data.drop(data[(data['GrLivArea']>4000) & (data['SalePrice']<300000)].index, inplace=True)

data.reset_index(drop=True, inplace=True)



# Some of the non-numeric predictors are stored as numbers; we convert them into strings 

data['MSSubClass'] = data['MSSubClass'].apply(str)

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
# Here we create funtion which fills all the missing values

# Pay attention that some of the missing values of numeric predictors first are filled in with zeros and then 

# small values are filled in with median/average (and indicator variables are created to account for such change: 

# for each variable we create  which are equal to one);



def fill_missings(res):



    res['Alley'] = res['Alley'].fillna('missing')

    res['PoolQC'] = res['PoolQC'].fillna(res['PoolQC'].mode()[0])

    res['MasVnrType'] = res['MasVnrType'].fillna('None')

    res['BsmtQual'] = res['BsmtQual'].fillna(res['BsmtQual'].mode()[0])

    res['BsmtCond'] = res['BsmtCond'].fillna(res['BsmtCond'].mode()[0])

    res['FireplaceQu'] = res['FireplaceQu'].fillna(res['FireplaceQu'].mode()[0])

    res['GarageType'] = res['GarageType'].fillna('missing')

    res['GarageFinish'] = res['GarageFinish'].fillna(res['GarageFinish'].mode()[0])

    res['GarageQual'] = res['GarageQual'].fillna(res['GarageQual'].mode()[0])

    res['GarageCond'] = res['GarageCond'].fillna('missing')

    res['Fence'] = res['Fence'].fillna('missing')

    res['Street'] = res['Street'].fillna('missing')

    res['LotShape'] = res['LotShape'].fillna('missing')

    res['LandContour'] = res['LandContour'].fillna('missing')

    res['BsmtExposure'] = res['BsmtExposure'].fillna(res['BsmtExposure'].mode()[0])

    res['BsmtFinType1'] = res['BsmtFinType1'].fillna('missing')

    res['BsmtFinType2'] = res['BsmtFinType2'].fillna('missing')

    res['CentralAir'] = res['CentralAir'].fillna('missing')

    res['Electrical'] = res['Electrical'].fillna(res['Electrical'].mode()[0])

    res['MiscFeature'] = res['MiscFeature'].fillna('missing')

    res['MSZoning'] = res['MSZoning'].fillna(res['MSZoning'].mode()[0])    

    res['Utilities'] = res['Utilities'].fillna('missing')

    res['Exterior1st'] = res['Exterior1st'].fillna(res['Exterior1st'].mode()[0])

    res['Exterior2nd'] = res['Exterior2nd'].fillna(res['Exterior2nd'].mode()[0])    

    res['KitchenQual'] = res['KitchenQual'].fillna(res['KitchenQual'].mode()[0])

    res["Functional"] = res["Functional"].fillna("Typ")

    res['SaleType'] = res['SaleType'].fillna(res['SaleType'].mode()[0])

    res['SaleCondition'] = res['SaleCondition'].fillna('missing')

    

    flist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                     'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                     'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                     'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                     'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

    for fl in flist:

        res[fl] = res[fl].fillna(0)

        

    res['TotalBsmtSF'] = res['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

    res['2ndFlrSF'] = res['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    res['GarageArea'] = res['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)

    res['GarageCars'] = res['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)

    res['LotFrontage'] = res['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)

    res['MasVnrArea'] = res['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)

    res['BsmtFinSF1'] = res['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

    

      

    return res
# Running function to fill in missings

data = fill_missings(data)

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']



# Working with ordinal predictors

def QualToInt(x):

    if(x=='Ex'):

        r = 0

    elif(x=='Gd'):

        r = 1

    elif(x=='TA'):

        r = 2

    elif(x=='Fa'):

        r = 3

    elif(x=='missing'):

        r = 4

    else:

        r = 5

    return r



data['ExterQual'] = data['ExterQual'].apply(QualToInt)

data['ExterCond'] = data['ExterCond'].apply(QualToInt)

data['KitchenQual'] = data['KitchenQual'].apply(QualToInt)

data['HeatingQC'] = data['HeatingQC'].apply(QualToInt)

data['BsmtQual'] = data['BsmtQual'].apply(QualToInt)

data['BsmtCond'] = data['BsmtCond'].apply(QualToInt)

data['FireplaceQu'] = data['FireplaceQu'].apply(QualToInt)

data['GarageQual'] = data['GarageQual'].apply(QualToInt)

data['PoolQC'] = data['PoolQC'].apply(QualToInt)



def SlopeToInt(x):

    if(x=='Gtl'):

        r = 0

    elif(x=='Mod'):

        r = 1

    elif(x=='Sev'):

        r = 2

    else:

        r = 3

    return r



data['LandSlope'] = data['LandSlope'].apply(SlopeToInt)

data['CentralAir'] = data['CentralAir'].apply( lambda x: 0 if x == 'N' else 1) 

data['Street'] = data['Street'].apply( lambda x: 0 if x == 'Pave' else 1) 

data['PavedDrive'] = data['PavedDrive'].apply( lambda x: 0 if x == 'Y' else 1)



def GFinishToInt(x):

    if(x=='Fin'):

        r = 0

    elif(x=='RFn'):

        r = 1

    elif(x=='Unf'):

        r = 2

    else:

        r = 3

    return r



data['GarageFinish'] = data['GarageFinish'].apply(GFinishToInt)



def BsmtExposureToInt(x):

    if(x=='Gd'):

        r = 0

    elif(x=='Av'):

        r = 1

    elif(x=='Mn'):

        r = 2

    elif(x=='No'):

        r = 3

    else:

        r = 4

    return r

data['BsmtExposure'] = data['BsmtExposure'].apply(BsmtExposureToInt)



def FunctionalToInt(x):

    if(x=='Typ'):

        r = 0

    elif(x=='Min1'):

        r = 1

    elif(x=='Min2'):

        r = 1

    else:

        r = 2

    return r



data['Functional_int'] = data['Functional'].apply(FunctionalToInt)





def HouseStyleToInt(x):

    if(x=='1.5Unf'):

        r = 0

    elif(x=='SFoyer'):

        r = 1

    elif(x=='1.5Fin'):

        r = 2

    elif(x=='2.5Unf'):

        r = 3

    elif(x=='SLvl'):

        r = 4

    elif(x=='1Story'):

        r = 5

    elif(x=='2Story'):

        r = 6  

    elif(x==' 2.5Fin'):

        r = 7          

    else:

        r = 8

    return r



data['HouseStyle_int'] = data['HouseStyle'].apply(HouseStyleToInt)

data['HouseStyle_1st'] = 1*(data['HouseStyle'] == '1Story')

data['HouseStyle_2st'] = 1*(data['HouseStyle'] == '2Story')

data['HouseStyle_15st'] = 1*(data['HouseStyle'] == '1.5Fin')



def FoundationToInt(x):

    if(x=='PConc'):

        r = 3

    elif(x=='CBlock'):

        r = 2

    elif(x=='BrkTil'):

        r = 1        

    else:

        r = 0

    return r



data['Foundation_int'] = data['Foundation'].apply(FoundationToInt)



def MasVnrTypeToInt(x):

    if(x=='Stone'):

        r = 3

    elif(x=='BrkFace'):

        r = 2

    elif(x=='BrkCmn'):

        r = 1        

    else:

        r = 0

    return r



data['MasVnrType_int'] = data['MasVnrType'].apply(MasVnrTypeToInt)



def BsmtFinType1ToInt(x):

    if(x=='GLQ'):

        r = 6

    elif(x=='ALQ'):

        r = 5

    elif(x=='BLQ'):

        r = 4

    elif(x=='Rec'):

        r = 3   

    elif(x=='LwQ'):

        r = 2

    elif(x=='Unf'):

        r = 1        

    else:

        r = 0

    return r



data['BsmtFinType1_int'] = data['BsmtFinType1'].apply(BsmtFinType1ToInt)

data['BsmtFinType1_Unf'] = 1*(data['BsmtFinType1'] == 'Unf')

data['HasWoodDeck'] = (data['WoodDeckSF'] == 0) * 1

data['HasOpenPorch'] = (data['OpenPorchSF'] == 0) * 1

data['HasEnclosedPorch'] = (data['EnclosedPorch'] == 0) * 1

data['Has3SsnPorch'] = (data['3SsnPorch'] == 0) * 1

data['HasScreenPorch'] = (data['ScreenPorch'] == 0) * 1

data['YearsSinceRemodel'] = data['YrSold'].astype(int) - data['YearRemodAdd'].astype(int)

data['Total_Home_Quality'] = data['OverallQual'] + data['OverallCond']
def addlogs(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   

        res.columns.values[m] = l + '_log'

        m += 1

    return res



loglist = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']



data = addlogs(data, loglist)
def getdummies(res, ls):

    def encode(encode_df):

        encode_df = np.array(encode_df)

        enc = OneHotEncoder()

        le = LabelEncoder()

        le.fit(encode_df)

        res1 = le.transform(encode_df).reshape(-1, 1)

        enc.fit(res1)

        return pd.DataFrame(enc.transform(res1).toarray()), le, enc



    decoder = []

    outres = pd.DataFrame({'A' : []})



    for l in ls:

        cat, le, enc = encode(res[l])

        cat.columns = [l+str(x) for x in cat.columns]

        outres.reset_index(drop=True, inplace=True)

        outres = pd.concat([outres, cat], axis = 1)

        decoder.append([le,enc])     

    

    return (outres, decoder)



catpredlist = ['MSSubClass','MSZoning','LotShape','LandContour','LotConfig',

               'Neighborhood','Condition1','Condition2','BldgType',

               'RoofStyle','RoofMatl','Exterior1st','Exterior2nd',

               'BsmtFinType2','Heating','HouseStyle','Foundation','MasVnrType','BsmtFinType1',

               'Electrical','Functional','GarageType','Alley','Utilities',

               'GarageCond','Fence','MiscFeature','SaleType','SaleCondition','LandSlope','CentralAir',

               'GarageFinish','BsmtExposure','Street']



# Applying function to get dummies

# Saving decoder - function which can be used to transform new data  

res = getdummies(data[catpredlist],catpredlist)

df = res[0]

decoder = res[1]



# Adding real valued features

floatpredlist = ['LotFrontage_log',

                 'LotArea_log',

                 'MasVnrArea_log','BsmtFinSF1_log','BsmtFinSF2_log','BsmtUnfSF_log',

                 'TotalBsmtSF_log','1stFlrSF_log','2ndFlrSF_log','LowQualFinSF_log','GrLivArea_log',

                 'BsmtFullBath_log','BsmtHalfBath_log','FullBath_log','HalfBath_log','BedroomAbvGr_log','KitchenAbvGr_log',

                 'TotRmsAbvGrd_log','Fireplaces_log','GarageCars_log','GarageArea_log',

                 'PoolArea_log','MiscVal_log',

                 'YearRemodAdd','TotalSF_log','OverallQual','OverallCond','ExterQual','ExterCond','KitchenQual',

                 'HeatingQC','BsmtQual','BsmtCond','FireplaceQu','GarageQual','PoolQC','PavedDrive',

                 'HasWoodDeck', 'HasOpenPorch','HasEnclosedPorch', 'Has3SsnPorch', 'HasScreenPorch']

df = pd.concat([df,data[floatpredlist]],axis=1)
def addSquared(res, ls):

    m = res.shape[1]

    for l in ls:

        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   

        res.columns.values[m] = l + '_sq'

        m += 1

    return res 



sqpredlist = ['YearRemodAdd', 'LotFrontage_log', 

              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',

              'GarageCars_log', 'GarageArea_log',

              'OverallQual','ExterQual','BsmtQual','GarageQual','FireplaceQu','KitchenQual']

df = addSquared(data, sqpredlist)