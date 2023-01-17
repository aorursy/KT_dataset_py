import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

import scipy.stats as st

from scipy import stats

from scipy.special import boxcox1p

from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import MaxAbsScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import PolynomialFeatures

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/home-data-for-ml-course/train.csv')

df_test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
df_train.columns
df_train['SalePrice'].describe()
df_train.describe()
df_train.head()
# Checking distribution

y = df_train['SalePrice']

plt.figure(1); plt.title('Johnson SU')

sns.distplot(y, kde=False, fit=st.johnsonsu)

plt.figure(2); plt.title('Normal')

sns.distplot(y, kde=False, fit=st.norm)

plt.figure(3); plt.title('Log Normal')

sns.distplot(y, kde=False, fit=st.lognorm)

plt.show()
# Looking for outliers on features

#numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#numeric_train = df_train.select_dtypes(include=numerics)



#for feature in numeric_train:

#    if feature != 'SalePrice' and feature != 'Id':

#        plt.figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')

#        plt.scatter(numeric_train[feature], numeric_train['SalePrice'])

#        plt.xlabel(feature)

#        plt.xticks(rotation=90)

#        plt.ylabel('SalePrice')

#        plt.show()

# Cutting off numerical outliers

df_train = df_train.drop(df_train[(df_train.SalePrice >= 700000)].index)

df_train = df_train.drop(df_train[(df_train.LotFrontage >= 200)].index)

df_train = df_train.drop(df_train[(df_train.LotArea >= 100000)].index)

df_train = df_train.drop(df_train[(df_train.OverallQual == 4) & (df_train.SalePrice > 200000)].index)

df_train = df_train.drop(df_train[(df_train.OverallQual == 8) & (df_train.SalePrice > 500000)].index)

df_train = df_train.drop(df_train[(df_train.OverallQual == 10) & (df_train.SalePrice < 300000)].index)

df_train = df_train.drop(df_train[(df_train.OverallCond == 2) & (df_train.SalePrice > 300000)].index)

df_train = df_train.drop(df_train[(df_train.OverallCond == 5) & (df_train.SalePrice > 700000)].index)

df_train = df_train.drop(df_train[(df_train.OverallCond == 6) & (df_train.SalePrice > 700000)].index)

df_train = df_train.drop(df_train[(df_train.YearBuilt <= 1900) & (df_train.SalePrice > 200000)].index)

df_train = df_train.drop(df_train[(df_train.MasVnrArea >= 1200)].index)

df_train = df_train.drop(df_train[(df_train.BsmtFinSF1 >= 3000)].index)

df_train = df_train.drop(df_train[(df_train.BsmtFinSF2 >= 1200)].index)

df_train = df_train.drop(df_train[(df_train.TotalBsmtSF >= 4000)].index)

df_train = df_train.drop(df_train[(df_train['1stFlrSF'] >= 4000)].index)

df_train = df_train.drop(df_train[(df_train.LowQualFinSF > 500) & (df_train.SalePrice > 400000)].index)

df_train = df_train.drop(df_train[(df_train.GrLivArea >= 4000)].index)

df_train = df_train.drop(df_train[(df_train.BedroomAbvGr >= 7)].index)

df_train = df_train.drop(df_train[(df_train.KitchenAbvGr < 1)].index)

df_train = df_train.drop(df_train[(df_train.TotRmsAbvGrd > 12)].index)

df_train = df_train.drop(df_train[(df_train.Fireplaces > 2)].index)

df_train = df_train.drop(df_train[(df_train.GarageCars > 3)].index)

df_train = df_train.drop(df_train[(df_train.GarageArea > 1200) & (df_train.SalePrice < 300000)].index)

df_train = df_train.drop(df_train[(df_train.WoodDeckSF > 700)].index)

df_train = df_train.drop(df_train[(df_train.OpenPorchSF > 450)].index)

df_train = df_train.drop(df_train[(df_train.EnclosedPorch > 450)].index)

df_train = df_train.drop(df_train[(df_train['3SsnPorch'] > 350)].index)
# Joining DFs for ease of processing

full = pd.concat([df_train, df_test], ignore_index = True)

full.drop(['SalePrice'], axis=1, inplace=True)



# Starting to handle categoric columns



# "MiscFeature" column

words_set = set()

for data in full['MiscFeature']:

    if not pd.isnull(data):

        for word in data.split(' '):

            words_set.add(word)

#print ("Raw words set: {}".format(words_set))



for feature in words_set:

    count = 0

    for data in full['MiscFeature']:

        if not pd.isnull(data):

            if feature in data:

                count += 1

    print ("Category {}: {} samples".format(feature, count))
def add_columns (dataframe):

    add_categorical = ['Othr', 'TenC', 'Gar2', 'Shed']

    for column in add_categorical:

        data = dataframe['MiscVal'][dataframe['MiscFeature'] == column]

        dataframe[column] = pd.Series(data, index = dataframe.index).fillna(value=0)

    dataframe.drop(columns = ['MiscFeature'], inplace = True)

    dataframe.drop(columns = ['MiscVal'], inplace = True)



add_columns(full)

    

# Checking data shapes

print ("Full data shape is {}".format(full.shape))
plt.scatter(range(0, len(df_train['SalePrice'])), df_train['SalePrice'])

plt.show()
categorical_features = [

    'Alley',            # Will have to check on plot

    'BldgType',         # Will have to check on plot

    'BsmtCond',         # Simple mapping w/ NaN

    'BsmtExposure',     # Simple mapping w/ NaN

    'BsmtFinType1',     # Will have to check on plot

    'BsmtFinType2',     # Will have to check on plot

    'BsmtQual',         # Simple mapping w/ NaN

    'CentralAir',       # Yes/No

    'Condition1',       # Will have to check on plot

    'Condition2',       # Will have to check on plot

    'Electrical',       # Will have to check on plot

    'ExterCond',        # Simple mapping

    'Exterior1st',      # Will have to check on plot

    'Exterior2nd',      # Will have to check on plot

    'ExterQual',        # Simple mapping

    'Fence',            # Will have to check on plot

    'FireplaceQu',      # Simple mapping w/ NaN

    'Foundation',       # Will have to check on plot

    'Functional',       # Will have to check on plot

    'GarageCond',       # Simple mapping w/ NaN

    'GarageFinish',     # Simple mapping w/ NaN

    'GarageQual',       # Simple mapping w/ NaN

    'GarageType',       # Will have to check on plot

    'Heating',          # Will have to check on plot

    'HeatingQC',        # Simple mapping

    'HouseStyle',       # Will have to check on plot

    'KitchenQual',      # Simple mapping

    'LandContour',      # Will have to check on plot

    'LandSlope',        # Will have to check on plot

    'LotConfig',        # Will have to check on plot

    'LotShape',         # Will have to check on plot

    'MasVnrType',       # Will have to check on plot

    'MSZoning',         # Will have to check on plot

    'Neighborhood',     # Will use this to extract latitude/longitude

    'PavedDrive',       # Simple mapping

    'PoolQC',           # Simple mapping w/ NaN

    'RoofMatl',         # Will have to check on plot

    'RoofStyle',        # Will have to check on plot

    'SaleCondition',    # Will have to check on plot

    'SaleType',         # Will have to check on plot

    'Street',           # Simple mapping

    'Utilities'         # Will parse this into separate features

]



numerical_features = [

    '1stFlrSF',         # SqFt value, won't change anything here for now

    '2ndFlrSF',         # SqFt value, won't change anything here for now

    '3SsnPorch',        # SqFt value, won't change anything here for now

    'BsmtFinSF1',       # SqFt value, won't change anything here for now

    'BsmtFinSF2',       # SqFt value, won't change anything here for now

    'BsmtFullBath',     # Absolute number, won't change anything here for now

    'BsmtHalfBath',     # Absolute number, won't change anything here for now

    'BsmtUnfSF',        # SqFt value, won't change anything here for now

    'EnclosedPorch',    # SqFt value, won't change anything here for now

    'Fireplaces',       # Absolute number, won't change anything here for now

    'FullBath',         # Absolute number, won't change anything here for now

    'GarageArea',       # SqFt value, won't change anything here for now

    'GarageCars',       # Absolute number, won't change anything here for now

                        #

    'GarageYrBlt',      # Will use this for creating a "GarageAge" feature

                        #

    'GrLivArea',        # SqFt value, won't change anything here for now

    'HalfBath',         # Absolute number, won't change anything here for now

    'LotArea',          # SqFt value, won't change anything here for now

    'LotFrontage',      # Absolute number, won't change anything here for now

    'LowQualFinSF',     # SqFt value, won't change anything here for now

    'MasVnrArea',       # SqFt value, won't change anything here for now

    'MoSold',           # Absolute number, won't change anything here for now

    'MSSubClass',       # Absolute number, won't change anything here for now

    'OpenPorchSF',      # SqFt value, won't change anything here for now

    'OverallCond',      # Absolute number, won't change anything here for now

    'OverallQual',      # Absolute number, won't change anything here for now

                        #

    'PoolArea',         # Will use this for creating a "Pool" feature

                        #

    'ScreenPorch',      # SqFt value, won't change anything here for now

    'TotalBsmtSF',      # SqFt value, won't change anything here for now

    'TotRmsAbvGrd',     # Absolute number, won't change anything here for now

    'WoodDeckSF',       # SqFt value, won't change anything here for now

                        #

    'YearBuilt',        # Will use this for creating a "HouseAge" feature

                        #

    'YearRemodAdd',     # Will use this for creating a "RemodelAge" feature

                        #

    'YrSold',           # Absolute number, won't change anything here for now

    'BedroomAbvGr',     # Absolute number, won't change anything here for now

    'KitchenAbvGr',     # Absolute number, won't change anything here for now

    'Othr',             # Extracted from 'MiscFeature' column

    'TenC',             # Extracted from 'MiscFeature' column

    'Gar2',             # Extracted from 'MiscFeature' column

    'Shed'              # Extracted from 'MiscFeature' column

]
# Creating "GarageAge", "RemodelAge" and "HouseAge" features

def createAgeFeatures (dataframe):

    dataframe['GarageAge'] = dataframe['GarageYrBlt'].apply(lambda x: 2019 - x)

    dataframe['RemodelAge'] = dataframe['YearRemodAdd'].apply(lambda x: 2019 - x)

    dataframe['HouseAge'] = dataframe['YearBuilt'].apply(lambda x: 2019 - x)



# Creating "Pool" feature

def createPoolFeature (dataframe):

    dataframe['Pool'] = dataframe['PoolArea'].apply(lambda x: x != 0).map({True: 1, False: 0})



# Handling numerical features

def handleNumericalFeatures (dataframe):

    createAgeFeatures(dataframe)

    createPoolFeature(dataframe)



handleNumericalFeatures(full)



# Checking data shapes

print ("Full data shape is {}".format(full.shape))
# Mapping the "Simple mapping" features

def mapSimpleMapping (dataframe):

    # features = ['ExterCond', 'ExterQual', 'HeatingQC', 'KitchenQual', 'PavedDrive', 'Street']

    dataframe['ExterCond'] = dataframe['ExterCond'].map({

        'Ex' : 2,

        'Gd' : 1,

        'TA' : 0,

        'Fa' : -1,

        'Po' : -2

    })

    dataframe['ExterQual'] = dataframe['ExterQual'].map({

        'Ex' : 2,

        'Gd' : 1,

        'TA' : 0,

        'Fa' : -1,

        'Po' : -2

    })

    dataframe['HeatingQC'] = dataframe['HeatingQC'].map({

        'Ex' : 2,

        'Gd' : 1,

        'TA' : 0,

        'Fa' : -1,

        'Po' : -2

    })

    dataframe['KitchenQual'] = dataframe['KitchenQual'].map({

        'Ex' : 2,

        'Gd' : 1,

        'TA' : 0,

        'Fa' : -1,

        'Po' : -2

    })

    dataframe['PavedDrive'] = dataframe['PavedDrive'].map({

        'Y' : 1,

        'P' : 0.5,

        'N' : 0,

    })

    dataframe['Street'] = dataframe['Street'].map({

        'Grvl' : -1,

        'Pave' : 1,

    })



# Mapping the "Simple mapping w/ NaN" features

def mapSimpleWithNan (dataframe):

    #features = ['BsmtCond', 'BsmtExposure', 'BsmtQual', 'FireplaceQu',

    #            'GarageCond', 'GarageFinish', 'GarageQual', 'PoolQC']

    dataframe['BsmtCond'] = dataframe['BsmtCond'].map({

        'Ex' : 4,

        'Gd' : 3,

        'TA' : 2,

        'Fa' : 1,

        'Po' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['BsmtExposure'] = dataframe['BsmtExposure'].map({

        'Gd' : 3,

        'Av' : 2,

        'Mn' : 1,

        'No' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['BsmtQual'] = dataframe['BsmtQual'].map({

        'Ex' : 4,

        'Gd' : 3,

        'TA' : 2,

        'Fa' : 1,

        'Po' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['FireplaceQu'] = dataframe['FireplaceQu'].map({

        'Ex' : 4,

        'Gd' : 3,

        'TA' : 2,

        'Fa' : 1,

        'Po' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['GarageCond'] = dataframe['GarageCond'].map({

        'Ex' : 4,

        'Gd' : 3,

        'TA' : 2,

        'Fa' : 1,

        'Po' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['GarageFinish'] = dataframe['GarageFinish'].map({

        'Fin' : 2,

        'RFn' : 1,

        'Unf' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['GarageQual'] = dataframe['GarageQual'].map({

        'Ex' : 4,

        'Gd' : 3,

        'TA' : 2,

        'Fa' : 1,

        'Po' : 0,

        'NA' : -1

    }).fillna(-1)

    dataframe['PoolQC'] = dataframe['PoolQC'].map({

        'Ex' : 3,

        'Gd' : 2,

        'TA' : 1,

        'Fa' : 0,

        'NA' : -1

    }).fillna(-1)



# Mapping boolean features

def mapBoolean (dataframe):

    dataframe['CentralAir'] = dataframe['CentralAir'].map({

        'N' : 0,

        'Y' : 1,

    })



# Handling mapped categorical features

def handleMappedCategorical (dataframe):

    mapSimpleMapping(dataframe)

    mapSimpleWithNan(dataframe)

    mapBoolean(dataframe)



handleMappedCategorical(full)
# Checking plots for other categorical features

check_plot_features = [

    'Alley',

    'BldgType',

    'BsmtFinType1',

    'BsmtFinType2',

    'Condition1',

    'Condition2',

    'Electrical',

    'Exterior1st',

    'Exterior2nd',

    'Fence',

    'Foundation',

    'Functional',

    'GarageType',

    'Heating',

    'HouseStyle',

    'LandContour',

    'LandSlope',

    'LotConfig',

    'LotShape',

    'MasVnrType',

    'MSZoning',

    'RoofMatl',

    'RoofStyle',

    'SaleCondition',

    'SaleType'

]



#for feature in check_plot_features:

#    data = pd.concat([df_train['SalePrice'], df_train[feature]], axis=1)

#    f, ax = plt.subplots(figsize=(16, 10))

#    fig = sns.boxplot(x=feature, y="SalePrice", data=data)

#    fig.axis(ymin=0);

#    xt = plt.xticks(rotation=45)

#    plt.xticks(rotation=90)

#    plt.show()

#

#    After plots observation, I decided to proceed with the following:

#

#    'Alley',

#	'Grvl' : 0, 'Pave' : 1

#

#    'BldgType',

#	One-hot encoding only

#

#    'BsmtFinType1',

#	One-hot encoding only

#

#    'BsmtFinType2',

#	One-hot encoding only

#

#    'Condition1',

#	Create features using information about distance

#	One-hot encoding

#

#    'Condition2',

#	Create features using information about distance

#	One-hot encoding

#

#    'Electrical',

#	'Mix' : 0, 'FuseP' : 1, 'FuseF' : 2, 'FuseA' : 3, 'SBrkr' : 4

#

#    'Exterior1st',

#	One-hot encoding only

#

#    'Exterior2nd',

#	One-hot encoding only

#

#    'Fence',

#	Separate in two features:

#	'FencePrivacy'

#		'MnPrv' : 0, 'GdPrv': 1, 'NA' : -1

#	'FenceWood'

#		'MnWw' : 0, 'GdWo' : 1, 'NA' : -1

#

#    'Foundation',

#	One-hot encoding only

#

#    'Functional',

#	'Typ' : 0, 'Min1' : -1, 'Min2' : -2, 'Mod' : -3, 'Maj1' : -4, 'Maj2' : -5, 'Sev' : -6, 'Sal' : -7

#

#    'GarageType',

#	One-hot encoding only

#

#    'Heating',

#	One-hot encoding only

#

#    'HouseStyle',

#	One-hot encoding only

#

#    'LandContour',

#	One-hot encoding only

#

#    'LandSlope',

#	One-hot encoding only

#

#    'LotConfig',

#	One-hot encoding only

#

#    'LotShape',

#	'Reg' : 0, 'IR1' : -1, 'IR2' : -2, 'IR3' : -3

#

#    'MasVnrType',

#	One-hot encoding only

#

#    'MSZoning',

#	One-hot encoding only

#

#    'RoofMatl',

#	One-hot encoding only

#

#    'RoofStyle',

#	One-hot encoding only

#

#    'SaleCondition',

#	One-hot encoding only

#

#    'SaleType'

#	One-hot encoding only

#

#    'Neighborhood'

#	Will use this to extract latitude/longitude

#

#    'Utilities'

#	Will parse this into separate features



# One-hot encodings

one_hots = ['BldgType', 'BsmtFinType1', 'BsmtFinType2', 'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd',

           'Foundation', 'GarageType', 'Heating', 'HouseStyle', 'LandContour', 'LandSlope', 'LotConfig',

           'MasVnrType', 'MSZoning', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType']



def doOneHot (dataframe):

    try:

        for category in one_hots:

            df = pd.Categorical(dataframe[category])

            dfDummies = pd.get_dummies(df, prefix = category)

            dataframe = pd.concat([dataframe, dfDummies], axis=1)

            if category not in ['Condition1', 'Condition2']:

                dataframe.drop(columns = [category], inplace = True)

        return dataframe

    except KeyError:

        print ("Oops! Category {} not found! Probably this has already been done...".format(category))

        return dataframe

        



# Processing one-hots

full = doOneHot(full)



# Checking data shapes

print ("Full data shape is {}".format(full.shape))
# Still missing handling:

# - Alley

# - Condition1

# - Condition2

# - Electrical

# - Fence

# - Functional

# - LotShape

# - Neighborhood

# - Utilities



def handleAlley (dataframe):

    dataframe['Alley'] = dataframe['Alley'].map({'Grvl' : 0, 'Pave' : 1})



def handleConditions (dataframe):

    handler = {

        'Normal' : 0,  # ??

        'RRNn'   : 1,  # Within 200'

        'RRNe'   : 1,  #

        'PosN'   : 2,  # Near

        'Artery' : 3,  # Adjacent

        'Feedr'  : 3,  #

        'RRAn'   : 3,

        'PosA'   : 3,

        'RRAe'   : 3

    }

    dataframe['Condition1'] = dataframe['Condition1'].map(handler)

    dataframe['Condition2'] = dataframe['Condition2'].map(handler)

    

def handleElectrical (dataframe):

    handler = {'Mix' : 0, 'FuseP' : 1, 'FuseF' : 2, 'FuseA' : 3, 'SBrkr' : 4}

    dataframe['Electrical'] = dataframe['Electrical'].map(handler)



def handleFunctional (dataframe):

    handler = {'Typ' : 0, 'Min1' : -1, 'Min2' : -2, 'Mod' : -3, 'Maj1' : -4, 'Maj2' : -5, 'Sev' : -6, 'Sal' : -7}

    dataframe['Functional'] = dataframe['Functional'].map(handler)



def handleLotShape (dataframe):

    handler = {'Reg' : 0, 'IR1' : -1, 'IR2' : -2, 'IR3' : -3}

    dataframe['LotShape'] = dataframe['LotShape'].map(handler)

    

def handlePostOneHots (dataframe):

    handleAlley(dataframe)

    handleConditions(dataframe)

    handleElectrical(dataframe)

    handleFunctional(dataframe)

    handleLotShape(dataframe)

    

handlePostOneHots(full)



# Checking data shapes

print ("Full data shape is {}".format(full.shape))
# Still missing handling:

# - Fence

# - Neighborhood

# - Utilities



def handleFence (dataframe):

    dataframe['FencePrivacy'] = dataframe['Fence'].map({'MnPrv' : 0, 'GdPrv': 1}).fillna(-1)

    dataframe['FenceWood'] = dataframe['Fence'].map({'MnWw' : 0, 'GdWo': 1}).fillna(-1)

    dataframe.drop(columns = ['Fence'], inplace = True)

    return dataframe



full = handleFence (full)



# Checking data shapes

print ("Full data shape is {}".format(full.shape))
# Still missing handling:

# - Neighborhood

# - Utilities



def handleUtilities (dataframe):

    dataframe['Electricity'] = dataframe['Utilities'].map({'AllPub' : 1, 'NoSewr' : 1, 'NoSeWa' : 1, 'ELO' : 1}).fillna(0)

    dataframe['Gas'] = dataframe['Utilities'].map({'AllPub' : 1, 'NoSewr' : 1, 'NoSeWa' : 1, 'ELO' : 0}).fillna(0)

    dataframe['Water'] = dataframe['Utilities'].map({'AllPub' : 1, 'NoSewr' : 1, 'NoSeWa' : 0, 'ELO' : 0}).fillna(0)

    dataframe['Septic Tank'] = dataframe['Utilities'].map({'AllPub' : 1, 'NoSewr' : 0, 'NoSeWa' : 0, 'ELO' : 0}).fillna(0)

    dataframe.drop(columns = ['Utilities'], inplace = True)

    return dataframe



full = handleUtilities (full)



# Checking data shapes

print ("Full data shape is {}".format(full.shape))
# Still missing handling:

# - Neighborhood



geo_heatmap = {

    'Neighborhood' : [

        'Blmngtn',

        'Blueste',

        'BrDale',

        'BrkSide',

        'ClearCr',

        'CollgCr',

        'Crawfor',

        'Edwards',

        'Gilbert',

        'IDOTRR',

        'MeadowV',

        'Mitchel',

        'NAmes',

        'NoRidge',

        'NPkVill',

        'NridgHt',

        'NWAmes',

        'OldTown',

        'SWISU',

        'Sawyer',

        'SawyerW',

        'Somerst',

        'StoneBr',

        'Timber',

        'Veenker'

    ],

    'Latitude' : [

        42.0563761,

        42.0218678,

        42.052795,

        42.024546,

        42.0360959,

        42.0214232,

        42.028025,

        42.0154024,

        42.1068177,

        42.0204395,

        41.997282,

        41.9903084,

        42.046618,

        42.048164,

        42.0258352,

        42.0597732,

        42.0457802,

        42.029046,

        42.0266187,

        42.0295218,

        42.034611,

        42.0508817,

        42.0595539,

        41.9999732,

        42.0413042

    ],

    'Longitude' : [

        -93.6466598,

        -93.6702853,

        -93.6310097,

        -93.6545201,

        -93.6575849,

        -93.6584089,

        -93.6093286,

        -93.6875441,

        -93.6553512,

        -93.6243787,

        -93.6138098,

        -93.603242,

        -93.6362807,

        -93.6496766,

        -93.6613958,

        -93.65166,

        -93.6472075,

        -93.6165288,

        -93.6486541,

        -93.7102833,

        -93.7024257,

        -93.6485768,

        -93.6365891,

        -93.6518812,

        -93.6524905

    ]

}



train_neighborhood = set(df_train['Neighborhood'].tolist())



geo_dataframe = pd.DataFrame.from_dict(geo_heatmap)



geo_dataframe = geo_dataframe[geo_dataframe['Neighborhood'].isin(train_neighborhood)]



geo_dataframe["SalePrice"] = pd.Series(df_train.groupby(["Neighborhood"]).mean()["SalePrice"].values, index = geo_dataframe.index)



import folium

from folium.plugins import HeatMap

max_amount = float(geo_dataframe['SalePrice'].max())

hmap = folium.Map(location = [42.045042,-93.6473567], zoom_start = 12)

hm_wide = HeatMap (list (zip (geo_dataframe.Latitude.values, geo_dataframe.Longitude.values, geo_dataframe.SalePrice.values)),

                   min_opacity = 0.4,

                   max_val = max_amount,

                       radius = 17,

                   blur = 15,

                   max_zoom = 1

                  )

hmap.add_child(hm_wide)

# Building auxiliar dicts

lat_dict = {}

lon_dict = {}

for i in range(len(geo_heatmap['Neighborhood'])):

    neighborhood = geo_heatmap['Neighborhood'][i]

    lat = geo_heatmap['Latitude'][i]

    lon = geo_heatmap['Longitude'][i]

    lat_dict[neighborhood] = lat

    lon_dict[neighborhood] = lon



# Method that adds latitude and longitude columns

def add_lat_lon_columns (df):

    df ['Latitude'] = df['Neighborhood'].map(lat_dict)

    df ['Longitude'] = df['Neighborhood'].map(lon_dict)

    df.drop(columns=['Neighborhood'], inplace=True)

    

# Adding columns to train and test sets

add_lat_lon_columns(full)



# Checking data shapes

print ("Full data shape is {}".format(full.shape))
# Checking for missing data

missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Handling KitchenQual w/ average values

full['KitchenQual'] = full['KitchenQual'].fillna(0)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing Garage stuff

full[full['GarageArea'].isnull()]
# Will fill this single one with "no-garage" information

new = full.iloc[2576].copy()

new['GarageArea'] = 0

new['GarageCars'] = 0

new['GarageCond'] = -1

new['GarageFinish'] = -1

new['GarageQual'] = -1



full.iloc[2576] = new.copy()



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# As the missing "Electrical" is from our train dataframe, put the average value (from 0 to 4)

full['Electrical'] = full['Electrical'].fillna(2)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing BsmtFinSF1

full[full['BsmtFinSF1'].isnull()]
# As BsmtFinSF1, BsmtFinSF2, TotalBsmtSF and BsmtUnfSF are sqft data of houses without basement, will fill

# with zeros

full['BsmtFinSF1'] = full['BsmtFinSF1'].fillna(0)

full['BsmtFinSF2'] = full['BsmtFinSF2'].fillna(0)

full['TotalBsmtSF'] = full['TotalBsmtSF'].fillna(0)

full['BsmtUnfSF'] = full['BsmtUnfSF'].fillna(0)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing BsmtFullBath

full[full['BsmtFullBath'].isnull()]
# As these samples refer to houses without basement, fill with zeros

full['BsmtFullBath'] = full['BsmtFullBath'].fillna(0)

full['BsmtHalfBath'] = full['BsmtHalfBath'].fillna(0)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing Functional

full[full['Functional'].isnull()]
# Before I handle this feature, I will check for others trying to acquire enough data to predict it's

# functionality



# Checking missing MasVnrArea

full[full['MasVnrArea'].isnull()]
# As MasVnrArea refers to sqft data, will assume these houses have none

full['MasVnrArea'] = full['MasVnrArea'].fillna(0)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing GarageYrBlt

full[full['GarageYrBlt'].isnull()]
# As it has no data on the year of construction of the garagage, I will assume it's been built the same year

# the house was built.

null_df = full[full['GarageYrBlt'].isnull()]

null_df['GarageAge'] = null_df['YearBuilt'].apply(lambda x: 2019 - x)

null_df['GarageYrBlt'] = null_df['YearBuilt']



# Replacing data

full[full['GarageYrBlt'].isnull()] = null_df



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing LotFrontage

full[full['LotFrontage'].isnull()]
# Tried to predict LotFrontage using LotArea and GrLivArea w/out success

# Will now handle Condition2 and Condition1

full['Condition1'] = full['Condition1'].fillna(0)

full['Condition2'] = full['Condition2'].fillna(0)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Handling Alley missing data based on the data_description file

full['Alley'] = full['Alley'].fillna(-1)



missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Will now retry to predict log(LotFrontage) using log(LotArea)

# Idea extracted from https://www.kaggle.com/clustersrus/house-prices-dealing-with-the-missing-data

full['LogLotArea'] = np.log(full['LotArea'])

full['LogLotFrontage'] = np.log(full['LotFrontage'])
plt.scatter(full['LogLotArea'], full['LogLotFrontage'] )
missing_mask = full.isnull()['LotFrontage']

filled_mask  = full.notnull()['LotFrontage']

predict = full[missing_mask]

train = full[filled_mask]



X_train = train['LogLotArea']

y_train = train['LogLotFrontage']

X_predict = predict['LogLotArea']

model = XGBRegressor(n_jobs = -1)



model.fit(np.array(X_train).reshape(-1,1), np.array(y_train))

y_predict = model.predict(np.array(X_predict).reshape(-1,1))

y_predict.shape



full['LogLotFrontage'][missing_mask] = y_predict
missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Filling LotFrontage by doing reverse log transformation

full['LotFrontage'][missing_mask] = np.exp(full['LogLotFrontage'][missing_mask])

missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Filling functional with the Typical value

full['Functional'] = full['Functional'].fillna(0)

missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Checking missing MasVnrArea

full[full['GarageCars'].isnull()]['GarageCond']
full['GarageCars'] = full['GarageCars'].fillna(-1)

missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
full[full['GarageArea'].isnull()]['GarageCond']
full['GarageArea'] = full['GarageArea'].fillna(-1)

missing_values = full.isnull().sum()

missing_values[missing_values>0].sort_values(ascending = False)
# Adding transformed features (only high skewed will be transformed)

# Idea extracted from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# Improvements were initially proposed here https://www.kaggle.com/gabrielmilan/pre-os-de-im-veis-em-recife-pe/notebook

skewed_feats = full.apply(lambda x: st.skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



skewed_features = skewness.index



walk_through_lambdas = True



# Method that gets best lambda for a given distribution

def getBestLambda (data):

    from scipy.stats import boxcox, normaltest



    statistic = []

    pvalue = []



    bclambda_range = np.linspace(-3, 3, 1000)



    for bclambda in bclambda_range:

        transform = boxcox1p(data, bclambda)

        stat, pval = normaltest(transform)

        statistic.append(stat)

        pvalue.append(pval)



    lowest_stat = 8000

    closest = -10

    best_stat = -1

    best_pval = -1

    for i in range(len(bclambda_range)):

        if statistic[i] < lowest_stat:

            lowest_stat = statistic[i]

            best_stat = i

        if (abs(pvalue[i]) - 1) > closest:

            closest = (abs(pvalue[i]) - 1)

            best_pval = i



    return bclambda_range[best_pval], bclambda_range[best_stat]



from tqdm import tqdm



for feat in tqdm(skewed_features):

    if walk_through_lambdas:

        full["BC_{}".format(feat)] = boxcox1p(full[feat], getBestLambda(full[feat])[0])

    else:

        full["BC_{}".format(feat)] = boxcox1p(full[feat], .15)

        

# Clearing columns with null values after transforming

missing_values = full.isnull().sum()

for column in missing_values[missing_values>0].keys():

    print ("Dropping feature {}".format(column))

    full.drop(column, axis = 1, inplace = True)
# Split again into train and test

X_train = full[full['Id'].isin(df_train['Id'])]

X_train.drop('Id', axis = 1, inplace = True)

X_test  = full[full['Id'].isin(df_test['Id'])]

X_test.drop('Id', axis = 1, inplace = True)

y_train = df_train['SalePrice']



print ("X_train shape is {}, y_train shape is {} and X_test shape is {}".format(

    X_train.shape,

    y_train.shape,

    X_test.shape

))
from sklearn.preprocessing import StandardScaler



# Scaling data

scaler = StandardScaler()



X_train_scaled = scaler.fit(X_train).transform(X_train)

X_test_scaled = scaler.transform(X_test)



bclambda = getBestLambda(y_train)[0]

y_train_transformed = boxcox1p(y_train, bclambda)#np.log1p(y_train)#y_train.copy()#
#

# Modeling

#



# Validation function extracted from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

n_folds = 5



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



# Define Root Mean Square Error w/ Cross-validation 

def rmse_cv (model, X, y, cv=5):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring = "neg_mean_squared_error", cv = cv))

    return rmse



# Define Pure Root Mean Square Error

from sklearn.metrics import mean_squared_error

from math import sqrt

def rmse (y_actual, y_predicted):

     return sqrt(mean_squared_error(y_actual, y_predicted))

#

# Extracted from https://www.kaggle.com/gabrielmilan/pre-os-de-im-veis-em-recife-pe/notebook

#



from xgboost import plot_importance



model = XGBRegressor(nthread = -1)

model.fit(X_train, y_train_transformed)



f, ax = plt.subplots(figsize=(16, 10))

plot_importance(model, ax = ax)

plt.show()



# Getting chart data

fscore_dict = model.get_booster().get_fscore()



# Getting threshold values

threshold_values = []

for feature in fscore_dict:

    if fscore_dict[feature] not in threshold_values:

        threshold_values.append(fscore_dict[feature])



# Building lists for later plotting

scores_y = []

        

# Setting a range for the threshold

for threshold in threshold_values:

    feature_list = [feature for feature in X_train.columns if (feature in fscore_dict) and (fscore_dict[feature] >=  threshold)]

    scores_y.append(rmse_cv(model, X_train[feature_list], y_train_transformed).mean())

    

# Plotting

plt.figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')

plt.scatter(threshold_values, scores_y)

plt.show()



# Getting best threshold

min_value = 1000000000000000

best_threshold = 1000

for i in range(len(threshold_values)):

    if scores_y [i] < min_value:

        best_threshold = threshold_values[i]



print ("Your best threshold is {}".format(best_threshold))

xgb_features_list = [feature for feature in fscore_dict if fscore_dict[feature] >= best_threshold]

print ("With this threshold, the features you want are: {}".format(xgb_features_list))

X_train_xgb = X_train [xgb_features_list]

X_train_xgb_scaled = scaler.fit(X_train_xgb).transform(X_train_xgb)

X_test_xgb = X_test [xgb_features_list]

X_test_xgb_scaled = scaler.transform(X_test_xgb)
from sklearn.feature_selection import RFECV



#model = lgb.LGBMRegressor()

#selector = RFECV (model)

#selector.fit(X_train, y_train_transformed)



#RFECV_mask = selector.support_



#rfecv_features_list = []

#for i in range(len(RFECV_mask)):

#    if RFECV_mask[i] == True:

#        rfecv_features_list.append(X_train.columns[i])

#    

#print ("For this metod, the features you want are: {}".format(rfecv_features_list))



rfecv_features_list = ['1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtCond', 'BsmtExposure',

                       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtQual', 'BsmtUnfSF',

                       'CentralAir', 'Condition1', 'Electrical', 'EnclosedPorch', 'ExterCond',

                       'ExterQual', 'FireplaceQu', 'FullBath', 'Functional', 'GarageArea',

                       'GarageCars', 'GarageFinish', 'GarageQual', 'GarageYrBlt', 'GrLivArea',

                       'HalfBath', 'HeatingQC', 'KitchenAbvGr', 'KitchenQual', 'LotArea',

                       'LotFrontage', 'LotShape', 'LowQualFinSF', 'MSSubClass', 'MasVnrArea',

                       'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PavedDrive',

                       'ScreenPorch', 'TotRmsAbvGrd', 'TotalBsmtSF', 'WoodDeckSF', 'YearBuilt',

                       'YearRemodAdd', 'YrSold', 'BsmtFinType1_ALQ', 'BsmtFinType1_GLQ',

                       'BsmtFinType2_BLQ', 'Condition1_Artery', 'Exterior1st_BrkFace',

                       'Exterior1st_HdBoard', 'Exterior1st_MetalSd', 'Exterior1st_Plywood',

                       'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Foundation_CBlock',

                       'Foundation_PConc', 'GarageType_Attchd', 'GarageType_Detchd',

                       'HouseStyle_1Story', 'LandContour_Lvl', 'LotConfig_CulDSac',

                       'LotConfig_FR2', 'MasVnrType_BrkFace', 'MasVnrType_Stone', 'MSZoning_RL',

                       'MSZoning_RM', 'RoofStyle_Gable', 'SaleCondition_Abnorml',

                       'SaleCondition_Normal', 'SaleType_New', 'SaleType_WD', 'FenceWood',

                       'Latitude', 'Longitude', 'BC_LotArea']
X_train_rfecv = X_train [rfecv_features_list]

X_train_rfecv_scaled = scaler.fit(X_train_rfecv).transform(X_train_rfecv)

X_test_rfecv = X_test [rfecv_features_list]

X_test_rfecv_scaled = scaler.transform(X_test_rfecv)
# Heatmap of positive correlation features

import seaborn as sns

correlation = df_train.corr()

k = len([i for i in correlation['SalePrice'] if abs(i) >= 0.4])

cols = correlation.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

f , ax = plt.subplots(figsize = (18,16))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)

ax.set_title('SalePrice correlation heatmap')

plt.show()

my_cols = list(cols)

if "SalePrice" in my_cols:

    my_cols.remove("SalePrice")
X_train_corr = X_train[my_cols]

X_train_corr.shape
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.svm import SVR



vanilla_input_size = 0



def VanillaRegressor (input_size = None):

    # Create model

    model = Sequential()

    model.add(Dense(5, input_dim=input_size, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def BetaRegressor (input_size = None):

    model = Sequential()

    model.add(Dense(128, kernel_initializer='normal',input_dim = input_size, activation='relu'))

    model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    model.add(Dense(256, kernel_initializer='normal',activation='relu'))

    model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

    

train_sets = [

    X_train,

    #X_train_scaled,

    X_train_xgb,

    #X_train_xgb_scaled,

    X_train_rfecv,

    #X_train_rfecv_scaled,

    X_train_corr

]



other_models = [

    ElasticNet(),                       # X_train          -> 0.1308

    Lasso(),                            # X_train          -> 0.1393

    BayesianRidge(),                    # X_train_rfecv    -> 0.0953

    LassoLarsIC(),                      # X_train_rfecv    -> 0.1034

    RandomForestRegressor(),            # X_train          -> 0.1193

    GradientBoostingRegressor(),        # X_train_rfecv    -> 0.1002

    KernelRidge(),                      # X_train_rfecv    -> 0.0959

    xgb.XGBRegressor(),                 # X_train_rfecv    -> 0.0997

    xgb.XGBRFRegressor(),               # <any>            -> 0.3244

    lgb.LGBMRegressor(),                # X_train_rfecv    -> 0.1024

    SVR()                               # X_train_corr     -> 0.3224

]



other_models_names = [

    'ElasticNet',

    'Lasso',

    'BayesianRidge',

    'LassoLarsIC',

    'RandomForestRegressor',

    'GradientBoostingRegressor',

    'KernelRidge',

    'XGBRegressor',

    'XGBRFRegressor',

    'LGBMRegressor',

    'SVR'

]



# Evaluating models

# for j in range(len(other_models)):

#     for i in range(len(train_sets)):

#         score = rmse_cv(other_models[j], train_sets[i], y_train_transformed)

#         print ("{} scored {:.4f}(+-{:.4f}) with the set #{}".format(other_models_names[j], score.mean(), score.std(), i+1))
# Choosing one X_train for hyper-parameter tuning

chosen_X_train = X_train_rfecv.copy()

chosen_X_test = X_test_rfecv.copy()
# Grid search for hyper-parameters

from sklearn.model_selection import GridSearchCV

class grid():

    def __init__ (self, model):

        self.model = model

    def grid_get (self, X, y, param_grid):

        grid_search = GridSearchCV(self.model, param_grid, scoring='neg_mean_squared_error', verbose=99)

        grid_search.fit(X, y)

        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))

        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])

        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])



make_search = False

max_iter = 15000

dataset = chosen_X_train



# Will keep the following models for hyper-params tuning:

# - BayesianRidge

# - GradientBoostingRegressor

# - KernelRidge

# - XGBRegressor

# - LGBMRegressor

#	BayRidge	GBR	KernelR	XGB	LGBM	

#1	0.0956	0.1018	0.0971	0.1008	0.1028	0.4981

#3	0.1066	0.1049	0.1063	0.106	0.1079	0.5317

#5	0.0953	0.1006	0.0959	0.1005	0.1022	0.4945

#	0.2975	0.3073	0.2993	0.3073	0.3129	

# Grid search for Gradient Boost Regressor

if (make_search):

    grid(GradientBoostingRegressor()).grid_get(dataset, y_train_transformed, {

        'loss' : ['ls', 'huber'],

        'learning_rate' : np.linspace(0.01, 0.1, 11),

        'n_estimators' : np.linspace(300, 700, 6).astype(int)

    })



# {'learning_rate': 0.064, 'loss': 'huber', 'n_estimators': 620}
# Grid search for BayesianRidge

#if (make_search):

#    grid(BayesianRidge()).grid_get(dataset, y_train_transformed, {

#        'alpha_1' : np.linspace(1e-7, 1e-5, 101),

#        'alpha_2' : np.linspace(1e-7, 1e-5, 101),

#        'lambda_1': np.linspace(1e-7, 1e-5, 101),

#        'lambda_2': np.linspace(1e-7, 1e-5, 101),

#    })



# Can't tune BayRidge for my PC crashes
# Grid search for LGBMRegressor

if (make_search):

    grid(lgb.LGBMRegressor()).grid_get(dataset, y_train_transformed, {

        'n_jobs'        : [-1],

        'boosting_type' : ['gbdt'],

        'num_leaves'    : np.linspace(1, 50, 11).astype(int),

        'learning_rate' : np.linspace(0.01, 0.1, 11),

        'n_estimators'  : np.linspace(100, 700, 6).astype(int)

    })



# {'boosting_type': 'gbdt', 'learning_rate': 0.028000000000000004, 'n_estimators': 700, 'n_jobs': -1, 'num_leaves': 5}
import xgboost as xgb

from sklearn.pipeline import make_pipeline



train_sets = [

    X_train_rfecv,

]



br = BayesianRidge()

llic = LassoLarsIC()

gb = GradientBoostingRegressor(learning_rate=0.064, loss='huber', n_estimators=620)

kr = KernelRidge()

lgbm = lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.028000000000000004, n_estimators=700, n_jobs=-1, num_leaves=5)

xgbr = xgb.XGBRegressor()



other_models = [

#     BayesianRidge(),                                                           # --> 0.0953

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), br),                  # 0.1010

#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), br),                  # 0.1011

#     make_pipeline(StandardScaler(), PolynomialFeatures(2), br),                # 0.1343

#     make_pipeline(RobustScaler(), PolynomialFeatures(2), br),                  # 0.2745

#     make_pipeline(Normalizer(), PolynomialFeatures(2), br),                    # 0.1106

#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), br),           # 0.1053

#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), br),              # 0.1392



#     LassoLarsIC(),                                                               # 0.1034

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), llic),                  # 0.1096

#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), llic),                  # --> 0.1034

#     make_pipeline(StandardScaler(), PolynomialFeatures(2), llic),                # 0.2979

#     make_pipeline(RobustScaler(), PolynomialFeatures(2), llic),                  # 0.2979

#     make_pipeline(Normalizer(), PolynomialFeatures(2), llic),                    # 0.3098

#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), llic),           # 0.1160

#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), llic),              # 0.2901



#     GradientBoostingRegressor(),                                               # --> 0.1002

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), gb),                  # 0.1020

#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), gb),                  # 0.1044

#     make_pipeline(StandardScaler(), PolynomialFeatures(2), gb),                # 0.1073

#     make_pipeline(RobustScaler(), PolynomialFeatures(2), gb),                  # 0.1053

#     make_pipeline(Normalizer(), PolynomialFeatures(2), gb),                    # 0.1211

#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), gb),           # 0.1034

#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), gb),              # 0.1080



#     KernelRidge(),                                                             # --> 0.0959

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), kr),                  # 0.2539

#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), kr),                  # 0.1260

#     make_pipeline(StandardScaler(), PolynomialFeatures(2), kr),                # 1.5792

#     make_pipeline(RobustScaler(), PolynomialFeatures(2), kr),                  # 7.8296

#     make_pipeline(Normalizer(), PolynomialFeatures(2), kr),                    # 0.2113

#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), kr),           # 0.3479

#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), kr),              # 1.2862



#     xgb.XGBRegressor(),                                                          # --> 0.0997

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), xgbr),                  # 0.1037

#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), xgbr),                  # 0.1068

#     make_pipeline(StandardScaler(), PolynomialFeatures(2), xgbr),                # 0.1075

#     make_pipeline(RobustScaler(), PolynomialFeatures(2), xgbr),                  # 0.1097

#     make_pipeline(Normalizer(), PolynomialFeatures(2), xgbr),                    # 0.1270

#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), xgbr),           # 0.1047

#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), xgbr),              # 0.1069



#     lgb.LGBMRegressor(),                                                         # 0.1024

#     make_pipeline(MinMaxScaler(), PolynomialFeatures(2), lgbm),                  # --> 0.0999

#     make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), lgbm),                  # 0.1031

#     make_pipeline(StandardScaler(), PolynomialFeatures(2), lgbm),                # 0.1054

#     make_pipeline(RobustScaler(), PolynomialFeatures(2), lgbm),                  # 0.1048

#     make_pipeline(Normalizer(), PolynomialFeatures(2), lgbm),                    # 0.1197

#     make_pipeline(QuantileTransformer(), PolynomialFeatures(2), lgbm),           # 0.1016

#     make_pipeline(PowerTransformer(), PolynomialFeatures(2), lgbm),              # 0.1059

    

]



# Evaluating models

for j in range(len(other_models)):

    for i in range(len(train_sets)):

        score = rmse_cv(other_models[j], train_sets[i], y_train_transformed, cv=2)

        print ("{} scored {:.4f}(+-{:.4f}) with the set #{}".format(other_models[j], score.mean(), score.std(), i+1))
# Defining function for making models combinations

def make_combinations (iterable):

    from itertools import combinations

    my_combs = []

    for item in iterable.copy():

        iterable.remove(item)

        for i in range(len(iterable)):

            for comb in combinations(iterable, i+1):

                my_combs.append((item, comb))

        iterable.append(item)

    return my_combs



models = [

    # 0.09534141759933776

    BayesianRidge(),

    # 0.10947055605680345

    make_pipeline(MaxAbsScaler(), PolynomialFeatures(2), LassoLarsIC()),

    # 0.09675792855029856

    GradientBoostingRegressor(learning_rate=0.064, loss='huber', n_estimators=620),

    # 0.09594388448467231

    KernelRidge(),

    # 0.09972041539723736

    xgb.XGBRegressor(),

    # 0.09624848214154685

    make_pipeline(MinMaxScaler(), PolynomialFeatures(2), lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.028000000000000004, n_estimators=700, n_jobs=-1, num_leaves=5))

]



my_combs = make_combinations(models)



print ("I have {} combinations to test!".format(len(my_combs)))
# Testing every possible combination

print ("Testing raw models...")

i = 0

results = []

best = 10000

# with progressbar.ProgressBar(max_value=len(models)) as bar:

#     for model in models:

#         X = chosen_X_train#.values

#         score = rmse_cv(model, X, y_train_transformed).mean()

#         results.append(score)

#         print (score)

#         if (score < best):

#             best = score

#             best_model = model

#         i+=1

#         bar.update(i)
class CustomEnsemble (BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models, meta_model, scaler=MaxAbsScaler(), feature_generator=None):

        self.models = models

        if scaler:

            if feature_generator:

                self.meta_model = make_pipeline(scaler, feature_generator, meta_model)

            else:

                self.meta_model = make_pipeline(scaler, meta_model)

        else:

            if feature_generator:

                self.meta_model = make_pipeline(feature_generator, meta_model)

            else:

                self.meta_model = meta_model

    def fit(self,X,y):

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):

            model.fit (X, y)

            predictions[:,i] = model.predict(X)

        self.meta_model.fit(predictions, y)

    def predict(self,X):

        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):

            predictions[:,i] = model.predict(X)

        return self.meta_model.predict(predictions)

    def __str__ (self):

        return "<CustomEnsemble (meta={}, models={})>".format(self.meta_model, self.models)

    def __repr__ (self):

        return self.__str__()



models = [

    # 0.09534141759933776

    BayesianRidge(),

    # 0.09594388448467231

    KernelRidge(),

    # 0.09624848214154685

    make_pipeline(MinMaxScaler(), PolynomialFeatures(2), lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.028000000000000004, n_estimators=700, n_jobs=-1, num_leaves=5))

]

my_combs = make_combinations(models)

print ("I chose {} combinations to test!".format(2*len(my_combs)))

    

# Testing every possible combination

# print ("Testing combinations...")

# i = 0

# results = []

# best = 10000

# with progressbar.ProgressBar(max_value=2*len(my_combs)) as bar:

#     for comb in my_combs:

#         X = chosen_X_train#.values

#         stack_model = CustomEnsemble(list(comb[1]), comb[0])

#         score = rmse_cv(stack_model, X, y_train_transformed).mean()

#         results.append(score)

#         #print ("Score: {}".format(score))

#         if (score < best):

#             print ("Score {:.4f} is better than previous best. Saving...".format(score))

#             best = score

#             best_model = stack_model

#         i+=1

#         bar.update(i)

#         stack_model = CustomEnsemble(list(comb[1]), comb[0], scaler=None)

#         score = rmse_cv(stack_model, X, y_train_transformed).mean()

#         results.append(score)

#         #print ("Score: {}".format(score))

#         if (score < best):

#             print ("Score {:.4f} is better than previous best. Saving...".format(score))

#             best = score

#             best_model = stack_model

#         i+=1

#         bar.update(i)



# === Best model was:

# <CustomEnsemble (meta=Pipeline(memory=None,

#          steps=[('maxabsscaler', MaxAbsScaler(copy=True)),

#                 ('kernelridge',

#                  KernelRidge(alpha=1, coef0=1, degree=3, gamma=None,

#                              kernel='linear', kernel_params=None))],

#          verbose=False), models=[Pipeline(memory=None,

#          steps=[('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1))),

#                 ('polynomialfeatures',

#                  PolynomialFeatures(degree=2, include_bias=True,

#                                     interaction_only=False, order='C')),

#                 ('lgbmregressor',

#                  LGBMRegressor(boosting_type='gbdt', class_weight=None,

#                                colsample_bytree=1.0, importance_type='split',

#                                learning_rate=0.028000000000000004, max_depth=-1,

#                                min_child_samples=20, min_child_weight=0.001,

#                                min_split_gain=0.0, n_estimators=700, n_jobs=-1,

#                                num_leaves=5, objective=None, random_state=None,

#                                reg_alpha=0.0, reg_lambda=0.0, silent=True,

#                                subsample=1.0, subsample_for_bin=200000,

#                                subsample_freq=0))],

#          verbose=False), BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,

#               fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,

#               normalize=False, tol=0.001, verbose=False)])>

best_model = CustomEnsemble(

    models = [

        make_pipeline(MinMaxScaler(), PolynomialFeatures(2), lgb.LGBMRegressor(boosting_type='gbdt', learning_rate=0.028000000000000004, n_estimators=700, n_jobs=-1, num_leaves=5)),

        BayesianRidge()

    ],

    meta_model = KernelRidge(),

    scaler = MaxAbsScaler()

)

best = 0.09228172938603223
print ("And the best model goes to...")

print (best_model)

print ("Its score was {}".format (best))
from scipy.special import inv_boxcox1p



best_model.fit(chosen_X_train, y_train_transformed)



sub = pd.DataFrame()

sub['Id'] = df_test['Id']

sub['SalePrice'] = inv_boxcox1p(best_model.predict(chosen_X_test), bclambda)

#sub['SalePrice'] = inv_boxcox1p(best_model.predict(chosen_X_test.values), bclambda)

sub.to_csv('submission.csv',index=False)