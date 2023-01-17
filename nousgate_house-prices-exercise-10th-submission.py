# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

# Set visualisation colours
mycols = ["#66c2ff", "#5cd6d6", "#00cc99", "#85e085", "#ffd966", "#ffb366", "#ffb3b3", "#dab3ff", "#c2c2d6"]
sns.set_palette(palette = mycols, n_colors = 4)
print('My colours are ready! :)')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Data reading

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("Train set size:", train.shape)
print("Test set size:", test.shape)
plt.scatter(train.GrLivArea, train.SalePrice)
train = train[train.GrLivArea < 4500]
plt.scatter(train.GrLivArea, train.SalePrice)
# Drop ID column
train = train.drop(['Id'], axis=1)
test = test.drop(['Id'], axis=1)
train.head()
print("Train set size:", train.shape)
print("Test set size:", test.shape)
plt.subplot(1, 2, 1)
sns.distplot(train.SalePrice, kde=False, fit = norm)

plt.subplot(1, 2, 2)
sns.distplot(np.log(train.SalePrice + 1), kde=False, fit = norm)
plt.xlabel('Log SalePrice')
train.SalePrice = np.log1p(train.SalePrice)
# Concatenation of train and test dataset
y = train.SalePrice.reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test
print(train_features.shape)
print(test_features.shape)
features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape
perc_na = (features.isnull().sum()/len(features))*100
ratio_na = perc_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :ratio_na})
missing_data.head(20)
# Let's investigate Pool features 
selectFeatures = ['PoolQC','PoolArea']
poolFeatures = features[selectFeatures]
poolFeaturesNulls = poolFeatures[poolFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
poolFeaturesNulls = poolFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
poolFeaturesNulls[(poolFeaturesNulls.isnull()).sum(axis=1) < 2]
# There are three NaN's foor PoolQC that have a PoolArea. Let's impute them based on overall quality of the house.
features.loc[2418, 'PoolQC'] = 'TA'
features.loc[2501, 'PoolQC'] = 'Gd'
features.loc[2597, 'PoolQC'] = 'Fa'
# Impute the other NaN values
features['PoolQC'] = features['PoolQC'].fillna('NA')
features['PoolArea'] = features['PoolArea'].fillna(0)
# Let's investigate Miscellaneous features 
selectFeatures = ['MiscFeature','MiscVal']
miscFeatures = features[selectFeatures]
miscFeaturesNulls = miscFeatures[miscFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
miscFeaturesNulls = miscFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
miscFeaturesNulls[(miscFeaturesNulls.isnull()).sum(axis=1) < 2]
# There are one NaN for MiscFeature that have a MiscVal. Let's impute it to a generic value Othr.
features.loc[2547, 'MiscFeature'] = 'Othr'
# Impute the other NaN values
features['MiscFeature'] = features['MiscFeature'].fillna('NA')
features['MiscVal'] = features['MiscVal'].fillna(0)
# Alley cannot be related to any other feature so let's substitute NaN with NA
features['Alley'] = features['Alley'].fillna('NA')
# Fence cannot be related to any other feature so let's substitute NaN with NA
features['Fence'] = features['Fence'].fillna('NA')
# Let's investigate Fireplace features 
selectFeatures = ['Fireplaces','FireplaceQu']
firepFeatures = features[selectFeatures]
firepFeaturesNulls = firepFeatures[firepFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
firepFeaturesNulls = firepFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
firepFeaturesNulls[(firepFeaturesNulls.isnull()).sum(axis=1) < 2]
# There is no incongruency so we can impute the other NaN values
features['FireplaceQu'] = features['FireplaceQu'].fillna('NA')
features['Fireplaces'] = features['Fireplaces'].fillna(0)
# Let's investigate LotFrontage related features 
selectFeatures = ['LotFrontage','Street','PavedDrive']
strFeatures = features[selectFeatures]
strFeaturesNulls = strFeatures[strFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
strFeaturesNulls = strFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 3 NA's, meaning there is incongruency in the row.
strFeaturesNulls[(strFeaturesNulls.isnull()).sum(axis=1) < 3]
# There are 486 with NaN values for LotFrontage and valid values for Street and PavedDrive.
# Let's impute LotFrontage using the most common value grouped by the Neighborhood
features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Impute the other NaN values
features['Street'] = features.groupby('Neighborhood')['Street'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['PavedDrive'] = features.groupby('Neighborhood')['PavedDrive'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Let's investigate the other Lot features 
selectFeatures = ['LotArea','LotShape','LotConfig','LandContour','LandSlope']
lotFeatures = features[selectFeatures]
lotFeaturesNulls = lotFeatures[lotFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
lotFeaturesNulls = lotFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 4 NA's, meaning there is incongruency in the row.
lotFeaturesNulls[(lotFeaturesNulls.isnull()).sum(axis=1) < 4]
# There is no incongruency so we can impute the other NaN values based on Neighborhood
features['LotArea'] = features.groupby('Neighborhood')['LotArea'].transform(lambda x: x.fillna(x.median()))
features['LotShape'] = features.groupby('Neighborhood')['LotShape'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['LandContour'] = features.groupby('Neighborhood')['LandContour'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['LotConfig'] = features.groupby('Neighborhood')['LotConfig'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['LandSlope'] = features.groupby('Neighborhood')['LandSlope'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Let's investigate Garage features 
selectFeatures = ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']
garageFeatures = features[selectFeatures]
garageFeaturesNulls = garageFeatures[garageFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
garageFeaturesNulls = garageFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 7 NA's, meaning there is incongruency in the row.
garageFeaturesNulls[(garageFeaturesNulls.isnull()).sum(axis=1) < 7]
# So we have only two incongruences for the Garage* fetures.
features.loc[[2124,2574]]
# Let's impute them manually

# FOR RECORD 2124
# Impute GarageYrBlt equal to YearBuilt
features.loc[2124, 'GarageYrBlt'] = features.loc[2122, 'YearBuilt']
# Impute GarageQual based on overall quality of the house
features.loc[2124, 'GarageQual'] = 'Gd'
# Impute GarageCond based on overall condition of the house
features.loc[2124, 'GarageCond'] = 'Ex'
# Impute GarageFinish based on overall quality of the house
features.loc[2124, 'GarageFinish'] = 'Fin'

# FOR RECORD 2574
# Impute GarageYrBlt equal to YearBuilt
features.loc[2574, 'GarageYrBlt'] = features.loc[2572, 'YearBuilt']
# Impute GarageQual based on overall quality of the house
features.loc[2574, 'GarageQual'] = 'TA'
# Impute GarageCond based on overall condition of the house
features.loc[2574, 'GarageCond'] = 'Gd'
# Impute GarageFinish based on overall quality of the house
features.loc[2574, 'GarageFinish'] = 'Fin'
# For GarageCars and GarageArea we have no hint so we impute the average value for the Neighborhood
features['GarageCars'] = features.groupby('Neighborhood')['GarageCars'].transform(lambda x: x.fillna(x.median()))
features['GarageArea'] = features.groupby('Neighborhood')['GarageArea'].transform(lambda x: x.fillna(x.median()))

# Check the results
features.loc[[2124,2574]]
# Impute the other NaN values
# For the moment I'll suppose a missing garage for all records that have missing values for all the garage features
# I'll try to impute them at the median and average value to see if it improves the final result
features['GarageType'] = features['GarageType'].fillna('NA')
features['GarageFinish'] = features['GarageFinish'].fillna('NA')
features['GarageQual'] = features['GarageQual'].fillna('NA')
features['GarageCond'] = features['GarageCond'].fillna('NA')
features['GarageYrBlt'] = features['GarageYrBlt'].fillna(0)
features['GarageCars'] = features['GarageCars'].fillna(0)
features['GarageArea'] = features['GarageArea'].fillna(0)
# Let's investigate Bsmt features 
selectFeatures = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2',
                  'BsmtUnfSF','TotalBsmtSF']
bsmtFeatures = features[selectFeatures]
bsmtFeaturesNulls = bsmtFeatures[bsmtFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
bsmtFeaturesNulls = bsmtFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 6 NA's, meaning there is incongruency in the row.
bsmtFeaturesNulls[(bsmtFeaturesNulls.isnull()).sum(axis=1) < 6]
# There are 9 records with incongruent values. Let's impute them manually.
features.loc[332, 'BsmtFinType2'] = 'NA'
features.loc[947, 'BsmtExposure'] = 'No'
features.loc[947, 'BsmtFinSF1'] = 0
features.loc[947, 'BsmtFinSF2'] = 0
features.loc[1485, 'BsmtExposure'] = 'No'
features.loc[1485, 'BsmtFinSF1'] = 0
features.loc[1485, 'BsmtFinSF2'] = 0
features.loc[2038, 'BsmtCond'] = 'TA'
features.loc[2038, 'BsmtUnfSF'] = 0
features.loc[2183, 'BsmtCond'] = 'Fa'
features.loc[2183, 'BsmtFinSF2'] = 0
features.loc[2215, 'BsmtQual'] = 'TA'
features.loc[2215, 'BsmtFinSF1'] = 0
features.loc[2215, 'BsmtFinSF2'] = 0
features.loc[2216, 'BsmtQual'] = 'TA'
features.loc[2216, 'BsmtFinSF1'] = 0
features.loc[2216, 'BsmtFinSF2'] = 0
features.loc[2346, 'BsmtExposure'] = 'No'
features.loc[2346, 'BsmtFinSF1'] = 0
features.loc[2346, 'BsmtFinSF2'] = 0
features.loc[2522, 'BsmtCond'] = 'Fa'
features.loc[2522, 'BsmtFinSF2'] = 0
# Impute the other NaN values
features['BsmtQual'] = features['BsmtQual'].fillna('NA')
features['BsmtCond'] = features['BsmtCond'].fillna('NA')
features['BsmtExposure'] = features['BsmtExposure'].fillna('NA')
features['BsmtFinType1'] = features['BsmtFinType1'].fillna('NA')
features['BsmtFinType2'] = features['BsmtFinType2'].fillna('NA')
features['BsmtFinSF1'] = features['BsmtFinSF1'].fillna(0)
features['BsmtFinSF2'] = features['BsmtFinSF2'].fillna(0)
features['BsmtUnfSF'] = features['BsmtUnfSF'].fillna(0)
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)
# Let's investigate Masonry features 
selectFeatures = ['MasVnrType','MasVnrArea']
masFeatures = features[selectFeatures]
masFeaturesNulls = masFeatures[masFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
masFeaturesNulls = masFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
masFeaturesNulls[(masFeaturesNulls.isnull()).sum(axis=1) < 2]
# There is one NaN foor MasVnrType that have a MasVnrArea. Let's impute them based on overall quality of the house.
features.loc[2608, 'MasVnrType'] = 'CBlock'
# Impute the other NaN values
features['MasVnrType'] = features['MasVnrType'].fillna('None')
features['MasVnrArea'] = features['MasVnrArea'].fillna(0)
# Let's investigate Sale zoning related features 
selectFeatures = ['MSSubClass','MSZoning','BldgType','HouseStyle','SaleType','SaleCondition','MoSold','YrSold']
zoneFeatures = features[selectFeatures]
zoneFeaturesNulls = zoneFeatures[zoneFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
zoneFeaturesNulls = zoneFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 8 NA's, meaning there is incongruency in the row.
zoneFeaturesNulls[(zoneFeaturesNulls.isnull()).sum(axis=1) < 8]
# There are 5 records with incongruent data in MSZoning and SaleType features.
# Let's impute them based on most common value for the neighbours.
features['MSZoning'] = features.groupby('Neighborhood')['MSZoning'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['SaleType'] = features.groupby('Neighborhood')['SaleType'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Impute the other NaN values based on most common value for the neighbours.
features['MSSubClass'] = features.groupby('Neighborhood')['MSSubClass'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['BldgType'] = features.groupby('Neighborhood')['BldgType'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['HouseStyle'] = features.groupby('Neighborhood')['HouseStyle'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['SaleCondition'] = features.groupby('Neighborhood')['SaleCondition'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['MoSold'] = features.groupby('Neighborhood')['MoSold'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['YrSold'] = features.groupby('Neighborhood')['YrSold'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Let's investigate Bath related features 
selectFeatures = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
bathFeatures = features[selectFeatures]
bathFeaturesNulls = bathFeatures[bathFeatures.isnull().any(axis=1)]
bathFeaturesNulls
# There are two NaN's for BsmtFullBath and BsmtHalfBath. Let's impute them manually.
features.loc[2118, 'BsmtFullBath'] = 0
features.loc[2118, 'BsmtHalfBath'] = 0
features.loc[2186, 'BsmtFullBath'] = 0
features.loc[2186, 'BsmtHalfBath'] = 0
# Let's investigate Porch features 
selectFeatures = ['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']
porchFeatures = features[selectFeatures]
porchFeaturesNulls = porchFeatures[porchFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
porchFeaturesNulls = porchFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 4 NA's, meaning there is incongruency in the row.
porchFeaturesNulls[(porchFeaturesNulls.isnull()).sum(axis=1) < 4]
# There are no incongruency and no NaN values, so check the type to see if some action is needed
porchFeatures.info()
# Let's investigate Exterior features 
selectFeatures = ['Exterior1st','Exterior2nd','ExterQual','ExterCond']
extFeatures = features[selectFeatures]
extFeaturesNulls = extFeatures[extFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
extFeaturesNulls = extFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 4 NA's, meaning there is incongruency in the row.
extFeaturesNulls[(extFeaturesNulls.isnull()).sum(axis=1) < 4]
# Impute the NaN values based on most common value for the neighbours.
features['Exterior1st'] = features.groupby('Neighborhood')['Exterior1st'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['Exterior2nd'] = features.groupby('Neighborhood')['Exterior2nd'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['ExterQual'] = features.groupby('Neighborhood')['ExterQual'].apply(lambda x:x.fillna(x.value_counts().index[0]))
features['ExterCond'] = features.groupby('Neighborhood')['ExterCond'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Let's investigate Floor releted features 
selectFeatures = ['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea']
floorFeatures = features[selectFeatures]
floorFeaturesNulls = floorFeatures[floorFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
floorFeaturesNulls = floorFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 4 NA's, meaning there is incongruency in the row.
floorFeaturesNulls[(floorFeaturesNulls.isnull()).sum(axis=1) < 4]
# There are no incongruency and no NaN values, so check the type to see if some action is needed
floorFeatures.info()
# Let's investigate Overall features 
selectFeatures = ['OverallQual','OverallCond']
ovallFeatures = features[selectFeatures]
ovallFeaturesNulls = ovallFeatures[ovallFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
ovallFeaturesNulls = ovallFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
ovallFeaturesNulls[(ovallFeaturesNulls.isnull()).sum(axis=1) < 2]
# There are no incongruency and no NaN values, so check the type to see if some action is needed
ovallFeatures.info()
# Let's investigate Heating features 
selectFeatures = ['Heating','HeatingQC']
heatFeatures = features[selectFeatures]
heatFeaturesNulls = heatFeatures[heatFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
heatFeaturesNulls = heatFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
heatFeaturesNulls[(heatFeaturesNulls.isnull()).sum(axis=1) < 2]
# There are no incongruency and no NaN values, so check the type to see if some action is needed
heatFeatures.info()
# Let's investigate YearBuilt features 
selectFeatures = ['YearBuilt','YearRemodAdd']
yearFeatures = features[selectFeatures]
yearFeaturesNulls = yearFeatures[yearFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
yearFeaturesNulls = yearFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
yearFeaturesNulls[(yearFeaturesNulls.isnull()).sum(axis=1) < 2]
# There are no incongruency and no NaN values, so check the type to see if some action is needed
yearFeatures.info()
# Let's investigate Roof features 
selectFeatures = ['RoofStyle','RoofMatl']
roofFeatures = features[selectFeatures]
roofFeaturesNulls = roofFeatures[roofFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
roofFeaturesNulls = roofFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
roofFeaturesNulls[(roofFeaturesNulls.isnull()).sum(axis=1) < 2]
# There are no incongruency and no NaN values, so check the type to see if some action is needed
roofFeatures.info()
# Let's investigate Kitchen features 
selectFeatures = ['KitchenAbvGr','KitchenQual']
kitcFeatures = features[selectFeatures]
kitcFeaturesNulls = kitcFeatures[kitcFeatures.isnull().any(axis=1)]
# we need to replace zero with nan in order to understand if there are incongruency in the data
kitcFeaturesNulls = kitcFeaturesNulls.replace({'0':np.nan, 0:np.nan})
# now select just the rows that have less then 2 NA's, meaning there is incongruency in the row.
kitcFeaturesNulls[(kitcFeaturesNulls.isnull()).sum(axis=1) < 2]
# There is 1 NaN for KitchenQual that have a KitchenAbvGr. Let's impute them based on overall quality of the house.
features.loc[1553, 'KitchenQual'] = 'TA'

# There are no other NaN values, so check the type to see if some action is needed
kitcFeatures.info()
# Functional: The documentation says that we should assume "Typ", so lets impute that.
features['Functional'] = features['Functional'].fillna('Typ')
# Utilities: The documentation doesn't give any information so 
# let's impute the most common value grouped by Neighborhood
features['Utilities'] = features.groupby('Neighborhood')['Utilities'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# Electrical: The documentation doesn't give any information but obviously every house has this so 
# let's impute the most common value grouped by Neighborhood
features['Electrical'] = features.groupby('Neighborhood')['Electrical'].apply(lambda x:x.fillna(x.value_counts().index[0]))
# let's check that we no longer have any missing values
perc_na = (features.isnull().sum()/len(features))*100
ratio_na = perc_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'missing_ratio' :ratio_na})
missing_data = missing_data.drop(missing_data[missing_data.missing_ratio == 0].index)
missing_data.head(5)
# Check max and min of each feature
features.describe().loc[['min','max']]
# There is a GarageYrBlt with a value of 2207
features[features['GarageYrBlt'] == 2207]
# This is probably a typo, let's correct it
features.loc[2590, 'GarageYrBlt'] = 2007
# create separate columns for area of each possible
# basement finish type
bsmt_fin_cols = ['BsmtGLQ','BsmtALQ','BsmtBLQ',
                 'BsmtRec','BsmtLwQ']

for col in bsmt_fin_cols:
    # initialise as columns of zeros
    features[col+'SF'] = 0

# fill remaining finish type columns
for row in features.index:
    fin1 = features.loc[row,'BsmtFinType1']
    if (fin1!='NA') and (fin1!='Unf'):
        # add area (SF) to appropriate column
        features.loc[row,'Bsmt'+fin1+'SF'] += features.loc[row,'BsmtFinSF1']
        
    fin2 = features.loc[row,'BsmtFinType2']
    if (fin2!='NA') and (fin2!='Unf'):
        features.loc[row,'Bsmt'+fin2+'SF'] += features.loc[row,'BsmtFinSF2']

# already have BsmtUnf column in dataset
bsmt_fin_cols.append('BsmtUnf')

# also create features representing the fraction of the basement that is each finish type
for col in bsmt_fin_cols:
    features[col+'Frac'] = features[col+'SF']/features['TotalBsmtSF']
    # replace any nans with zero (for properties without a basement)
    features[col+'Frac'].fillna(0,inplace=True)

# from https://www.kaggle.com/amitchoudhary/house-prices-advanced-regression-techniques/script-v6
# IR2 and IR3 don't appear that often, so just make a distinction
# between regular and irregular.
features["IsRegularLotShape"] = (features["LotShape"] == "Reg") * 1

# Most properties are level; bin the other possibilities together
# as "not level".
features["IsLandLevel"] = (features["LandContour"] == "Lvl") * 1

# Most land slopes are gentle; treat the others as "not gentle".
features["IsLandSlopeGentle"] = (features["LandSlope"] == "Gtl") * 1

# Most land slopes are gentle; treat the others as "not gentle".
features["IsLandSlopeGentle"] = (features["LandSlope"] == "Gtl") * 1

# Most properties use standard circuit breakers.
features["IsElectricalSBrkr"] = (features["Electrical"] == "SBrkr") * 1

# About 2/3rd have an attached garage.
features["IsGarageDetached"] = (features["GarageType"] == "Detchd") * 1

# Most have a paved drive. Treat dirt/gravel and partial pavement
# as "not paved".
features["IsPavedDrive"] = (features["PavedDrive"] == "Y") * 1

# The only interesting "misc. feature" is the presence of a shed.
features["HasShed"] = (features["MiscFeature"] == "Shed") * 1.

# If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
features["Remodeled"] = (features["YearRemodAdd"] != features["YearBuilt"]) * 1

# Did a remodeling happen in the year the house was sold?
features["RecentRemodel"] = (features["YearRemodAdd"] == features["YrSold"]) * 1

# Was this house sold in the year it was built?
features["VeryNewHouse"] = (features["YearBuilt"] == features["YrSold"]) * 1

# Has Central Air?
features["HasCentralAir"] = (features["CentralAir"] == "Y") * 1.

#very few properties with Pool or 3SsnPorch
#replace columns with binary indicator
features["HasMasVnr"] = (features["MasVnrArea"] == 0) * 1
features["HasWoodDeck"] = (features["WoodDeckSF"] == 0) * 1
features["HasOpenPorch"] = (features["OpenPorchSF"] == 0) * 1
features["HasEnclosedPorch"] = (features["EnclosedPorch"] == 0) * 1
features["Has3SsnPorch"] = (features["3SsnPorch"] == 0) * 1
features["HasPool"] = (features["PoolArea"] == 0) * 1
features["HasScreenPorch"] = (features["ScreenPorch"] == 0) * 1

#create basement features
features['LowQualFinFrac'] = features['LowQualFinSF']/features['GrLivArea']
features['1stFlrFrac'] = features['1stFlrSF']/features['GrLivArea']
features['2ndFlrFrac'] = features['2ndFlrSF']/features['GrLivArea']
features['TotalAreaSF'] = features['GrLivArea']+features['TotalBsmtSF']+features['GarageArea']+features['EnclosedPorch']+features['ScreenPorch']
features['LivingAreaSF'] = features['1stFlrSF'] + features['2ndFlrSF'] + features['BsmtGLQSF'] + features['BsmtALQSF'] + features['BsmtBLQSF']
features['StorageAreaSF'] = features['LowQualFinSF'] + features['BsmtRecSF'] + features['BsmtLwQSF'] + features['BsmtUnfSF'] + features['GarageArea']

# Months with the largest number of deals may be significant.
features["HighSeason"] = features["MoSold"].replace(
 {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

features["NewerDwelling"] = features["MSSubClass"].replace(
 {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
  90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

# shape        
print('Shape features: {}'.format(features.shape))
features.head()
# Encode some categorical features as ordered numbers when there is information in the order

features = features.replace({"Alley" : {"NA" : 0, "Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"NA" : 0, "No" : 1, "Mn" : 2, "Av": 3, "Gd" : 4},
                       "BsmtFinType1" : {"NA" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"NA" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},                             
                       "GarageFinish" : {"NA" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},   
                       "Heating" : {"Wall" : 1, "OthW" : 2, "Grav" : 3, "GasW" : 4, "GasA" : 5, "Floor" : 6},      
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "LandContour" : {"Low" : 1, "HLS" : 2, "Bnk" : 3, "Lvl" : 4},      
                       "RoofMatl" : {"WdShngl" : 1, "WdShake" : 2, "Tar&Grv" : 3, "Roll" : 4, 
                                     "Metal" : 5, "Membran" : 6, "CompShg" : 7, "ClyTile" : 8},
                       "Fence" : {"NA" : 1, "MnWw" : 1, "GdWo" : 2, "MnPrv" : 3, "GdPrv" : 4},   
                       "MasVnrType" : {"Stone" : 1, "None" : 2, "CBlock" : 3, "BrkFace" : 4, "BrkCmn" : 5}, 
                       "Foundation" : {"Wood" : 0, "Stone" : 1, "Slab" : 2, "PConc" : 3, "CBlock" : 4, "BrkTil" : 5},         
                       "Electrical" : {"Mix" : 0, "FuseP" : 1, "FuseF" : 2, "FuseA" : 3, "SBrkr" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"NA" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )
from sklearn.preprocessing import LabelEncoder

cols = ('MSSubClass','OverallCond','YrSold','MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    features[c] = lbl.transform(list(features[c].values))
features[['Alley','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual'
          ,'ExterCond','ExterQual','FireplaceQu','Functional','GarageCond','GarageQual','GarageFinish'
          ,'Heating','HeatingQC','KitchenQual','LandSlope','LotShape','LandContour','RoofMatl','Fence'
          ,'MasVnrType','Foundation','Electrical','PavedDrive','PoolQC','Street','Utilities']].astype('int')

features[['Alley','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual'
          ,'ExterCond','ExterQual','FireplaceQu','Functional','GarageCond','GarageQual','GarageFinish'
          ,'Heating','HeatingQC','KitchenQual','LandSlope','LotShape','LandContour','RoofMatl','Fence'
          ,'MasVnrType','Foundation','Electrical','PavedDrive','PoolQC','Street','Utilities']].head(10)
# remove initial BsmtFin columns
features.drop(['BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2'], axis=1, inplace=True)

# remove features for which we have created boolean columns
features.drop(['MasVnrArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','PoolQC','PoolArea','ScreenPorch','CentralAir'],axis=1,inplace=True)

# shape        
print('Shape features: {}'.format(features.shape))
features.head(10)
# check type of all categorical features to verify that only the columns
# for which we will create the dummie variables are remained
features.select_dtypes(['object']).head(10)
# Set Latitude and Longitude location based on the school location in the district    
features["Latitude"] = features.Neighborhood.replace({'Blmngtn' : 42.062806,
                                               'Blueste' : 42.009408,
                                                'BrDale' : 42.052500,
                                                'BrkSide': 42.033590,
                                                'ClearCr': 42.025425,
                                                'CollgCr': 42.021051,
                                                'Crawfor': 42.025949,
                                                'Edwards': 42.022800,
                                                'Gilbert': 42.027885,
                                                'GrnHill': 42.000854,
                                                'IDOTRR' : 42.019208,
                                                'Landmrk': 42.044777,
                                                'MeadowV': 41.991866,
                                                'Mitchel': 42.031307,
                                                'NAmes'  : 42.042966,
                                                'NoRidge': 42.050307,
                                                'NPkVill': 42.050207,
                                                'NridgHt': 42.060356,
                                                'NWAmes' : 42.051321,
                                                'OldTown': 42.028863,
                                                'SWISU'  : 42.017578,
                                                'Sawyer' : 42.033611,
                                                'SawyerW': 42.035540,
                                                'Somerst': 42.052191,
                                                'StoneBr': 42.060752,
                                                'Timber' : 41.998132,
                                                'Veenker': 42.040106})

features["Longitude"] = features.Neighborhood.replace({'Blmngtn' : -93.639963,
                                               'Blueste' : -93.645543,
                                                'BrDale' : -93.628821,
                                                'BrkSide': -93.627552,
                                                'ClearCr': -93.675741,
                                                'CollgCr': -93.685643,
                                                'Crawfor': -93.620215,
                                                'Edwards': -93.663040,
                                                'Gilbert': -93.615692,
                                                'GrnHill': -93.643377,
                                                'IDOTRR' : -93.623401,
                                                'Landmrk': -93.646239,
                                                'MeadowV': -93.602441,
                                                'Mitchel': -93.626967,
                                                'NAmes'  : -93.613556,
                                                'NoRidge': -93.656045,
                                                'NPkVill': -93.625827,
                                                'NridgHt': -93.657107,
                                                'NWAmes' : -93.633798,
                                                'OldTown': -93.615497,
                                                'SWISU'  : -93.651283,
                                                'Sawyer' : -93.669348,
                                                'SawyerW': -93.685131,
                                                'Somerst': -93.643479,
                                                'StoneBr': -93.628955,
                                                'Timber' : -93.648335,
                                                'Veenker': -93.657032})    
plt.scatter(features["Latitude"], features["Longitude"])
from sklearn.cluster import KMeans

# Compute the clusters
N_CLUSTERS = 5
kmeans = KMeans(init='k-means++', n_clusters=N_CLUSTERS, n_init=10)
selectFeatures = ['Latitude','Longitude']
kmeans.fit(features[selectFeatures])
features['Neighborhood_Cluster'] = kmeans.predict(features[selectFeatures])

features.drop(['Neighborhood'],axis=1,inplace=True)
features.head()
# fraction of zeros in each column
frac_zeros = ((features==0).sum()/len(features))

# no. unique values in each column
n_unique = features.nunique()

# difference between frac. zeros and expected
# frac. zeros if values evenly distributed between
# classes
xs_zeros = frac_zeros - 1/n_unique

# create dataframe and display which columns may be problematic
zero_cols = pd.DataFrame({'frac_zeros':frac_zeros,'n_unique':n_unique,'xs_zeros':xs_zeros})
zero_cols = zero_cols[zero_cols.frac_zeros>0]
zero_cols.sort_values(by='xs_zeros',ascending=False,inplace=True)
display(zero_cols[(zero_cols.xs_zeros>0)])

# 'half' bathrooms - add half value to 'full' bathrooms
features['BsmtFullBath'] = features['BsmtFullBath'] + 0.5*features['BsmtHalfBath']
features['FullBath'] = features['FullBath'] + 0.5*features['HalfBath']
features.drop(['BsmtHalfBath','HalfBath'],axis=1,inplace=True)

# create additional dummy variable for
# continuous variables with a lot of zeros
dummy_cols = ['LowQualFinSF','2ndFlrSF',
              'MiscVal','GarageArea','Fireplaces',             
              'BsmtGLQSF','BsmtALQSF','BsmtBLQSF','BsmtRecSF',
              'BsmtLwQSF','BsmtUnfSF','TotalBsmtSF']

for col in dummy_cols:
    features['Has'+col] = (features[col]>0).astype(int)

# shape        
print('Shape features: {}'.format(features.shape))
# Let's check skew in our features and transform if necessary.
from scipy.stats import skew

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics = []
for i in features.columns:
    if features[i].dtype in numeric_dtypes: 
        numerics.append(i)

skew_features = features[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews
# Let's use a boxcox trasformation for all the features with a skew > 0.5
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skew_features[skew_features > 0.5]
high_skew = high_skew
skew_index = high_skew.index

for i in skew_index:
    features[i]= boxcox1p(features[i], boxcox_normmax(features[i]+1))

        
skew_features = features[numerics].apply(lambda x: skew(x)).sort_values(ascending=False)
skews = pd.DataFrame({'skew':skew_features})
skews
dummies_features = pd.get_dummies(features).reset_index(drop=True)
dummies_features.shape
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=.02)
selector.fit(dummies_features) 

f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements

lowVarianceFeautures = dummies_features.columns[f(selector.get_support())]
print('{} variables have too low variance.'.format(len(lowVarianceFeautures)))
print('These variables are {}'.format(list(lowVarianceFeautures)))
# Try to drop the features with a very low variance
# Dropping low variance features result in a lower score so I comment the following line
dummies_features.drop(lowVarianceFeautures, axis=1, inplace=True)
print(dummies_features.shape)
print(y.shape)
# Now split again in train and test features
X = dummies_features.iloc[:len(y),:]
X_test = dummies_features.iloc[len(X):,:]

print(X.shape)
print(X_test.shape)
# function to detect outliers based on the predictions of a model
def find_outliers(model, X, y, sigma=3):

    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X,y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        
    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid)/std_resid    
    outliers = z[abs(z)>sigma].index
    
    # print and plot the results
    print('R2=',model.score(X,y))
    print('rmse=',rmse(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:',mean_resid)
    print('std of residuals:',std_resid)
    print('---------------------------------------')

    print(len(outliers),'outliers:')
    print(outliers.tolist())

    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y_pred');

    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('y')
    plt.ylabel('y - y_pred');

    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
    plt.legend(['Accepted','Outlier'])
    plt.xlabel('z')
    
    plt.savefig('outliers.png')
    
    return outliers

# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)   
    
    return np.sqrt(sum_sq/n)
from sklearn.linear_model import Ridge

# find and remove outliers using a Ridge model
outliers = find_outliers(Ridge(), X, y)

# permanently remove these outliers from the data
dummies_features = dummies_features.drop(outliers)
X_clean = X.drop(outliers)
y_clean = y.drop(outliers)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

#Build our model method
lm = LinearRegression()

#Build our cross validation method
kfolds = KFold(n_splits=10, shuffle=True, random_state=23)

#build our model scoring function
def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, 
                                   scoring="neg_mean_squared_error", 
                                   cv = kfolds))
    return(rmse)


#second scoring metric
def cv_rmsle(model):
    rmsle = np.sqrt(np.log(-cross_val_score(model, X, y,
                                           scoring = 'neg_mean_squared_error',
                                           cv=kfolds)))
    return(rmsle)
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# scorer to be used in sklearn model fitting
rmse_scorer = make_scorer(rmse, greater_is_better=False)

def train_model(model, param_grid=[], X=[], y=[], 
                splits=5, repeats=5):
    
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    
    # perform a grid search if param_grid given
    if len(param_grid)>0:
        # setup grid search parameters
        gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                               scoring=rmse_scorer,
                               verbose=1, return_train_score=True)

        # search the grid
        gsearch.fit(X,y)

        # extract best model from the grid
        model = gsearch.best_estimator_        
        best_idx = gsearch.best_index_

        # get cv-scores for best model
        grid_results = pd.DataFrame(gsearch.cv_results_)       
        cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
        cv_std = grid_results.loc[best_idx,'std_test_score']

    # no grid search, just cross-val score for given model    
    else:
        grid_results = []
        cv_results = cross_val_score(model, X, y, scoring=rmse_scorer, cv=rkfold)
        cv_mean = abs(np.mean(cv_results))
        cv_std = np.std(cv_results)
    
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)
    
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=',model.score(X,y))
    print('rmse=',rmse(y, y_pred))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    
    # residual plots
    y_pred = pd.Series(y_pred,index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid)/std_resid    
    n_outliers = sum(abs(z)>3)
    
    plt.figure(figsize=(15,5))
    ax_131 = plt.subplot(1,3,1)
    plt.plot(y,y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y_pred');
    plt.title('corr = {:.3f}'.format(np.corrcoef(y,y_pred)[0][1]))
    ax_132=plt.subplot(1,3,2)
    plt.plot(y,y-y_pred,'.')
    plt.xlabel('y')
    plt.ylabel('y - y_pred');
    plt.title('std resid = {:.3f}'.format(std_resid))
    
    ax_133=plt.subplot(1,3,3)
    z.plot.hist(bins=50,ax=ax_133)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results
# places to store optimal models and scores
opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])

# no. k-fold splits
splits=5
# no. k-fold iterations
repeats=10
model = 'Lasso'

opt_models[model] = Lasso()
alph_range = np.arange(1e-4,1e-3,4e-5)
param_grid = {'alpha': np.arange(1e-4,1e-3,4e-5)}

opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, 
                                            X=X_clean, y=y_clean, splits=splits, repeats=repeats)

cv_score.name = model
score_models = score_models.append(cv_score)

plt.figure()
plt.errorbar(alph_range, abs(grid_results['mean_test_score']),abs(grid_results['std_test_score'])/np.sqrt(splits*repeats))
plt.xlabel('alpha')
plt.ylabel('score');
# get coefficients from previously optimised Lasso model
en_coefs = pd.Series(opt_models['Lasso'].coef_,index=X.columns)

plt.figure(figsize=(8,20))
en_coefs[en_coefs.abs()>0.0002].sort_values().plot.barh()
plt.title('Coefficients with magnitude greater than 0.02')

print('---------------------------------------')
print(sum(en_coefs==0),'zero coefficients')
print(sum(en_coefs!=0),'non-zero coefficients')
print('---------------------------------------')
print('Intercept: ',opt_models['Lasso'].intercept_)
print('---------------------------------------')
print('Top 20 contributers to increased price:')
print('---------------------------------------')
print(en_coefs.sort_values(ascending=False).head(20))
print('---------------------------------------')
print('Top 10 contributers to decreased price:')
print('---------------------------------------')
print(en_coefs.sort_values(ascending=True).head(10))
print('---------------------------------------')
print('Zero coefficients:')
print('---------------------------------------')
print(en_coefs[en_coefs==0].index.sort_values().tolist())
from sklearn.preprocessing import PolynomialFeatures

X = X_clean
y = y_clean

#restrict to non-binary columns with non-zero coefficients only
select_cols = [col for col in X.columns if X[col].nunique()>2 and np.abs(en_coefs[col])>0.01]
X = X[select_cols]

# add interaction terms (x1*x2, but not x1**2) for all remaining features
poly = PolynomialFeatures(interaction_only=True)
X_poly = poly.fit_transform(X,y)
X_poly = pd.DataFrame(X_poly,index=y.index)

# save info on which features contribute to each term in X_poly
powers = pd.DataFrame(poly.powers_,columns=X.columns)

s = np.where(powers>0, pd.Series(X.columns)+', ', '')
poly_terms = pd.Series([''.join(x).strip() for x in s])

X_poly.shape
# fit a new model with the interaction terms
alph_range = np.arange(1e-4,1e-3,1e-4)
param_grid = {'alpha': np.arange(1e-4,1e-3,1e-4),
              'max_iter':[100000]}

model,_,_ = train_model(Lasso(), X=X_poly,y=y,param_grid=param_grid, 
                                              splits=splits, repeats=1)

poly_coefs = pd.Series(model.coef_)

print('------------------------')
print(sum(poly_coefs==0),'zero coefficients')
print(sum(poly_coefs!=0),'non-zero coefficients')
print(len(poly_coefs[(powers.sum(axis=1)==2) & (poly_coefs>0)]),'non-zero interaction terms.')
print('------------------------')
print('Features with largest coefficients:')
print('------------------------')
print(poly_terms[poly_coefs.abs().sort_values(ascending=False).index[:30]])
print('------------------------')
# function to normalise a column of values to lie between 0 and 1
def scale_minmax(col):
    return (col-col.min())/(col.max()-col.min())
# sort coefficients by magnitude, and calculate no. features that contribute
# to the polynomial term that coefficient represents.
poly_coef_nterms = powers.loc[poly_coefs.abs().sort_values(ascending=False).index].sum(axis=1)

# extract n_ints top interactions (interactions have 2 non-zero features in the powers vector)
n_ints = 30
top_n_int_idx = poly_coef_nterms[poly_coef_nterms==2].head(n_ints).index

# create a column for each of the top n_ints interactions in X
for idx in top_n_int_idx:
    # extract names of columns to multiply
    int_terms = powers.loc[idx]
    int_terms = int_terms[int_terms==1].index    
    term1 = int_terms[0]
    term2 = int_terms[1]
    
    # create interaction column
    
    dummies_features[term1+'_x_'+term2] = scale_minmax(dummies_features[term1]*dummies_features[term2])

# have a look at the new columns
display(dummies_features[dummies_features.columns[dummies_features.columns.str.contains('_x_')]].head(5))
dummies_features.shape
# Now split again in train and test features
y = y_clean
X = dummies_features.iloc[:len(y),:]
X_test = dummies_features.iloc[len(X):,:]

print(X.shape)
print(X_test.shape)
# fit a new model including the added interaction terms
model ='ElasticNet'
opt_models[model] = ElasticNet()

alph_range = np.arange(1e-4,1e-3,1e-4)
param_grid = {'alpha': np.arange(1e-4,1e-3,1e-4),
              'l1_ratio': np.arange(0.1,1.0,0.1),
              'max_iter':[100000]}

opt_models[model], cv_score, grid_results = train_model(opt_models[model], param_grid=param_grid, 
                                              X=X, y=y, splits=splits, repeats=repeats)

en_int_coef = pd.Series(opt_models[model].coef_,index=X.columns)
print('------------------------')
print(sum(en_int_coef==0),'zero coefficients')
print(sum(en_int_coef!=0),'non-zero coefficients')
print('--------------------------')
print('Interaction Coefficients:')
print('--------------------------')
print(en_int_coef[en_int_coef.index.str.contains('_x_')].sort_values(ascending=False))
print('---------------------------------------')
print('Top 10 contributers to increased price:')
print('---------------------------------------')
print(en_int_coef.sort_values(ascending=False).head(10))
print('---------------------------------------')
print('Top 5 contributers to decreased price:')
print('---------------------------------------')
print(en_int_coef.sort_values(ascending=True).head(5))
print('---------------------------------------')
submission = pd.read_csv("../input/sample_submission.csv")

y_test_pred = opt_models['ElasticNet'].predict(X_test)
submission.iloc[:,1] = np.expm1(y_test_pred)
submission.to_csv('10th_submission.csv', index=False)