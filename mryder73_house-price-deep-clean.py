import pandas as pd
train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)
#remove outliers

train = train[train['GrLivArea'] < 4000]
labels = train['SalePrice']

train = train.drop('SalePrice', axis=1)
all_data = pd.concat([train, test])
nums = all_data.select_dtypes(exclude = ['object']).columns

cat = all_data.select_dtypes(include = ['object']).columns
#find missing values in numeric data

for n in nums:

    if all_data[n].isnull().values.sum() > 0:

        print(n, all_data[n].isnull().sum())
#find missing values in categorical data

for c in cat:

    if all_data[c].isnull().values.sum() > 0:

        print(c, all_data[c].isnull().sum())
#MasVnrType null=24, MasVnrArea=23, where is the difference?

all_data[(all_data['MasVnrType'].isnull()) & (all_data['MasVnrArea'].notnull())][['MasVnrType', 'MasVnrArea']]
#Id 2611 MasVnrType=NaN, MasVnrArea=198

all_data['MasVnrType'].value_counts()
#MasVnrArea=198 so MasVnrType set to most common value that isn't None, i.e. BrkFace

#set other MasVnrType NaNs to 'None' and MasVnrArea to 0

all_data.loc[2611, 'MasVnrType'] = 'BrkFace'

all_data['MasVnrType'].fillna('None', inplace=True)

all_data['MasVnrArea'].fillna(0, inplace=True)
#BsmtFullBath & BsmtHalfBath have 2 NaNs, other numeric basement features have only one

bsmt_num_features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

all_data[all_data['BsmtFullBath'].isnull()][bsmt_num_features]
#Since both BsmtFullBath & BsmtHalfBath ids have 0 or NaN for TotalBsmtSF change both to show 0 for all numeric basement features

all_data.loc[[2121, 2189], bsmt_num_features] = 0
#BsmtCond has highest number of NaNs

#check discrepancy between BsmtCond NaNs and BsmtQual NaNs

bsmt_cat_features = ['BsmtCond', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

print(all_data[(all_data['BsmtCond'].isnull()) & (all_data['BsmtQual'].notnull())][bsmt_cat_features])

print(all_data[(all_data['BsmtCond'].notnull()) & (all_data['BsmtQual'].isnull())][bsmt_cat_features])
#check ids above against numeric basement features to see if it looks like there is a basement

all_data.loc[[2041, 2186, 2525, 2218, 2219]][bsmt_num_features]
#it looks like above have a basement, so set BsmtCond = BsmtQual & BsmtQual = BsmtCond

all_data.loc[[2041, 2186, 2525], 'BsmtCond'] = all_data.loc[[2041, 2186, 2525], 'BsmtQual']

all_data.loc[[2218, 2219], 'BsmtQual'] = all_data.loc[[2218, 2219], 'BsmtCond']
#BsmtExposure NaNs have higher number that other basement features

print(all_data[(all_data['BsmtExposure'].isnull()) & (all_data['BsmtCond'].notnull())][bsmt_cat_features])
#check ids above against numeric basement features to see if it looks like there is a basement

all_data.loc[[949, 1488, 2349]][bsmt_num_features]
#determine BsmtExposure mode where there is a basement

all_data[(all_data['BsmtExposure'].notnull()) & (all_data['BsmtExposure'] != 'No')]['BsmtExposure'].value_counts()
#set above ids to BsmtExposure mode: 'Av'

all_data.loc[[949, 1488, 2349], 'BsmtExposure'] = 'Av'
#BsmtFinType2 NaNs have higher number that other basement features

all_data[(all_data['BsmtFinType2'].isnull()) & (all_data['BsmtFinType1'].notnull())][bsmt_cat_features]
#check id above against numeric basement features to see if it looks like there is a basement

all_data.loc[333][bsmt_num_features]
#since there is a good chunk of unfinished area set value to 'Unf'

all_data.loc[333, 'BsmtFinType2'] = 'Unf'
#set remaining categorical basement feature NaNs to show no basement

for i in bsmt_cat_features:

    all_data[i] = all_data[i].fillna('NA')
#GarageType has fewer nulls that other Garage features

grg_num_features = ['GarageCars', 'GarageArea', 'GarageYrBlt']

grg_cat_features = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

all_data[(all_data['GarageYrBlt'].isnull()) & (all_data['GarageType'].notnull())][grg_cat_features+grg_num_features]
#ids above show as having a GarageType but missing other numeric or categorical garage features

#check other features that might show up garage features: MiscFeature & SaleCondition

print(all_data[all_data['MiscFeature'] == 'Gar2'].index)

print(all_data[all_data['SaleCondition'] == 'Alloca'].index)
#SaleCondition 'Alloca': '...typically condo with a garage unit', includes id2577

#Assuming there is a garage, impute median taken from similar properties for garage size features

all_data['GarageCars'] = all_data['GarageCars'].fillna(all_data[all_data['SaleCondition'] == 'Alloca']['GarageCars'].median())

all_data['GarageArea'] = all_data['GarageArea'].fillna(all_data[all_data['SaleCondition'] == 'Alloca']['GarageArea'].median())



#property build in 1923, unlikely to build garage at that date, however remod date is 1999 which seems more reasonable

#set GarageYrBuilt to YearRemodAdd

all_data.loc[2577, 'GarageYrBlt'] = all_data.loc[2577, 'YearRemodAdd']



#property OverallCond & OverallQual are average/above average, assume garage is finished & set to finished mode 'RFn'

#set GarageQual & GarageCond to reflect property condition & quality: 'TA'

all_data.loc[2577, 'GarageFinish'] = 'Rfn'

all_data.loc[2577, ['GarageQual', 'GarageCond']] = 'TA'
#id2127 not in MiscFeature or SaleCondition.

#property build in 1910, unlikely to build garage at that date, however remod date is 1983 which seems more reasonable

all_data.loc[2127, 'GarageYrBlt'] = all_data.loc[2127, 'YearRemodAdd']



#property OverallCond & OverallQual are very good/above average, assume garage is finished & set to finished mode 'Fin'

#set GarageQual & GarageCond to reflect property condition & quality: 'Gd' & 'TA'

all_data.loc[2127, 'GarageFinish'] = 'Fin'

all_data.loc[2127, 'GarageCond'] = 'Gd'

all_data.loc[2127, 'GarageQual'] = 'TA'
#set remaining numerical garage feature NaNs to show no garage

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)



#set remaining categorical garage feature NaNs to show no garage

for i in grg_cat_features:

    all_data[i] = all_data[i].fillna('NA')
#find MSZoning NaNs plus some other features potentially helpful for imputing missing data

all_data[all_data['MSZoning'].isnull()][['MSZoning', 'MSSubClass', 'Neighborhood', 'BldgType', 'HouseStyle']]
#Check potential values for MSZoning based on MSSubClass, Neighborhood, BldgType, HouseStyle

#id1916

print('id1916')

print(all_data[(all_data['MSSubClass'] == 30) & 

               (all_data['Neighborhood'] == 'IDOTRR') & 

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == '1Story')]

      ['MSZoning'].value_counts())



#id2217

print('\nid2217')

print(all_data[(all_data['MSSubClass'] == 20) & 

               (all_data['Neighborhood'] == 'IDOTRR') & 

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == '1Story')]

      ['MSZoning'].value_counts())



#id2251

#all 4 features=zero response

#remove 'HouseStyle' = RM:8, C(all):3

#remove 'BldgType = zero response

#remove 'Neighborhood = RL:2

#remove 'MSSubClass' = RM:1

#Result RM:9, C(all):3, RL:1

print('\nid2251')

print(all_data[(all_data['MSSubClass'] == 70) & 

               (all_data['Neighborhood'] == 'IDOTRR') &

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == '2.5Unf')]

      ['MSZoning'].value_counts())



#id2905

print('\nid2905')

print(all_data[(all_data['MSSubClass'] == 20) & 

               (all_data['Neighborhood'] == 'Mitchel') & 

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == '1Story')]

      ['MSZoning'].value_counts())
#results:

#set id1916 MSZoning = 'RM'

#set id2217 MSZoning = 'C (all)'

#set id2251 MSZoning = 'RM'

#set id2905 MSZoning = 'RL'

all_data.loc[1916, 'MSZoning'] = 'RM'

all_data.loc[2217, 'MSZoning'] = 'C (all)'

all_data.loc[2251, 'MSZoning'] = 'RM'

all_data.loc[2905, 'MSZoning'] = 'RL'
#nothing interesting here, set to 'NA' to represent no Alley

all_data['Alley'] = all_data['Alley'].fillna('NA')
#nothing interesting here, 2 NaNs (ids 1916 & 1946), AllPub=2916, NoSeWa=1

#set NaNs to AllPub

all_data['Utilities'] = all_data['Utilities'].fillna('AllPub')
all_data[all_data['Utilities'].isnull()]['Utilities']
#get a better idea of what the missing data might be from a few other potentially useful features

all_data[all_data['Exterior1st'].isnull()][['Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

                                            'MSSubClass', 'Neighborhood', 'BldgType', 'HouseStyle']]
#Check a range of features to find suitable value for missing Exterior1st & Exterior2nd

print(all_data[(all_data['MasVnrType'] == 'None') & 

               (all_data['MSSubClass'] == 30) & 

               (all_data['Neighborhood'] == 'Edwards') & 

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == '1Story')]

      ['Exterior1st'].value_counts(), '\n')



print(all_data[(all_data['MasVnrType'] == 'None') & 

               (all_data['MSSubClass'] == 30) & 

               (all_data['Neighborhood'] == 'Edwards') & 

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == '1Story')]

      ['Exterior2nd'].value_counts())



#set both Exterior1st & Exterior2nd to 'Wd Sdng'

all_data.loc[2152, ['Exterior1st', 'Exterior2nd']] = 'Wd Sdng'
#little of interest here, only 1 NaN, mode across dataset and in similar properties is SBrkr

print(all_data['Electrical'].value_counts(), '\n')



print(all_data[(all_data['MSSubClass'] == 80) & 

               (all_data['Neighborhood'] == 'Timber') & 

               (all_data['BldgType'] == '1Fam') & 

               (all_data['HouseStyle'] == 'SLvl')]

      ['Electrical'].value_counts())



#set missing value to SBrkr

all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr')
#KitchenQual has a single NaN, other features suggest that the house isn't in top condition

print(all_data[all_data['KitchenQual'].isnull()][['KitchenQual', 'Functional', 'OverallQual', 'OverallCond']])



#impute missing KitchenQual value as lower than average to reflect this

all_data.loc[1556, 'KitchenQual'] = 'Fa'
all_data[all_data['Functional'].isnull()][['Functional', 'OverallQual', 'OverallCond', 'KitchenQual',

                                          'BsmtQual', 'BsmtCond', 'HeatingQC', 'GarageQual', 'GarageCond', 'ExterQual',

                                          'ExterCond']]
#based on above:

#Functional has an 8 point scale, OverallQual & OverallCond have 10 point scales, the others have 5 point scales

# id2217 = (1+5+2+1+2+1+2+1)/50 = 0.3, roughly 20% below average, set Functional to Moderate Deductions

# id2474 = (4+1+3+3+2+2+3+2+2+2)/60 = 0.4 roughly 10% below average, set Functional to Minor Deductions2

all_data.loc[2217, 'Functional'] = 'Mod'

all_data.loc[2474, 'Functional'] = 'Min2'
#not much of interest here, all missing values for FireplaceQu have zero for Fireplaces

print(all_data[(all_data['FireplaceQu'].isnull()) & (all_data['Fireplaces'] != 0)][['FireplaceQu', 'Fireplaces']])



#set missing values for FireplaceQu to NA to reflect no fireplace

all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('NA')
#3 NaNs for PoolQC & PoolArea greater than zero

print(all_data[(all_data['PoolQC'].isnull()) & (all_data['PoolArea'] != 0)][['PoolQC', 'PoolArea', 

                                                                             'OverallQual', 'OverallCond', 'Functional']])



#based on OverallQual, OverallCond and Functional set first 2 to TA, 3rd to Fa

all_data.loc[[2421, 2504], 'PoolQC'] = 'TA'

all_data.loc[2600, 'PoolQC'] = 'Fa'



#set all other missing PoolQC values to NA

all_data['PoolQC'] = all_data['PoolQC'].fillna('NA')
#not much can be done with Fence, set missing values to NA

all_data['Fence'] = all_data['Fence'].fillna('NA')
# 4 sheds have low values, 2 being zero. One MiscFeature is Other and Zero

print(all_data[(all_data['MiscFeature'].notnull()) & (all_data['MiscVal'] < 100)][['MiscFeature', 'MiscVal']])



#set low shed values to median shed value, and change "other" to no MiscFeature

all_data.loc[[813, 1201, 1606, 2432], 'MiscVal'] = all_data[all_data['MiscFeature'] == 'Shed']['MiscVal'].median()

all_data.loc[874, 'MiscFeature'] = 'NA'



#set remaining missing values to NA

all_data['MiscFeature'] = all_data['MiscFeature'].fillna('NA')
#1 missing value for SaleType, not a new property, SaleCondition=Normal

print(all_data[all_data['SaleType'].isnull()][['SaleCondition', 'SaleType', 'YrSold', 'YearBuilt']])



#set missing value to mode for SaleType

all_data['SaleType'] = all_data['SaleType'].fillna('WD')
#just from eyeballing the data for a few neighbourhoods, and potentially useful features,

#HouseStyle, LotConfig & BldgType don't seem to impact LotFrontage

#LotArea and Neighborhood look faintly correlated with LotFrontage

#for each neighborhood:

#for each LotFrontage NaN, find lowest difference between LotArea and the LotArea of a known LotFrontage

#impute the Lotfrontage from the LotArea with the lowest difference



for n in all_data[all_data['LotFrontage'].isnull()]['Neighborhood'].unique():

    known = all_data[(all_data['LotFrontage'].notnull()) & (all_data['Neighborhood'] == n)]

    unknown = all_data[(all_data['LotFrontage'].isnull()) & (all_data['Neighborhood'] == n)]

    #print('started: ', n)



    for i in unknown.index:

        old_val = 100000

        new_LF = None

        for j in known.index:

            new_val = min(abs(unknown.loc[i]['LotArea'] - known.loc[j]['LotArea']), old_val) 

            if new_val < old_val:

                old_val, new_val = new_val, old_val

                new_LF = known.loc[j]['LotFrontage']

        all_data.loc[i, 'LotFrontage'] = new_LF

    #print('completed: ', n)



#apologies for the slow code :)
train = all_data[:train.shape[0]]

test = all_data[train.shape[0]:]



train['SalePrice'] = labels



train.to_csv('train_clean.csv')

test.to_csv('test_clean.csv')