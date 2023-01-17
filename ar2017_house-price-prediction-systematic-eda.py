# Import data analysis packages

import numpy as np

import pandas as pd



# Import data visualization packages

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')



# Import stats packages

from scipy.stats import kurtosis, skew, pearsonr



# Miscellaneous

import time

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_colwidth',80)
# Import datasets

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Show the first five rows of the training dataset

df_train.head()
# Set the 'Id' column as the index of the datasets

df_train.set_index('Id', inplace=True)

df_test.set_index('Id', inplace=True)
def create_boxen_count_2x1(first_feature, figsize):

    plt.figure(figsize=figsize)



    # Create boxen plot of first_feature and Log_SalePrice

    ax2 = plt.subplot(212)

    sns.boxenplot(x=first_feature, y='Log_SalePrice', data=df_train, color='tomato')

    plt.xticks(rotation='horizontal')



    # Create countplot of first_feature

    ax1 = plt.subplot(211, sharex=ax2)

    sns.countplot(x=first_feature, data=df_train, color='dimgrey')

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.xlabel('')



    # Adjusting the spaces between graphs

    plt.subplots_adjust(hspace = 0)



    plt.show()
def create_boxen_count_2x2(first_feature, second_feature, figsize):

    plt.figure(figsize=figsize)



    # Create boxen plot of first_feature and Log_SalePrice

    ax3 = plt.subplot(223)

    sns.boxenplot(x=first_feature, y='Log_SalePrice', data=df_train, color='dimgrey')



    # Create boxen plot of second_feature and Log_SalePrice

    ax4 = plt.subplot(224, sharey=ax3)

    sns.boxenplot(x=second_feature, y='Log_SalePrice', data=df_train, color='tomato')

    plt.setp(ax4.get_yticklabels(), visible=False)

    plt.ylabel('')



    #---------------------------------------------------------------------------------------



    # Create countplot of Condition1

    ax1 = plt.subplot(221, sharex=ax3)

    sns.countplot(x=first_feature, data=df_train, color='dimgrey')

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.xlabel('')



    # Create countplot of Condition2

    ax2 = plt.subplot(222, sharey=ax1, sharex=ax4)

    sns.countplot(x=second_feature, data=df_train, color='tomato')

    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.ylabel('')

    plt.xlabel('')



    # Adjusting the spaces between graphs

    plt.subplots_adjust(wspace=0, hspace=0)



    plt.show()
def create_boxen_count_2x3(first_feature, second_feature, third_feature, figsize):

    plt.figure(figsize=figsize)



    # Create boxenplot of first_feature and Log_SalePrice

    ax4 = plt.subplot(234)

    sns.boxenplot(x=first_feature, y='Log_SalePrice', data=df_train, color='dimgrey')



    # Create boxenplot of second_feature and Log_SalePrice

    ax5 = plt.subplot(235, sharey=ax4)

    sns.boxenplot(x=second_feature, y='Log_SalePrice', data=df_train, color='tomato')

    plt.setp(ax5.get_yticklabels(), visible=False)

    plt.ylabel('')



    # Create boxenplot of third_feature and Log_SalePrice

    ax6 = plt.subplot(236, sharey=ax4)

    sns.boxenplot(x=third_feature, y='Log_SalePrice', data=df_train, color='darkseagreen')

    plt.setp(ax6.get_yticklabels(), visible=False)

    plt.ylabel('')



    #---------------------------------------------------------------------------------------



    # Create countplot of first_feature

    ax1 = plt.subplot(231, sharex=ax4)

    sns.countplot(x=first_feature, data=df_train, color='dimgrey')

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.xlabel('')



    # Create countplot of second_feature

    ax2 = plt.subplot(232, sharey=ax1, sharex=ax5)

    sns.countplot(x=second_feature, data=df_train, color='tomato')

    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.ylabel('')

    plt.xlabel('')



    # Create countplot of second_feature

    ax3 = plt.subplot(233, sharey=ax1, sharex=ax6)

    sns.countplot(x=third_feature, data=df_train, color='darkseagreen')

    plt.setp(ax3.get_yticklabels(), visible=False)

    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.ylabel('')

    plt.xlabel('')



    # Adjusting the spaces between graphs

    plt.subplots_adjust(wspace=0, hspace=0)



    plt.show()
def create_boxen_count_2x4(first_feature, second_feature, third_feature, fourth_feature, figsize):

    plt.figure(figsize=figsize)



    # Create boxenplot of first_feature and Log_SalePrice

    ax5 = plt.subplot(245)

    sns.boxenplot(x=first_feature, y='Log_SalePrice', data=df_train, color='dimgrey')



    # Create boxenplot of second_feature and Log_SalePrice

    ax6 = plt.subplot(246, sharey=ax5)

    sns.boxenplot(x=second_feature, y='Log_SalePrice', data=df_train, color='tomato')

    plt.setp(ax6.get_yticklabels(), visible=False)

    plt.ylabel('')



    # Create boxenplot of third_feature and Log_SalePrice

    ax7 = plt.subplot(247, sharey=ax5)

    sns.boxenplot(x=third_feature, y='Log_SalePrice', data=df_train, color='darkseagreen')

    plt.setp(ax7.get_yticklabels(), visible=False)

    plt.ylabel('')



    # Create boxenplot of fourth_feature and Log_SalePrice

    ax8 = plt.subplot(248, sharey=ax5)

    sns.boxenplot(x=fourth_feature, y='Log_SalePrice', data=df_train, color='seagreen')

    plt.setp(ax8.get_yticklabels(), visible=False)

    plt.ylabel('')



    #---------------------------------------------------------------------------------------



    # Create countplot of first_feature

    ax1 = plt.subplot(241, sharex=ax5)

    sns.countplot(x=first_feature, data=df_train, color='dimgrey')

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.xlabel('')



    # Create countplot of second_feature

    ax2 = plt.subplot(242, sharey=ax1, sharex=ax6)

    sns.countplot(x=second_feature, data=df_train, color='tomato')

    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.ylabel('')

    plt.xlabel('')



    # Create countplot of third_feature

    ax3 = plt.subplot(243, sharey=ax1, sharex=ax7)

    sns.countplot(x=third_feature, data=df_train, color='darkseagreen')

    plt.setp(ax3.get_yticklabels(), visible=False)

    plt.setp(ax3.get_xticklabels(), visible=False)

    plt.ylabel('')

    plt.xlabel('')



    # Create countplot of fourth_feature

    ax4 = plt.subplot(244, sharey=ax1, sharex=ax8)

    sns.countplot(x=fourth_feature, data=df_train, color='seagreen')

    plt.setp(ax4.get_yticklabels(), visible=False)

    plt.setp(ax4.get_xticklabels(), visible=False)

    plt.ylabel('')

    plt.xlabel('')



    # Adjusting the spaces between graphs

    plt.subplots_adjust(wspace=0, hspace=0)



    plt.show()
# Create distribution plot of SalePrice feature

plt.figure(figsize=(14,5))

sns.distplot(df_train['SalePrice'], bins=150, color='g')

plt.show()
# Calculate the skewness and kurtosis of SalePrice distribution

print('Skewness of the distribution of SalePrice: {}'.format(skew(df_train['SalePrice'])))

print('Kurtosis of the distribution of SalePrice: {}'.format(kurtosis(df_train['SalePrice'])))
# Log transform the SalePrice feature

df_train['Log_SalePrice'] = np.log(df_train['SalePrice'])
# Create a distribution plot of SalePrice feature after log transformation

plt.figure(figsize=(14,5))

sns.distplot(df_train['Log_SalePrice'], bins=150, color='r')

plt.show()
# Calculate the skewness and kurtosis of SalePrice distribution after log transformation

print('Skewness of the distribution of SalePrice after log transformation: {}'.format(skew(df_train['Log_SalePrice'])))

print('Kurtosis of the distribution of SalePrice after log transformation: {}'.format(kurtosis(df_train['Log_SalePrice'])))
# Transform MSZoning into a numerical feature 

MSZoning_map = {'FV':1, 'RL':2, 'RM':3, 'RH':4, 'C (all)':5}

df_train['MSZoning'].replace(MSZoning_map, inplace=True)

df_test['MSZoning'].replace(MSZoning_map, inplace=True)
create_boxen_count_2x1('MSZoning', (7,8))
create_boxen_count_2x1('Neighborhood', (20,8))
create_boxen_count_2x2('Condition1', 'Condition2', (13,8))
surround_feats = df_train[['MSZoning', 'Log_SalePrice']]
plt.figure(figsize=(2,2))



sns.heatmap(surround_feats.corr(), annot=True, cmap='RdBu')



plt.show()
plt.figure(figsize=(14,10))



# Create distribution plot of LotFrontage

plt.subplot(211)

sns.distplot(df_train['LotFrontage'].dropna(), bins=100, color='r')



# Create distribution plot of LotArea

plt.subplot(212)

sns.distplot(df_train['LotArea'].dropna(), bins=100, color='g')



# Adjusting the spaces between graphs

plt.subplots_adjust(hspace = 0.2)



plt.show()
# Calculate the skewness and kurtosis of LotFrontage and LotArea distribution

print('Skewness of the distribution of LotFrontage: {}'.format(skew(df_train['LotFrontage'].dropna())))

print('Kurtosis of the distribution of LotFrontage: {}'.format(kurtosis(df_train['LotFrontage'].dropna())))

print('')

print('Skewness of the distribution of LotArea: {}'.format(skew(df_train['LotArea'].dropna())))

print('Kurtosis of the distribution of LotArea: {}'.format(kurtosis(df_train['LotArea'].dropna())))
# Log transform LotFrontage and LotArea

df_train['LotFrontage'] = np.log(df_train['LotFrontage'])

df_train['LotArea'] = np.log(df_train['LotArea'])



df_test['LotFrontage'] = np.log(df_test['LotFrontage'])

df_test['LotArea'] = np.log(df_test['LotArea'])
# Calculate the skewness and kurtosis of LotFrontage and LotArea distribution after log transformation

print('Skewness of the distribution of LotFrontage after log transformation: {}'.format(skew(df_train['LotFrontage'].dropna())))

print('Kurtosis of the distribution of LotFrontage after log transformation: {}'.format(kurtosis(df_train['LotFrontage'].dropna())))

print('')

print('Skewness of the distribution of LotArea after log transformation: {}'.format(skew(df_train['LotArea'].dropna())))

print('Kurtosis of the distribution of LotArea after log transformation: {}'.format(kurtosis(df_train['LotArea'].dropna())))
plt.figure(figsize=(10,8))



ax1 = plt.subplot(121)

sns.regplot(x='LotFrontage', y='Log_SalePrice', data=df_train, color='lightcoral')



ax2 = plt.subplot(122, sharey=ax1)

sns.regplot(x='LotArea', y='Log_SalePrice', data=df_train, color='darkseagreen')

plt.setp(ax2.get_yticklabels(), visible=False)

plt.ylabel('')



# Adjusting the spaces between graphs

plt.subplots_adjust(wspace = 0)



plt.show()
# Fill NaN with 'No alley access'

df_train['Alley'].fillna('No alley access', inplace=True)

df_test['Alley'].fillna('No alley access', inplace=True)
create_boxen_count_2x2('Street', 'Alley', (10,10))
# Transform LotShape into a numerical feature 

LotShape_map = {'Reg':1, 'IR1':2, 'IR2':3, 'IR3':4}

df_train['LotShape'].replace(LotShape_map, inplace=True)

df_test['LotShape'].replace(LotShape_map, inplace=True)



# Transform LandContour into a numerical feature 

LandContour_map = {'Lvl':0, 'Bnk':1, 'Low':1, 'HLS':1}

df_train['LandContour'].replace(LandContour_map, inplace=True)

df_test['LandContour'].replace(LandContour_map, inplace=True)



# Transform LotConfig into a numerical feature 

LotConfig_map = {'Inside':0, 'FR2':1, 'Corner':1, 'CulDSac':1, 'FR3':1}

df_train['LotConfig'].replace(LotConfig_map, inplace=True)

df_test['LotConfig'].replace(LotConfig_map, inplace=True)



# Transform LandSlope into a numerical feature 

LandSlope_map = {'Gtl':0, 'Mod':1, 'Sev':2}

df_train['LandSlope'].replace(LandSlope_map, inplace=True)

df_test['LandSlope'].replace(LandSlope_map, inplace=True)
create_boxen_count_2x4('LotShape', 'LandContour', 'LotConfig', 'LandSlope', (20,12))
lot_feats = df_train[['LotFrontage', 'LotArea', 'Street', 'Alley', 

                      'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 

                      'Log_SalePrice']]
plt.figure(figsize=(10,8))



sns.heatmap(lot_feats.corr(), annot=True, cmap='RdBu')



plt.show()
create_boxen_count_2x3('MSSubClass', 'BldgType', 'HouseStyle', (16,8))
# Transform Functional into a numerical feature 

Functional_map = {'Sal':0, 'Sev':1, 'Maj2':2, 'Maj1':3, 'Mod':4, 'Min2':5, 'Min1':6, 'Typ':7}

df_train['Functional'].replace(Functional_map, inplace=True)

df_test['Functional'].replace(Functional_map, inplace=True)
create_boxen_count_2x3('OverallQual', 'OverallCond', 'Functional', (16,8))
# Create TotalAge feature

df_train['TotalAge'] = df_train['YrSold'] - df_train['YearBuilt']

df_test['TotalAge'] = df_test['YrSold'] - df_test['YearBuilt']



# Create YrSinceRemod feature

df_train['YrSinceRemod'] = df_train['YrSold'] - df_train['YearRemodAdd']

df_test['YrSinceRemod'] = df_test['YrSold'] - df_test['YearRemodAdd']
plt.figure(figsize=(14,8))



ax1 = plt.subplot(121)

sns.regplot(x='TotalAge', y='Log_SalePrice', data=df_train, color='lightcoral')



ax2 = plt.subplot(122, sharey=ax1)

sns.regplot(x='YrSinceRemod', y='Log_SalePrice', data=df_train, color='darkseagreen')

plt.setp(ax2.get_yticklabels(), visible=False)

plt.ylabel('')



# Adjusting the spaces between graphs

plt.subplots_adjust(wspace = 0)



plt.show()
overall_feats = df_train[['OverallQual', 'OverallCond', 'Functional', 

                          'TotalAge', 'YrSinceRemod', 'Log_SalePrice']]
plt.figure(figsize=(10,8))



sns.heatmap(overall_feats.corr(), annot=True, cmap='RdBu')



plt.show()
create_boxen_count_2x2('RoofStyle', 'RoofMatl', (15,9))
create_boxen_count_2x2('Exterior1st', 'Exterior2nd', (15,9))
create_boxen_count_2x1('MasVnrType', (8,8))
# Transform MasVnrType into a numerical feature 

MasVnrType_map = {'None':0, 'BrkFace':1, 'Stone':1, 'BrkCmn':1}

df_train['MasVnrType'].replace(MasVnrType_map, inplace=True)

df_test['MasVnrType'].replace(MasVnrType_map, inplace=True)
create_boxen_count_2x1('MasVnrType', (8,8))
plt.figure()



sns.jointplot(x='MasVnrArea', 

              y='Log_SalePrice', 

              data=df_train, 

              kind='reg', 

              height=9,

              color='darkseagreen')



plt.show()
ExterQual_map = {'Po':1 ,'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}



df_train['ExterQual'].replace(ExterQual_map, inplace=True)

df_test['ExterQual'].replace(ExterQual_map, inplace=True)



df_train['ExterCond'].replace(ExterQual_map, inplace=True)

df_test['ExterCond'].replace(ExterQual_map, inplace=True)
create_boxen_count_2x2('ExterQual', 'ExterCond', (10,9))
create_boxen_count_2x1('Foundation', (8,8))
ext_feats = df_train[['MasVnrType', 'MasVnrArea', 'ExterQual', 

                      'ExterCond', 'Log_SalePrice']]
plt.figure(figsize=(10,8))



sns.heatmap(ext_feats.corr(), annot=True, cmap='RdBu')



plt.show()
BsmtQualCond_map = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}



df_train['BsmtQual'].replace(BsmtQualCond_map, inplace=True)

df_test['BsmtQual'].replace(BsmtQualCond_map, inplace=True)



df_train['BsmtCond'].replace(BsmtQualCond_map, inplace=True)

df_test['BsmtCond'].replace(BsmtQualCond_map, inplace=True)



BsmtExposure_map = {'No':0 ,'Mn':1, 'Av':2, 'Gd':3}

df_train['BsmtExposure'].replace(BsmtExposure_map, inplace=True)

df_test['BsmtExposure'].replace(BsmtExposure_map, inplace=True)
df_train['BsmtQual'].fillna(0, inplace=True)

df_test['BsmtQual'].fillna(0, inplace=True)



df_train['BsmtCond'].fillna(0, inplace=True)

df_test['BsmtCond'].fillna(0, inplace=True)



df_train['BsmtExposure'].fillna(0, inplace=True)

df_test['BsmtExposure'].fillna(0, inplace=True)
create_boxen_count_2x3('BsmtQual', 'BsmtCond', 'BsmtExposure', (16,10))
BsmtFinType_map = {'Unf':0, 'LwQ':1, 'Rec':2, 'BLQ':3, 'ALQ':4, 'GLQ':5}



df_train['BsmtFinType1'].replace(BsmtFinType_map, inplace=True)

df_test['BsmtFinType1'].replace(BsmtFinType_map, inplace=True)



df_train['BsmtFinType2'].replace(BsmtFinType_map, inplace=True)

df_test['BsmtFinType2'].replace(BsmtFinType_map, inplace=True)
create_boxen_count_2x2('BsmtFinType1', 'BsmtFinType2', (14,10))
plt.figure(figsize=(20,15))



g = sns.pairplot(df_train[['BsmtFinSF1', 'BsmtFinSF2', 

                           'BsmtUnfSF', 'TotalBsmtSF', 'Log_SalePrice']], 

             palette='Accent',

             kind='scatter',

             diag_kind='auto',

             height=3)



plt.show()
Bsmt_feats = df_train[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

                       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 

                       'BsmtUnfSF', 'TotalBsmtSF', 'Log_SalePrice']].dropna()
plt.figure(figsize=(10,8))



sns.heatmap(Bsmt_feats.corr(), annot=True, cmap='RdBu')



plt.show()
HeatingQC_map = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

df_train['HeatingQC'].replace(HeatingQC_map, inplace=True)

df_test['HeatingQC'].replace(HeatingQC_map, inplace=True)



Heating_map = {'GasA':1, 'GasW':1, 'Grav':0, 'Wall':0, 'OthW':0, 'Floor':0}

df_train['Heating'].replace(Heating_map, inplace=True)

df_test['Heating'].replace(Heating_map, inplace=True)
create_boxen_count_2x2('Heating', 'HeatingQC', (14,10))
Electrical_map = {'SBrkr':1, 'FuseA':0, 'FuseF':0, 'FuseP':0, 'Mix':0}

df_train['Electrical'].replace(Electrical_map, inplace=True)

df_test['Electrical'].replace(Electrical_map, inplace=True)
create_boxen_count_2x3('Utilities', 'CentralAir', 'Electrical', (16,10))
utilities_feats = df_train[['Heating', 'HeatingQC', 'Utilities', 

                            'CentralAir', 'Electrical', 'Log_SalePrice']]
plt.figure(figsize=(10,8))



sns.heatmap(utilities_feats.corr(), annot=True, cmap='RdBu')



plt.show()
plt.figure(figsize=(13,10))



plt.subplot(221)

sns.distplot(df_train['1stFlrSF'], bins=100, color='g')



plt.subplot(222)

sns.distplot(df_train['2ndFlrSF'], bins=100, color='r')



plt.subplot(223)

sns.distplot(df_train['LowQualFinSF'], bins=100, color='b', kde=False)



plt.subplot(224)

sns.distplot(df_train['GrLivArea'], bins=100, color='g')



plt.show()
# Calculate the skewness and kurtosis of 1stFlrSF and GrLivArea distribution

print('Skewness of the distribution of 1stFlrSF: {}'.format(skew(df_train['1stFlrSF'].dropna())))

print('Kurtosis of the distribution of 1stFlrSF: {}'.format(kurtosis(df_train['1stFlrSF'].dropna())))

print('')

print('Skewness of the distribution of GrLivArea: {}'.format(skew(df_train['GrLivArea'].dropna())))

print('Kurtosis of the distribution of GrLivArea: {}'.format(kurtosis(df_train['GrLivArea'].dropna())))
df_train['1stFlrSF'] = np.log(df_train['1stFlrSF'])

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])



df_test['1stFlrSF'] = np.log(df_test['1stFlrSF'])

df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
# Calculate the skewness and kurtosis of 1stFlrSF and GrLivArea distribution

print('Skewness of the distribution of 1stFlrSF: {}'.format(skew(df_train['1stFlrSF'].dropna())))

print('Kurtosis of the distribution of 1stFlrSF: {}'.format(kurtosis(df_train['1stFlrSF'].dropna())))

print('')

print('Skewness of the distribution of GrLivArea: {}'.format(skew(df_train['GrLivArea'].dropna())))

print('Kurtosis of the distribution of GrLivArea: {}'.format(kurtosis(df_train['GrLivArea'].dropna())))
plt.figure(figsize=(10,9))



ax1 = plt.subplot(121)

sns.regplot(x='1stFlrSF', y='Log_SalePrice', data=df_train, color='lightcoral')



ax2 = plt.subplot(122, sharey=ax1, sharex=ax1)

sns.regplot(x='GrLivArea', y='Log_SalePrice', data=df_train, color='darkseagreen')

plt.setp(ax2.get_yticklabels(), visible=False)

plt.ylabel('')



# Adjusting the spaces between graphs

plt.subplots_adjust(wspace = 0)



plt.show()
create_boxen_count_2x4('BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', (20,12))
# Create 'BsmtFull+HalfBath' feature

df_train['BsmtFull+HalfBath'] = df_train['BsmtFullBath'] + (df_train['BsmtHalfBath']*0.5)

df_test['BsmtFull+HalfBath'] = df_test['BsmtFullBath'] + (df_test['BsmtHalfBath']*0.5)



# Create 'Full+HalfBath' feature

df_train['Full+HalfBath'] = df_train['FullBath'] + (df_train['HalfBath']*0.5)

df_test['Full+HalfBath'] = df_test['FullBath'] + (df_test['HalfBath']*0.5)



# Create 'TotalBath' feature

df_train['TotalBath'] = df_train['Full+HalfBath'] + df_train['BsmtFull+HalfBath']

df_test['TotalBath'] = df_test['Full+HalfBath'] + df_test['BsmtFull+HalfBath']
plt.figure(figsize=(16,6))



ax1 = plt.subplot(131)

sns.boxenplot(x='BsmtFull+HalfBath', y='Log_SalePrice', data=df_train, color='dimgrey')



ax2 = plt.subplot(132, sharey=ax1)

sns.boxenplot(x='Full+HalfBath', y='Log_SalePrice', data=df_train, color='lightcoral')

plt.setp(ax2.get_yticklabels(), visible=False)

plt.ylabel('')



ax3 = plt.subplot(133, sharey=ax1)

sns.boxenplot(x='TotalBath', y='Log_SalePrice', data=df_train, color='darkseagreen')

plt.setp(ax3.get_yticklabels(), visible=False)

plt.ylabel('')





# Adjusting the spaces between graphs

plt.subplots_adjust(wspace = 0)



plt.show()
create_boxen_count_2x1('BedroomAbvGr', (8,8))
bathbed_feats = df_train[['BsmtFull+HalfBath', 'Full+HalfBath', 'TotalBath', 'BedroomAbvGr']]
plt.figure(figsize=(10,8))



sns.heatmap(bathbed_feats.corr(), annot=True, cmap='RdBu')



plt.show()
KitchenQual_map = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

df_train['KitchenQual'].replace(KitchenQual_map, inplace=True)

df_test['KitchenQual'].replace(KitchenQual_map, inplace=True)
create_boxen_count_2x2('KitchenAbvGr', 'KitchenQual', (14,10))
FireplaceQu_map = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}

df_train['FireplaceQu'].replace(FireplaceQu_map, inplace=True)

df_test['FireplaceQu'].replace(FireplaceQu_map, inplace=True)
df_train['FireplaceQu'].fillna(0, inplace=True)

df_test['FireplaceQu'].fillna(0, inplace=True)
create_boxen_count_2x2('Fireplaces', 'FireplaceQu', (14,10))
GarageQual_map = {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}



df_train['GarageQual'].replace(GarageQual_map, inplace=True)

df_test['GarageQual'].replace(GarageQual_map, inplace=True)



df_train['GarageCond'].replace(GarageQual_map, inplace=True)

df_test['GarageCond'].replace(GarageQual_map, inplace=True)
df_train['GarageQual'].fillna(0, inplace=True)

df_test['GarageQual'].fillna(0, inplace=True)



df_train['GarageCond'].fillna(0, inplace=True)

df_test['GarageCond'].fillna(0, inplace=True)
create_boxen_count_2x4('GarageCars', 'GarageQual', 'GarageCond', 'GarageType', (20,12))
plt.figure()



sns.jointplot(x='GarageArea', 

              y='Log_SalePrice', 

              data=df_train, 

              kind='reg', 

              height=9,

              color='seagreen')



plt.show()
plt.figure(figsize=(24,6))



ax1 = plt.subplot(211, sharex=ax2)

sns.countplot(x='GarageYrBlt', data=df_train, color='dimgrey')

plt.xticks(rotation='vertical')

plt.setp(ax1.get_xticklabels(), visible=False)

plt.xlabel('')



ax2 = plt.subplot(212)

sns.boxenplot(x='GarageYrBlt', y='Log_SalePrice', data=df_train, color='lightcoral')

plt.xticks(rotation='vertical')



# Adjusting the spaces between graphs

plt.subplots_adjust(hspace=0)



plt.show()
garage_feats = df_train[['GarageCars', 'GarageQual', 'GarageCond', 

                         'GarageType', 'GarageArea', 'GarageYrBlt', 'Log_SalePrice']]
plt.figure(figsize=(10,8))



sns.heatmap(garage_feats.corr(), annot=True, cmap='RdBu')



plt.show()
plt.figure(figsize=(20,15))



g = sns.pairplot(df_train[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 

                           '3SsnPorch', 'ScreenPorch', 'Log_SalePrice']], 

             palette='Accent',

             kind='scatter',

             diag_kind='auto',

             height=3)



plt.show()
deck_feats = df_train[['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 

                        '3SsnPorch', 'ScreenPorch', 'Log_SalePrice']]
plt.figure(figsize=(10,8))



sns.heatmap(deck_feats.corr(), annot=True, cmap='RdBu')



plt.show()
plt.figure()



sns.jointplot(x='PoolArea', 

              y='Log_SalePrice', 

              data=df_train, 

              kind='scatter', 

              height=9,

              color='seagreen')



plt.show()
PoolQC_map = {'NA':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}

df_train['PoolQC'].replace(PoolQC_map, inplace=True)

df_test['PoolQC'].replace(PoolQC_map, inplace=True)
df_train['PoolQC'].fillna(0, inplace=True)

df_test['PoolQC'].fillna(0, inplace=True)
create_boxen_count_2x1('PoolQC', (8,8))
Fence_map = {'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4}

df_train['Fence'].replace(Fence_map, inplace=True)

df_test['Fence'].replace(Fence_map, inplace=True)
df_train['Fence'].fillna(0, inplace=True)

df_test['Fence'].fillna(0, inplace=True)
create_boxen_count_2x1('Fence', (8,8))
df_train['MiscFeature'].fillna('None', inplace=True)

df_test['MiscFeature'].fillna('None', inplace=True)
create_boxen_count_2x1('MiscFeature', (8,8))
plt.figure()



sns.jointplot(x='MiscVal', 

              y='Log_SalePrice', 

              data=df_train, 

              kind='scatter', 

              height=9,

              color='seagreen')



plt.show()
create_boxen_count_2x1('MoSold', (8,8))
create_boxen_count_2x1('SaleType', (8,8))
create_boxen_count_2x1('SaleCondition', (8,8))