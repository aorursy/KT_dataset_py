import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
df_train_copy = df_train.copy()

df_test_copy = df_test.copy()
print('Training Data:')

print(df_train_copy.shape,'\n')

print('Test Data:')

print(df_test.shape)
# Utility function to view features that have missing data

def view_missing_data(df):

    df_missing = pd.concat([(len(df)-df.count()),df.isna().sum()*100/len(df)],axis=1)

    df_missing.columns = ['Missing Count','%Missing']

    return (df_missing[df_missing['%Missing']>0])
print('Training Set:')

print(view_missing_data(df_train_copy))

print('\n')

print('Test Set:')

print(view_missing_data(df_test))
print('Training Set:')

print(df_train_copy['Alley'].unique(),'\n')

print('Test Set:')

print(df_test['Alley'].unique())
# Training set

df_train_copy['Alley'].fillna('No Alley Access',inplace=True)



# Test set

df_test['Alley'].fillna('No Alley Access', inplace=True)
mean_area_per_type = df_train_copy.groupby('MasVnrType').mean()['MasVnrArea']



# Utility function to calculate mean Area per Veneer Type

def mean_area_type(features):

    area = features[1]

    vtype = features[0]

    if pd.isnull(area):

        return mean_area_per_type.loc[vtype]

    else:

        return area
# Training set

# Both MasVnrType and MasVnrArea missing

len_1 = len(df_train_copy[(df_train_copy['MasVnrType'].isna()) & (df_train_copy['MasVnrArea'].isna())])

if len_1>0:

    indices_1 = (df_train_copy['MasVnrType'].isna()) & (df_train_copy['MasVnrArea'].isna())

    df_train_copy.loc[indices_1, 'MasVnrType'] = df_train_copy.loc[indices_1, 'MasVnrType'].fillna('None')

# Only MasVnrType missing

len_2 = len(df_train_copy[(df_train_copy['MasVnrType'].isna()) & (df_train_copy['MasVnrArea'].notnull())])

if len_2>0:

    max_count_idx = df_train_copy['MasVnrType'].value_counts().index[0]

    indices_2 = (df_train_copy['MasVnrType'].isna()) & (df_train_copy['MasVnrArea'].notnull())

    df_train_copy.loc[indices_2, 'MasVnrType'] = df_train_copy.loc[indices_2, 'MasVnrType'].fillna(max_count_idx)

# Only MasVnrArea missing

len_3 = len(df_train_copy[(df_train_copy['MasVnrType'].notnull()) & (df_train_copy['MasVnrArea'].isna())])

if len_3>0:

    indices_3 = (df_train_copy['MasVnrType'].notnull()) & (df_train_copy['MasVnrArea'].isna())

    df_train_copy.loc[indices_3, 'MasVnrArea'] = df_train_copy.loc[indices_3, ['MasVnrType','MasVnrArea']].apply(mean_area_type, axis=1)
# Test set

# Both MasVnrType and MasVnrArea missing

len_1 = len(df_test[(df_test['MasVnrType'].isna()) & (df_test['MasVnrArea'].isna())])

if len_1>0:

    indices_1 = (df_test['MasVnrType'].isna()) & (df_test['MasVnrArea'].isna())

    df_test.loc[indices_1, 'MasVnrType'] = df_test.loc[indices_1, 'MasVnrType'].fillna('None')

# Only MasVnrType missing

len_2 = len(df_test[(df_test['MasVnrType'].isna()) & (df_test['MasVnrArea'].notnull())])

if len_2>0:

    max_count_idx = df_train_copy['MasVnrType'].value_counts().index[0]

    indices_2 = (df_test['MasVnrType'].isna()) & (df_test['MasVnrArea'].notnull())

    df_test.loc[indices_2, 'MasVnrType'] = df_test.loc[indices_2, 'MasVnrType'].fillna(max_count_idx)

# Only MasVnrArea missing

len_3 = len(df_test[(df_train_copy['MasVnrType'].notnull()) & (df_test['MasVnrArea'].isna())])

if len_3>0:

    indices_3 = (df_test['MasVnrType'].notnull()) & (df_test['MasVnrArea'].isna())

    df_test.loc[indices_3, 'MasVnrArea'] = df_test.loc[indices_3, ['MasVnrType','MasVnrArea']].apply(mean_area_type, axis=1)
feat_1 = 'BsmtQual'

feat_2 = 'BsmtCond'

feat_3 = 'BsmtExposure'

feat_4 = 'BsmtFinType1'

feat_5 = 'BsmtFinSF1'

feat_6 = 'BsmtFinType2'

feat_7 = 'BsmtFinSF2'

feat_8 = 'BsmtUnfSF'

feat_9 = 'TotalBsmtSF'

feat_10 = 'BsmtFullBath'

feat_11 = 'BsmtHalfBath'



all_feat = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8, feat_9, feat_10, feat_11]
# Training Set

# Print unique values of features 1,2,3,4 & 6

print(feat_1, df_train_copy[feat_1].unique())

print(feat_2, df_train_copy[feat_2].unique())

print(feat_3, df_train_copy[feat_3].unique())

print(feat_4, df_train_copy[feat_4].unique())

print(feat_6, df_train_copy[feat_6].unique())



df_train_bsmt_feat_1_2_3_4_6_idx = df_train_copy[(df_train_copy[feat_1].isna()) & 

                                       (df_train_copy[feat_2].isna()) & 

                                       (df_train_copy[feat_3].isna()) & 

                                       (df_train_copy[feat_4].isna()) & 

                                       (df_train_copy[feat_6].isna())].index



# Find sum of total basement square feet for observations that is missing data for features 1,2,3,4 & 6

print(feat_9+' Sum:', df_train_copy[(df_train_copy[feat_1].isna()) & 

                                    (df_train_copy[feat_2].isna()) & 

                                    (df_train_copy[feat_3].isna()) & 

                                    (df_train_copy[feat_4].isna()) & 

                                    (df_train_copy[feat_6].isna())][feat_9].sum())



# Number of observations missing data for features 1,2,3,4 & 6

print('Number of Training Observations missing all features: ', len(df_train_copy[(df_train_copy[feat_1].isna()) & 

                                                                                  (df_train_copy[feat_2].isna()) & 

                                                                                  (df_train_copy[feat_3].isna()) & 

                                                                                  (df_train_copy[feat_4].isna()) & 

                                                                                  (df_train_copy[feat_6].isna())]))
# Training Set

df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_1] = df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_1].fillna('No Basement')

df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_2] = df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_2].fillna('No Basement')

df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_3] = df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_3].fillna('No Basement')

df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_4] = df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_4].fillna('No Basement')

df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_6] = df_train_copy.loc[df_train_bsmt_feat_1_2_3_4_6_idx,feat_6].fillna('No Basement')
# Training Set

# Print unique values of features 1,2,3,4 & 6

print(feat_1, df_test[feat_1].unique())

print(feat_2, df_test[feat_2].unique())

print(feat_3, df_test[feat_3].unique())

print(feat_4, df_test[feat_4].unique())

print(feat_6, df_test[feat_6].unique())



df_test_bsmt_feat_1_2_3_4_6_idx = df_test[(df_test[feat_1].isna()) & 

                                          (df_test[feat_2].isna()) & 

                                          (df_test[feat_3].isna()) & 

                                          (df_test[feat_4].isna()) & 

                                          (df_test[feat_6].isna())].index



# Find sum of total basement square feet for observations that is missing data for features 1,2,3,4 & 6

print(feat_9+' Sum:', df_test[(df_test[feat_1].isna()) & 

                              (df_test[feat_2].isna()) & 

                              (df_test[feat_3].isna()) & 

                              (df_test[feat_4].isna()) & 

                              (df_test[feat_6].isna())][feat_9].sum())



# Number of observations missing data for features 1,2,3,4 & 6

print('Number of Training Observations missing all features: ', len(df_test[(df_test[feat_1].isna()) &

                                                                            (df_test[feat_2].isna()) & 

                                                                            (df_test[feat_3].isna()) & 

                                                                            (df_test[feat_4].isna()) & 

                                                                            (df_test[feat_6].isna())]))
# Test Set

df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_1] = df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_1].fillna('No Basement')

df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_2] = df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_2].fillna('No Basement')

df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_3] = df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_3].fillna('No Basement')

df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_4] = df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_4].fillna('No Basement')

df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_6] = df_test.loc[df_test_bsmt_feat_1_2_3_4_6_idx,feat_6].fillna('No Basement')
print('Training Set:')

print(view_missing_data(df_train_copy[all_feat]),'\n')

print('Test Set:')

print(view_missing_data(df_test[all_feat]),'\n')
# TotalBsmtSF

totalbsmtsf_mean = df_train_copy['TotalBsmtSF'].mean()

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['TotalBsmtSF'].iloc[i]):

        if df_train_copy['BsmtQual'].iloc[i] == 'No Basement':

            df_train_copy.loc[i,'TotalBsmtSF'] = 0

        else:

            df_train_copy.loc[i,'TotalBsmtSF'] = totalbsmtsf_mean

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['TotalBsmtSF'].iloc[i]):

        if df_test['BsmtQual'].iloc[i] == 'No Basement':

            df_test.loc[i,'TotalBsmtSF'] = 0

        else:

            df_test.loc[i,'TotalBsmtSF'] = totalbsmtsf_mean

    else:

        pass



# Bsmt Exposure

bsmtexpos_maxfreq = df_train_copy[df_train_copy['BsmtExposure']!='No Basement']['BsmtExposure'].value_counts().index[0]

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtExposure'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtExposure'] = 'No Basement'

        else:

            df_train_copy.loc[i,'BsmtExposure'] = bsmtexpos_maxfreq

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtExposure'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtExposure'] = 'No Basement'

        else:

            df_test.loc[i,'BsmtExposure'] = bsmtexpos_maxfreq

    else:

        pass



# Bsmt FinType2

bsmtfintype2_maxfreq = df_train_copy[df_train_copy['BsmtFinType2']!='No Basement']['BsmtFinType2'].value_counts().index[0]

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtFinType2'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtFinType2'] = 'No Basement'

        else:

            df_train_copy.loc[i,'BsmtFinType2'] = bsmtfintype2_maxfreq

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtFinType2'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtFinType2'] = 'No Basement'

        else:

            df_test.loc[i,'BsmtFinType2'] = bsmtfintype2_maxfreq

    else:

        pass



# Bsmt Qual

bsmtqual_maxfreq = df_train_copy[df_train_copy['BsmtQual']!='No Basement']['BsmtQual'].value_counts().index[0]

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtQual'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtQual'] = 'No Basement'

        else:

            df_train_copy.loc[i,'BsmtQual'] = bsmtqual_maxfreq

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtQual'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtQual'] = 'No Basement'

        else:

            df_test.loc[i,'BsmtQual'] = bsmtqual_maxfreq

    else:

        pass



# Bsmt Cond

bsmtcond_maxfreq = df_train_copy[df_train_copy['BsmtCond']!='No Basement']['BsmtCond'].value_counts().index[0]

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtCond'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtCond'] = 'No Basement'

        else:

            df_train_copy.loc[i,'BsmtCond'] = bsmtcond_maxfreq

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtCond'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtCond'] = 'No Basement'

        else:

            df_test.loc[i,'BsmtCond'] = bsmtcond_maxfreq

    else:

        pass



# Bsmt FinSF1

bsmtfinsf1_mean = df_train_copy[df_train_copy['BsmtQual']!='No Basement']['BsmtFinSF1'].mean()

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtFinSF1'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtFinSF1'] = 0

        else:

            df_train_copy.loc[i,'BsmtFinSF1'] = bsmtfinsf1_mean

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtFinSF1'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtFinSF1'] = 0

        else:

            df_test.loc[i,'BsmtFinSF1'] = bsmtfinsf1_mean

    else:

        pass





# Bsmt FinSF2

bsmtfinsf2_mean = df_train_copy[df_train_copy['BsmtQual']!='No Basement']['BsmtFinSF2'].mean()

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtFinSF2'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtFinSF2'] = 0

        else:

            df_train_copy.loc[i,'BsmtFinSF2'] = bsmtfinsf2_mean

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtFinSF2'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtFinSF2'] = 0

        else:

            df_test.loc[i,'BsmtFinSF2'] = bsmtfinsf2_mean

    else:

        pass

    

    

# Bsmt UnfSF

bsmtunfsf_mean = df_train_copy[df_train_copy['BsmtQual']!='No Basement']['BsmtUnfSF'].mean()

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtUnfSF'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtUnfSF'] = 0

        else:

            df_train_copy.loc[i,'BsmtUnfSF'] = bsmtunfsf_mean

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtUnfSF'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtUnfSF'] = 0

        else:

            df_test.loc[i,'BsmtUnfSF'] = bsmtunfsf_mean

    else:

        pass



# Bsmt Full Bath

bsmtfullbath_median = df_train_copy[df_train_copy['BsmtQual']!='No Basement']['BsmtFullBath'].median()

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtFullBath'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtFullBath'] = 0

        else:

            df_train_copy.loc[i,'BsmtFullBath'] = bsmtfullbath_median

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtFullBath'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtFullBath'] = 0

        else:

            df_test.loc[i,'BsmtFullBath'] = bsmtfullbath_median

    else:

        pass

    

# Bsmt Half Bath

bsmthalfbath_median = df_train_copy[df_train_copy['BsmtQual']!='No Basement']['BsmtHalfBath'].median()

# Training Set

for i in range(len(df_train_copy)):

    if pd.isnull(df_train_copy['BsmtHalfBath'].iloc[i]):

        if df_train_copy['TotalBsmtSF'].iloc[i] == 0:

            df_train_copy.loc[i,'BsmtHalfBath'] = 0

        else:

            df_train_copy.loc[i,'BsmtHalfBath'] = bsmthalfbath_median

    else:

        pass



# Test Set

for i in range(len(df_test)):

    if pd.isnull(df_test['BsmtHalfBath'].iloc[i]):

        if df_test['TotalBsmtSF'].iloc[i] == 0:

            df_test.loc[i,'BsmtHalfBath'] = 0

        else:

            df_test.loc[i,'BsmtHalfBath'] = bsmthalfbath_median

    else:

        pass
feat_1 = 'GarageArea'

feat_2 = 'GarageCars'

feat_3 = 'GarageCond'

feat_4 = 'GarageFinish'

feat_5 = 'GarageQual'

feat_6 = 'GarageType'

feat_7 = 'GarageYrBlt'



all_feat = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7]



print('Training Set:')

print(view_missing_data(df_train_copy[all_feat]), '\n')

print('Test Set:')

print(view_missing_data(df_test[all_feat]), '\n')
print('Mean Garage Area of observations missing data for Garage Cond: ', df_train_copy[df_train_copy['GarageCond'].isna()]['GarageArea'].sum())

print('Mean Garage Area of observations missing data for Garage Finish: ', df_train_copy[df_train_copy['GarageFinish'].isna()]['GarageArea'].sum())

print('Mean Garage Area of observations missing data for Garage Qual: ', df_train_copy[df_train_copy['GarageQual'].isna()]['GarageArea'].sum())

print('Mean Garage Area of observations missing data for Garage Type: ', df_train_copy[df_train_copy['GarageType'].isna()]['GarageArea'].sum())

print('Mean Garage Area of observations missing data for Garage Year Built: ', df_train_copy[df_train_copy['GarageYrBlt'].isna()]['GarageArea'].sum())
df_train_copy[all_feat]=df_train_copy[all_feat].fillna('No Garage')
df_test[(df_test['GarageArea'].isna()) | (df_test['GarageCars'].isna())][all_feat]
idx = df_test[(df_test['GarageArea'].isna()) | (df_test['GarageCars'].isna())].index



# Garage Area

df_test.loc[idx,'GarageArea'] = int(df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageArea'].mean())



# GarageCars

df_test.loc[idx,'GarageCars'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageCars'].median()



# GarageCond

df_test.loc[idx,'GarageCond'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageCond'].value_counts().index[0]



# GarageFinish

df_test.loc[idx,'GarageFinish'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageFinish'].value_counts().index[0]



# GarageQual

df_test.loc[idx,'GarageQual'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageQual'].value_counts().index[0]



# GarageYrBlt

df_test.loc[idx,'GarageYrBlt'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageYrBlt'].median()
print('Total Garage Area of observations missing data for Garage Cond: ', df_test[df_test['GarageCond'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Finish: ', df_test[df_test['GarageFinish'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Qual: ', df_test[df_test['GarageQual'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Type: ', df_test[df_test['GarageType'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Year Built: ', df_test[df_test['GarageYrBlt'].isna()]['GarageArea'].sum())
df_test[((df_test['GarageCond'].isna()) | (df_test['GarageFinish'].isna()) | (df_test['GarageQual'].isna()) |(df_test['GarageYrBlt'].isna())) & (df_test['GarageArea']>0)][all_feat]
idx = df_test[((df_test['GarageCond'].isna()) | (df_test['GarageFinish'].isna()) | (df_test['GarageQual'].isna()) |(df_test['GarageYrBlt'].isna())) & (df_test['GarageArea']>0)].index



# GarageCond

df_test.loc[idx,'GarageCond'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageCond'].value_counts().index[0]



# GarageFinish

df_test.loc[idx,'GarageFinish'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageFinish'].value_counts().index[0]



# GarageQual

df_test.loc[idx,'GarageQual'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageQual'].value_counts().index[0]



# GarageYrBlt

df_test.loc[idx,'GarageYrBlt'] = df_train_copy[df_train_copy['GarageType'] == 'Detchd']['GarageYrBlt'].median()
print('Total Garage Area of observations missing data for Garage Cond: ', df_test[df_test['GarageCond'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Finish: ', df_test[df_test['GarageFinish'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Qual: ', df_test[df_test['GarageQual'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Type: ', df_test[df_test['GarageType'].isna()]['GarageArea'].sum())

print('Total Garage Area of observations missing data for Garage Year Built: ', df_test[df_test['GarageYrBlt'].isna()]['GarageArea'].sum())
df_test[all_feat]=df_test[all_feat].fillna('No Garage')
# Training Set

print('Training Set:')

print(df_train_copy[df_train_copy['FireplaceQu'].isna()]['Fireplaces'].value_counts())



# Test Set

print('Test Set:')

print(df_test[df_test['FireplaceQu'].isna()]['Fireplaces'].value_counts())
# Training Set

df_train_copy['FireplaceQu'].fillna('No Fireplace', inplace=True)



# Test Set

df_test['FireplaceQu'].fillna('No Fireplace', inplace=True)
# Training Set

print('Training Set:')

print(df_train_copy[df_train_copy['PoolQC'].isna()]['PoolArea'].value_counts())



# Test Set

print('Test Set:')

print(df_test[df_test['PoolQC'].isna()]['PoolArea'].value_counts())
# Training Set

df_train_copy['PoolQC'].fillna('No Pool', inplace=True)



max_poolqc = df_train_copy[df_train_copy['PoolQC']!='No Pool']['PoolQC'].value_counts().index[0]

# Test Set

idx_zero = df_test[(df_test['PoolArea']==0) & df_test['PoolQC'].isna()].index

df_test.loc[idx_zero,'PoolQC'] = df_test.loc[idx_zero,'PoolQC'].fillna('No Pool')

idx_nonzero = df_test[(df_test['PoolArea']>0) & df_test['PoolQC'].isna()].index

df_test.loc[idx_nonzero,'PoolQC'] = df_test.loc[idx_nonzero,'PoolQC'].fillna(max_poolqc)
print('Training Set:')

print(df_train_copy['Fence'].unique(),'\n')

print('Test Set:')

print(df_test['Fence'].unique(),'\n')
# Training Set

df_train_copy['Fence'].fillna('No Fence', inplace=True)



# Test Set

df_test['Fence'].fillna('No Fence', inplace=True)
df_mean_area_per_neigh = df_train_copy.groupby('Neighborhood').mean()['LotFrontage']



# Function to calculate mean area per neighborhood

def fill_area_neigh(features):

    area = features[0]

    neigh = features[1]

    

    if pd.isnull(area):

        return df_mean_area_per_neigh[neigh]

    else:

        return area

    

# Training Set

df_train_copy['LotFrontage'] = df_train_copy[['LotFrontage','Neighborhood']].apply(fill_area_neigh,axis=1)



# Test Set

df_test['LotFrontage'] = df_test[['LotFrontage','Neighborhood']].apply(fill_area_neigh,axis=1)
df_train_copy.drop(df_train_copy[df_train_copy['Electrical'].isna()].index, axis=0, inplace=True)
# Test Set

df_test['MSZoning'].fillna(df_train_copy['MSZoning'].value_counts().index[0], inplace=True)

df_test['Utilities'].fillna(df_train_copy['Utilities'].value_counts().index[0], inplace=True)

df_test['Exterior1st'].fillna(df_train_copy['Exterior1st'].value_counts().index[0], inplace=True)

df_test['Exterior2nd'].fillna(df_train_copy['Exterior2nd'].value_counts().index[0], inplace=True)

df_test['KitchenQual'].fillna(df_train_copy['KitchenQual'].value_counts().index[0], inplace=True)

df_test['Functional'].fillna(df_train_copy['Functional'].value_counts().index[0], inplace=True)

df_test['SaleType'].fillna(df_train_copy['SaleType'].value_counts().index[0], inplace=True)
# Training Set

print('Training Set:')

print(df_train_copy[df_train_copy['MiscFeature'].isna()]['MiscVal'].value_counts())

# Test Set

print('Test Set:')

print(df_test[df_test['MiscFeature'].isna()]['MiscVal'].value_counts())
# Training Set

df_train_copy['MiscFeature'].fillna('None', inplace=True)



max_miscfeat = df_train_copy[df_train_copy['MiscFeature']!='None']['MiscFeature'].value_counts().index[0]

# Test Set

idx_zero = df_test[(df_test['MiscVal']==0) & df_test['MiscFeature'].isna()].index

df_test.loc[idx_zero,'MiscFeature'] = df_test.loc[idx_zero,'MiscFeature'].fillna('None')

idx_nonzero = df_test[(df_test['MiscVal']>0) & df_test['MiscFeature'].isna()].index

df_test.loc[idx_nonzero,'MiscFeature'] = df_test.loc[idx_nonzero,'MiscFeature'].fillna(max_miscfeat)
print('Training Set:')

print(view_missing_data(df_train_copy))

print('\n')

print('Test Set:')

print(view_missing_data(df_test))
plt.figure(figsize=(14,6))

df_train_copy.corr()['SalePrice'].sort_values()[:-1].plot(kind='bar',color='r')

plt.show()
plt.figure(figsize=(14,6))

sns.lineplot(x='OverallQual', y='SalePrice', data=df_train_copy, c='red',size=1)

sns.lineplot(x='OverallCond', y='SalePrice', data=df_train_copy, c='blue',size=1)

plt.legend(labels=['OverallQual','OverallCond'])

plt.xlabel('Rating from 1-10')

plt.ylabel('SalePrice')

plt.show()
plt.figure(figsize=(14,6))

sns.scatterplot(x='GarageArea',y='SalePrice',data=df_train_copy)

plt.show()
plt.figure(figsize=(14,6))

sns.boxplot(x='GarageCars',y='SalePrice',data=df_train_copy,palette='pastel')

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x='GarageQual',y='SalePrice',data=df_train_copy,color='black')

plt.show()
plt.figure(figsize=(10,6))

sns.scatterplot(x='GarageCond',y='SalePrice',data=df_train_copy,color='black')

plt.show()
# GarageQual

df_train_copy.drop('GarageQual',axis=1,inplace=True)

df_test.drop('GarageQual',axis=1,inplace=True)



# GarageCond

df_train_copy.drop('GarageCond',axis=1,inplace=True)

df_test.drop('GarageCond',axis=1,inplace=True)
plt.figure(figsize=(12,6))

#plt.subplot(221)

sns.scatterplot(x='GrLivArea', y='SalePrice', data=df_train_copy)

plt.tight_layout()

plt.figure(figsize=(12,6))

#plt.subplot(222)

sns.scatterplot(x='1stFlrSF', y='SalePrice', data=df_train_copy)

plt.tight_layout()

plt.figure(figsize=(12,6))

#plt.subplot(223)

sns.scatterplot(x='2ndFlrSF', y='SalePrice', data=df_train_copy)

plt.tight_layout()

plt.show()
df_train_copy['LivArea'] = df_train_copy['GrLivArea'] + df_train_copy['1stFlrSF'] + df_train_copy['2ndFlrSF']

df_test['LivArea'] = df_test['GrLivArea'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']



# Drop GrLivArea, 1stFlrSF and 2ndFlrSF

df_train_copy.drop(['GrLivArea','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)

df_test.drop(['GrLivArea','1stFlrSF','2ndFlrSF'],axis=1,inplace=True)
plt.figure(figsize=(14,6))

sns.scatterplot(x='YearBuilt',y='SalePrice',data=df_train_copy,hue='YearRemodAdd')

plt.show()
df_train_copy['YearLastBuilt'] = df_train_copy[['YearBuilt','YearRemodAdd']].max(axis=1)

df_test['YearLastBuilt'] = df_test[['YearBuilt','YearRemodAdd']].max(axis=1)



# Drop Year Built and Year RemodAdd

df_train_copy.drop(['YearBuilt','YearRemodAdd'],axis=1,inplace=True)

df_test.drop(['YearBuilt','YearRemodAdd'],axis=1,inplace=True)
df_train_copy['PercBsmtFinArea'] = df_train_copy['BsmtFinSF1'] + df_train_copy['BsmtFinSF2']

df_train_copy['PercBsmtFinArea'] = df_train_copy.apply(lambda row:0 if row.TotalBsmtSF==0 else row.PercBsmtFinArea*100/row.TotalBsmtSF,axis=1)

df_test['PercBsmtFinArea'] = df_test['BsmtFinSF1'] + df_test['BsmtFinSF2']

df_test['PercBsmtFinArea'] = df_test.apply(lambda row:0 if row.TotalBsmtSF==0 else row.PercBsmtFinArea*100/row.TotalBsmtSF,axis=1)



# Drop GrLivArea, 1stFlrSF and 2ndFlrSF

df_train_copy.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis=1,inplace=True)

df_test.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis=1,inplace=True)
plt.figure(figsize=(14,6))

sns.scatterplot(x='TotalBsmtSF',y='SalePrice',data=df_train_copy,hue='PercBsmtFinArea',palette='viridis')

plt.show()
# Delete outlier

df_train_copy.drop(df_train_copy[df_train_copy['TotalBsmtSF']>4500].index,axis=0,inplace=True)



# Replot the previous plot

plt.figure(figsize=(14,6))

sns.scatterplot(x='TotalBsmtSF',y='SalePrice',data=df_train_copy,hue='PercBsmtFinArea',palette='viridis')

plt.show()
# Training Set

df_train_copy['TotalBathrooms'] = df_train_copy['FullBath'] + 0.5*df_train_copy['HalfBath'] + df_train_copy['BsmtFullBath'] + 0.5*df_train_copy['BsmtHalfBath']

# Test Set

df_test['TotalBathrooms'] = df_test['FullBath'] + 0.5*df_test['HalfBath'] + df_test['BsmtFullBath'] + 0.5*df_test['BsmtHalfBath']



# Drop other features

df_train_copy.drop(['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'],axis=1,inplace=True)

df_test.drop(['FullBath','HalfBath','BsmtFullBath','BsmtHalfBath'],axis=1,inplace=True)
plt.figure(figsize=(14,6))

sns.violinplot(x='TotalBathrooms',y='SalePrice',data=df_train_copy,palette='pastel')

plt.show()
# Delete outlier

df_train_copy.drop(df_train_copy[df_train_copy['TotalBathrooms']>4.5].index,axis=0,inplace=True)
plt.figure(figsize=(14,6))

df_train_copy.corr()['SalePrice'].sort_values()[:-1].plot(kind='bar',color='green')

plt.show()
num_feat = []

for feat in list(df_train_copy.select_dtypes('int64').columns):

    num_feat.append(feat)

for feat in list(df_train_copy.select_dtypes('float64').columns):

    num_feat.append(feat)

    

for feat in num_feat:

    if df_train_copy.corr()['SalePrice'][feat] < 0.5:

        df_train_copy.drop(feat,axis=1,inplace=True)

        df_test.drop(feat,axis=1,inplace=True)
# Create a list of categorical features

cat_feat = []

for feat in list(df_train_copy.select_dtypes('object').columns):

    cat_feat.append(feat)

print(cat_feat)
#Training Set

df_train_copy['GarageYrBlt'] = df_train_copy['GarageYrBlt'].apply(lambda yr: 0 if yr=='No Garage' else int(yr))



#Test Set

df_test['GarageYrBlt'] = df_test['GarageYrBlt'].apply(lambda yr: 0 if yr=='No Garage' else int(yr))



# Remove GarageYrBlt from cat_feat

cat_feat.remove('GarageYrBlt')
# ExterQual & ExterCond

exter_rating = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

df_train_copy['ExterQual'] = df_train_copy['ExterQual'].apply(lambda x: exter_rating[x])

df_train_copy['ExterCond'] = df_train_copy['ExterCond'].apply(lambda x: exter_rating[x])

df_test['ExterQual'] = df_test['ExterQual'].apply(lambda x: exter_rating[x])

df_test['ExterCond'] = df_test['ExterCond'].apply(lambda x: exter_rating[x])



# BsmtCond & BsmtQual

bsmt_rating = {'No Basement':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

df_train_copy['BsmtCond'] = df_train_copy['BsmtCond'].apply(lambda x: bsmt_rating[x])

df_train_copy['BsmtQual'] = df_train_copy['BsmtQual'].apply(lambda x: bsmt_rating[x])

df_test['BsmtCond'] = df_test['BsmtCond'].apply(lambda x: bsmt_rating[x])

df_test['BsmtQual'] = df_test['BsmtQual'].apply(lambda x: bsmt_rating[x])



# BsmtExposure

bsmtexp_rating = {'No Basement':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}

df_train_copy['BsmtExposure'] = df_train_copy['BsmtExposure'].apply(lambda x: bsmtexp_rating[x])

df_test['BsmtExposure'] = df_test['BsmtExposure'].apply(lambda x: bsmtexp_rating[x])



# HeatingQC

heat_rating = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

df_train_copy['HeatingQC'] = df_train_copy['HeatingQC'].apply(lambda x: heat_rating[x])

df_test['HeatingQC'] = df_test['HeatingQC'].apply(lambda x: heat_rating[x])



# KitchenQual

kitchen_rating = {'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

df_train_copy['KitchenQual'] = df_train_copy['KitchenQual'].apply(lambda x: kitchen_rating[x])

df_test['KitchenQual'] = df_test['KitchenQual'].apply(lambda x: kitchen_rating[x])



# FireplaceQu

fire_rating = {'No Fireplace':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

df_train_copy['FireplaceQu'] = df_train_copy['FireplaceQu'].apply(lambda x: fire_rating[x])

df_test['FireplaceQu'] = df_test['FireplaceQu'].apply(lambda x: fire_rating[x])



# PoolQC

pool_rating = {'No Pool':0, 'Fa':1,'TA':2,'Gd':3,'Ex':4}

df_train_copy['PoolQC'] = df_train_copy['PoolQC'].apply(lambda x: pool_rating[x])

df_test['PoolQC'] = df_test['PoolQC'].apply(lambda x: pool_rating[x])



# Fence

fence_rating = {'No Fence':0, 'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4}

df_train_copy['Fence'] = df_train_copy['Fence'].apply(lambda x: fence_rating[x])

df_test['Fence'] = df_test['Fence'].apply(lambda x: fence_rating[x])



cat_feat.remove('ExterQual')

cat_feat.remove('ExterCond')

cat_feat.remove('BsmtCond')

cat_feat.remove('BsmtQual')

cat_feat.remove('BsmtExposure')

cat_feat.remove('HeatingQC')

cat_feat.remove('KitchenQual')

cat_feat.remove('FireplaceQu')

cat_feat.remove('PoolQC')

cat_feat.remove('Fence')
df_train_copy.drop(['BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)

df_test.drop(['BsmtFinType1','BsmtFinType2'],axis=1,inplace=True)



cat_feat.remove('BsmtFinType1')

cat_feat.remove('BsmtFinType2')
feat_list = []

feat_unique_list = []

feat_unique_list2 = []

for col in cat_feat:

    feat_list.append(col)

    feat_unique_list.append(df_train_copy[col].nunique())

    feat_unique_list2.append(df_test[col].nunique())

df_cat_unique = pd.DataFrame({'Feature': feat_list, 'Train Unique Values': feat_unique_list, 'Test Unique Values': feat_unique_list2})

df_cat_unique['Delta'] = df_cat_unique['Train Unique Values'] - df_cat_unique['Test Unique Values']

df_cat_unique[df_cat_unique['Delta']>0]
df_combined = pd.concat([df_train_copy,df_test])

# Fill missing Saleprice in the test observations with zero

df_combined['SalePrice'].fillna(0,inplace=True)

df_combined['SalePrice']



print('Training Set:')

print(df_train_copy.shape)

print('Test Set:')

print(df_test.shape)

print('Combined Set:')

print(df_combined.shape)
for col in cat_feat:

    df_combined = pd.concat([df_combined,pd.get_dummies(df_combined[col],drop_first=True)],axis=1)

    df_combined.drop(col,axis=1,inplace=True)

print(df_combined.shape)
df_train_copy = df_combined.iloc[0:1456,:]

df_test = df_combined.iloc[1456:,:]

#Drop the target variable SalePrice from the test set

df_test.drop('SalePrice',axis=1,inplace=True)
plt.figure(figsize=(30,6))

df_train_copy.corr()['SalePrice'].sort_values()[:-1].plot(kind='bar',color='green')

plt.show()
num_feat = []

for feat in list(df_train_copy.select_dtypes('int64').columns):

    num_feat.append(feat)

for feat in list(df_train_copy.select_dtypes('float64').columns):

    num_feat.append(feat)

    

for feat in num_feat:

    if df_train_copy.corr()['SalePrice'][feat] < 0.5:

        df_train_copy.drop(feat,axis=1,inplace=True)

        df_test.drop(feat,axis=1,inplace=True)
# Create input matrix and output vector

X_train = df_train_copy.drop('SalePrice',axis=1)

y_train = df_train_copy['SalePrice']

X_test = df_test
plt.figure(figsize=(12,6))

sns.distplot(y_train)

plt.show()
# Natural log transformation

y_train = np.log1p(y_train)

# Distribution plot

plt.figure(figsize=(12,6))

sns.distplot(y_train)

plt.show()
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Import Grid Search and Random Forest Regreesor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

# Grid Search

parameters = [{'n_estimators': [3000,4000,5000,6000] , 'max_features': ['sqrt'] ,'n_jobs': [-1]}]

regressor = RandomForestRegressor()

grid_search = GridSearchCV(estimator = regressor,

                           param_grid = parameters,

                           scoring = 'neg_mean_squared_log_error',

                           cv = 10)

# Fit on training set

grid_search.fit(X_train, y_train)
grid_search.best_params_
rf_reg = grid_search.best_estimator_

rf_reg.fit(X_train,y_train)
predictions = np.expm1(rf_reg.predict(X_test))

output = pd.DataFrame({'Id': df_test_copy.Id, 'SalePrice': (predictions)})

output.to_csv('my_submission.csv', index=False)

print('Your submission was successfully saved!')