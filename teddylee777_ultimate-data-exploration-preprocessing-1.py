# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

sample_submission = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
train.shape, test.shape
train = train.drop('Id', 1)

test = test.drop('Id', 1)
def compare_box(col):

    print(col)

    print("Train MAX : {0},  Test MAX : {1}".format(train[col].max(), test[col].max()))

    print("Train MIN : {0},  Test MIN : {1}".format(train[col].min(), test[col].min()))

    print("Train MEAN: {0:.2f},  Test MEAN: {1:.2f}".format(train[col].mean(), test[col].mean()))

    print("Train STD : {0:.2f},  Test STD : {1:.2f}".format(train[col].std(), test[col].std()))

    print("Train NaN : {0},  Test STD : {1}".format(train[col].isnull().sum(), test[col].isnull().sum()))

    print("----"*10)

    fg, ax = plt.subplots(figsize=(12, 6))

    fg.add_subplot(1, 2, 1)

    sns.boxplot(y=train[col])

    plt.xlabel('Train')

    fg.add_subplot(1, 2, 2)

    sns.boxplot(y=test[col])

    plt.xlabel('Test')
# Here I want to select numerical data columns

num_col = train.select_dtypes(exclude='object').drop('SalePrice', 1).columns
vis_col = len(num_col)/4+1
fg, ax = plt.subplots(figsize=(12, 18))

for i, col in enumerate(num_col):

    fg.add_subplot(vis_col, 4, i+1)

    sns.distplot(train[col].dropna())

    plt.xlabel(col)

    

plt.tight_layout()

plt.show()
fg, ax = plt.subplots(figsize=(12, 18))

for i, col in enumerate(num_col):

    fg.add_subplot(vis_col, 4, i+1)

    sns.scatterplot(x=train[col], y=train['SalePrice'])

    plt.xlabel(col)

    

plt.tight_layout()

plt.show()
train[num_col].isnull().sum().sort_values(ascending=False).head()
test[num_col].isnull().sum().sort_values(ascending=False).head(15)
train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0], inplace=True)
test['TotalBsmtSF'].fillna(0, inplace=True)

test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0], inplace=True)

test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0], inplace=True)

test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0], inplace=True)

test['GarageArea'].fillna(test['GarageArea'].mode()[0], inplace=True)

test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0], inplace=True)

test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0], inplace=True)

test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mode()[0], inplace=True)

test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mode()[0], inplace=True)

test['GarageCars'].fillna(test['GarageCars'].mode()[0], inplace=True)
train['LotFrontage'].fillna(train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.median()), inplace=True)

test['LotFrontage'].fillna(test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.median()), inplace=True)
train.corr()['GarageYrBlt'].sort_values(ascending=False).head()
train['GarageYrBlt'].fillna(0, inplace=True)

test['GarageYrBlt'].fillna(0, inplace=True)
train[num_col].isnull().sum().sort_values(ascending=False).head()
test[num_col].isnull().sum().sort_values(ascending=False).head()
fg, ax = plt.subplots(figsize=(12, 18))

for i, col in enumerate(num_col):

    fg.add_subplot(9, 4, i+1)

    sns.boxplot(y=train[col])

    plt.xlabel(col)

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(7, 6))

sns.boxplot(y='SalePrice', data=train)
train.loc[train['SalePrice'] > 700000, :]
train = train.loc[train['SalePrice'] < 700000, :]
compare_box('LotArea')
train = train.loc[train['LotArea'] < 100000, :]
compare_box('LotArea')
compare_box('TotalBsmtSF')
compare_box('LotFrontage')
train = train.loc[train['LotFrontage'] < 210, :]
compare_box('LotFrontage')
compare_box('MasVnrArea')
train = train.loc[train['MasVnrArea'] < 1300, :]
compare_box('MasVnrArea')
compare_box('BsmtFinSF1')
train = train.loc[train['BsmtFinSF1'] < 2000, :]
compare_box('BsmtFinSF1')
compare_box('BsmtFinSF2')
train = train.loc[train['BsmtFinSF2'] < 1200, :]
compare_box('BsmtFinSF2')
train.shape, test.shape
train['MoSold'].value_counts()
train['MoSold'] = train['MoSold'].astype('object')

test['MoSold'] = test['MoSold'].astype('object')
train['YrSold'].value_counts()
train['YrSold'] = train['YrSold'].astype('object')

test['YrSold'] = test['YrSold'].astype('object')
num_col = train.select_dtypes(exclude='object').drop('SalePrice', 1).columns

num_col
len(num_col)
cat_col = train.select_dtypes(include='object').columns

cat_col
len(cat_col), len(num_col)
vis_col = len(cat_col)/4 +1

fg, ax = plt.subplots(figsize=(12, 18))



for i, col in enumerate(cat_col):

    fg.add_subplot(vis_col, 4, i+1)

    sns.countplot(train[col])

    plt.xlabel(col)



plt.tight_layout()

plt.show()
train[cat_col].isnull().sum().sort_values(ascending=False).head(20)
test[cat_col].isnull().sum().sort_values(ascending=False).head(20)
train['RoofMatl'].value_counts()
test['RoofMatl'].value_counts()
train['RoofMatl_clean'] = train['RoofMatl'].apply(lambda x: x if x == 'CompShg' else 'Other')

test['RoofMatl_clean'] = test['RoofMatl'].apply(lambda x: x if x == 'CompShg' else 'Other')
train['Alley'].value_counts()
test['Alley'].value_counts()
train['Alley'].fillna('None', inplace=True)

test['Alley'].fillna('None', inplace=True)
train['Alley_bool'] = train['Alley'].apply(lambda x: 0 if x == 'None' else 1)

test['Alley_bool'] = test['Alley'].apply(lambda x: 0 if x == 'None' else 1)
train['Alley_bool'].value_counts()
test['Alley_bool'].value_counts()
train['Electrical'].value_counts()
test['Electrical'].value_counts()
train['Electrical'].fillna(train['Electrical'].mode()[0], inplace=True)
train['Electrical_clean'] = train['Electrical'].apply(lambda x: x if x == 'SBrkr' else 'Fuse')

test['Electrical_clean'] = test['Electrical'].apply(lambda x: x if x == 'SBrkr' else 'Fuse')
train['Electrical_clean'].value_counts()
test['Electrical_clean'].value_counts()
train['MasVnrType'].value_counts()
test['MasVnrType'].value_counts()
train['MasVnrType'].isnull().sum()
test['MasVnrType'].isnull().sum()
train['MasVnrType'].fillna(train['MasVnrType'].mode()[0], inplace=True)

test['MasVnrType'].fillna(test['MasVnrType'].mode()[0], inplace=True)
test['MSZoning'].isnull().sum()
test['MSZoning'].value_counts()
test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace=True)
train['Functional'].value_counts()
test['Functional'].value_counts()
train['Functional'].isnull().sum()
test['Functional'].isnull().sum()
test['Functional'].fillna(test['Functional'].mode()[0], inplace=True)
train['Functional_clean'] = train['Functional'].apply(lambda x: x if x =='Typ' else 'Other')

test['Functional_clean'] = test['Functional'].apply(lambda x: x if x =='Typ' else 'Other')
train['Utilities'].value_counts()
test['Utilities'].value_counts()
test['Utilities'].fillna(test['Utilities'].mode()[0], inplace=True)
train['Exterior2nd'].isnull().sum()
test['Exterior2nd'].isnull().sum()
train['Exterior2nd'].value_counts()
test['Exterior2nd'].value_counts()
ext_other = [

    'Stone',

    'AsphShn',

    'Other',

    'CBlock',

    'ImStucc',

    'Brk Cmn'

]
train['Exterior2nd'] = train['Exterior2nd'].apply(lambda x: 'Other' if x in ext_other else x)

test['Exterior2nd'] = test['Exterior2nd'].apply(lambda x: 'Other' if x in ext_other else x)
train['Exterior2nd'].fillna('Other', inplace=True)

test['Exterior2nd'].fillna('Other', inplace=True)
train['Exterior2nd'].value_counts()
test['Exterior2nd'].value_counts()
train['SaleType'].value_counts()
test['SaleType'].value_counts()
saletype_other = [

    'ConLD',

    'ConLw',

    'ConLI',

    'CWD',

    'Oth',

    'Con'

]
train['SaleType'] = train['SaleType'].apply(lambda x: x if x not in saletype_other else 'Other')

test['SaleType'] = test['SaleType'].apply(lambda x: x if x not in saletype_other else 'Other')
test['SaleType'].fillna('Other', inplace=True)
train['SaleType'].isnull().sum()
test['SaleType'].isnull().sum()
train['SaleType'].value_counts()
test['SaleType'].value_counts()
train['KitchenQual'].value_counts()
test['KitchenQual'].value_counts()
train['KitchenQual'].isnull().sum()
test['KitchenQual'].isnull().sum()
test['KitchenQual'].fillna('TA', inplace=True)
train['Exterior1st'].value_counts()
test['Exterior1st'].value_counts()
ext_other = [

    'Stone',

    'BrkComm',

    'ImStucc',

    'AsphShn',

    'Other',

    'CBlock',

]
train['Exterior1st'] = train['Exterior1st'].apply(lambda x: 'Other' if x in ext_other else x)

test['Exterior1st'] = test['Exterior1st'].apply(lambda x: 'Other' if x in ext_other else x)
train['Exterior1st'].fillna('Other', inplace=True)

test['Exterior1st'].fillna('Other', inplace=True)
train['Exterior1st'].isnull().sum()
test['Exterior1st'].isnull().sum()
train['MiscFeature'].value_counts()
train['MiscFeature'].fillna('None', inplace=True)

test['MiscFeature'].fillna('None', inplace=True)
train['MiscFeature_bool'] = train['MiscFeature'].apply(lambda x: 1 if x == 'None' else 0)

test['MiscFeature_bool'] = test['MiscFeature'].apply(lambda x: 1 if x == 'None' else 0)
train['Alley'].value_counts()
train['Alley'].fillna('None', inplace=True)

test['Alley'].fillna('None', inplace=True)
train['Fence'].value_counts()
test['Fence'].value_counts()
train['Fence'].fillna('None', inplace=True)

test['Fence'].fillna('None', inplace=True)
train['FireplaceQu'].value_counts()
test['FireplaceQu'].value_counts()
train['FireplaceQu'].fillna('None', inplace=True)

test['FireplaceQu'].fillna('None', inplace=True)
train[cat_col].isnull().sum().sort_values(ascending=False).head(20)
test[cat_col].isnull().sum().sort_values(ascending=False).head(20)
train['GarageQual'].value_counts()
train['GarageCond'].value_counts()
train['GarageQual'].fillna('None', inplace=True)

test['GarageQual'].fillna('None', inplace=True)
train['GarageQual_TA'] = train['GarageQual'].apply(lambda x: x if x == 'TA' else 'Other')

test['GarageQual_TA'] = test['GarageQual'].apply(lambda x: x if x == 'TA' else 'Other')
train['GarageFinish'].value_counts()
test['GarageFinish'].value_counts()
train['GarageFinish'].fillna('None', inplace=True)

test['GarageFinish'].fillna('None', inplace=True)
train['GarageType'].value_counts()
test['GarageType'].value_counts()
train['GarageType'].fillna('None', inplace=True)

test['GarageType'].fillna('None', inplace=True)
train['BsmtFinType2'].value_counts()
train['PoolQC'].fillna('None', inplace=True)

test['PoolQC'].fillna('None', inplace=True)
train['GarageCond'].fillna('None', inplace=True)

test['GarageCond'].fillna('None', inplace=True)
train['BsmtExposure'].fillna('None', inplace=True)

test['BsmtExposure'].fillna('None', inplace=True)
train['BsmtCond'].fillna('None', inplace=True)

test['BsmtCond'].fillna('None', inplace=True)
train['BsmtQual'].fillna('None', inplace=True)

test['BsmtQual'].fillna('None', inplace=True)
train['BsmtFinType1'].fillna('None', inplace=True)

test['BsmtFinType1'].fillna('None', inplace=True)
train['BsmtFinType2'].fillna('None', inplace=True)

test['BsmtFinType2'].fillna('None', inplace=True)
train[cat_col].isnull().sum().sort_values(ascending=False).head()
test[cat_col].isnull().sum().sort_values(ascending=False).head()
train['WoodDeckSF_bool'] = train['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)

test['WoodDeckSF_bool'] = test['WoodDeckSF'].apply(lambda x: 1 if x > 0 else 0)
train['OpenPorchSF_bool'] = train['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)

test['OpenPorchSF_bool'] = test['OpenPorchSF'].apply(lambda x: 1 if x > 0 else 0)
train['EnclosedPorch_bool'] = train['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)

test['EnclosedPorch_bool'] = test['EnclosedPorch'].apply(lambda x: 1 if x > 0 else 0)
train['3SsnPorch_bool'] = train['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)

test['3SsnPorch_bool'] = test['3SsnPorch'].apply(lambda x: 1 if x > 0 else 0)
train['ScreenPorch_bool'] = train['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)

test['ScreenPorch_bool'] = test['ScreenPorch'].apply(lambda x: 1 if x > 0 else 0)
train['PoolArea_bool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

test['PoolArea_bool'] = test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
train['FirePlaces_bool'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

test['FirePlaces_bool'] = test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
def draw_rel(df):

    plt.figure(figsize=(9, 6))

    sns.regplot(y=train['SalePrice'], x=df)
area_map = {

    0: 'GrLivArea',

    1: 'TotalBsmtSF',

    2: 'LotArea',

    3: 'GarageArea',

}
for col in area_map.values():

    draw_rel(train[col])
train['LotArea_log'] = np.log1p(train['LotArea'])

test['LotArea_log'] = np.log1p(test['LotArea'])
draw_rel(train['LotArea_log'])
area_map = {

    0: 'GrLivArea',

    1: 'TotalBsmtSF',

    2: 'LotArea_log',

    3: 'GarageArea',

}
to_draw = {}

to_add_col = {}
for i in range(4):

    for j in range(i+1, 4):

        col_name = area_map[i]+"_"+area_map[j]+"_sum"

        col_value = train[area_map[i]] +train[area_map[j]]

        corr_val = col_value.corr(train['SalePrice'])

        print("CORR: {0} ===> {1}".format(col_name, corr_val))

        to_add_col[col_name] = col_value

        to_draw[col_name] = col_value
for i in range(4):

    for j in range(i+1, 4):

        col_name = area_map[i]+"_"+area_map[j]+"_mul"

        col_value = train[area_map[i]] * train[area_map[j]]

        corr_val = col_value.corr(train['SalePrice'])

        print("CORR: {0} ===> {1}".format(col_name, corr_val))

        to_add_col[col_name] = col_value

        to_draw[col_name] = col_value
for i in range(4):

    for j in range(i+1, 4):

        for z in range(j+1, 4):

            col_name = area_map[i]+"_"+area_map[j]+"_"+area_map[z]+"_sum"

            col_value = train[area_map[i]]+train[area_map[j]]+train[area_map[z]]

            corr_val = col_value.corr(train['SalePrice'])

            print("CORR: {0} ===> {1}".format(col_name, corr_val))

            to_add_col[col_name] = col_value

            to_draw[col_name] = col_value
for i in range(4):

    for j in range(i+1, 4):

        for z in range(j+1, 4):

            col_name = area_map[i]+"_"+area_map[j]+"_"+area_map[z]+"_mul"

            col_value = np.sqrt(train[area_map[i]]*train[area_map[j]]*train[area_map[z]])

            corr_val = col_value.corr(train['SalePrice'])

            print("CORR: {0} ===> {1}".format(col_name, corr_val))

            to_add_col[col_name] = col_value

            to_draw[col_name] = col_value
fg, ax = plt.subplots(figsize=(18,18))

# fg, ax = plt.subplots()

n_col = 4

n_row = len(to_draw) // n_col

cnt = 1

for col, value in to_draw.items():

    fg.add_subplot(n_row, n_col, cnt)

    sns.regplot(x=value, y=train['SalePrice'])

    plt.xlabel(col)

    cnt+=1

plt.tight_layout()

plt.show()
key_df = pd.DataFrame(to_add_col)

key_df
plt.figure(figsize=(12, 12))

sns.heatmap(key_df.corr())

plt.tight_layout()

plt.show()
train['GrLivArea_TotalBsmtSF_GarageArea_sum'] = train['GrLivArea'] + train['TotalBsmtSF'] +train['GarageArea']

train['LotArea_log_GarageArea_mul'] = train['LotArea_log'] * train['GarageArea']
test['GrLivArea_TotalBsmtSF_GarageArea_sum'] = test['GrLivArea'] + test['TotalBsmtSF'] +test['GarageArea']

test['LotArea_log_GarageArea_mul'] = test['LotArea_log'] * test['GarageArea']
train.shape, test.shape
other_area_map = {

    0: 'WoodDeckSF',

    1: 'OpenPorchSF',

    2: 'EnclosedPorch',

    3: '3SsnPorch',

    4: 'ScreenPorch',

    5: 'PoolArea',

}
to_draw = {

    'WoodDeckSF': train['WoodDeckSF'],

    'OpenPorchSF': train['OpenPorchSF'],

    'EnclosedPorch': train['EnclosedPorch'],

    '3SsnPorch': train['3SsnPorch'],

    'ScreenPorch': train['ScreenPorch'],

    'PoolArea': train['PoolArea'],

}
for i in range(6):

    for j in range(i+1, 6):

        col_name = other_area_map[i]+"_"+other_area_map[j]+"_sum"

        col_value = train[other_area_map[i]] +train[other_area_map[j]]

        corr_val = col_value.corr(train['SalePrice'])

        print("CORR: {0} ===> {1}".format(col_name, corr_val))

        if corr_val > 0.3:

            to_add_col[col_name] = col_value

        to_draw[col_name] = col_value
for i in range(6):

    for j in range(i+1, 6):

        for z in range(j+1, 6):

            col_name = other_area_map[i]+"_"+other_area_map[j]+"_" + other_area_map[z] +"_sum"

            col_value = train[other_area_map[i]] +train[other_area_map[j]] + train[other_area_map[z]]

            corr_val = col_value.corr(train['SalePrice'])

            print("CORR: {0} ===> {1}".format(col_name, corr_val))

            if corr_val > 0.3:

                to_add_col[col_name] = col_value

            to_draw[col_name] = col_value
for i in range(6):

    for j in range(i+1, 6):

        col_name = other_area_map[i]+"_"+other_area_map[j]+"_mul"

        col_value = train[other_area_map[i]]*train[other_area_map[j]]

        corr_val = col_value.corr(train['SalePrice'])

        print("CORR: {0} ===> {1}".format(col_name, corr_val))

        if corr_val > 0.3:

            to_add_col[col_name] = col_value

        to_draw[col_name] = col_value
for i in range(6):

    for j in range(i+1, 6):

        for z in range(j+1, 6):

            col_name = other_area_map[i]+"_"+other_area_map[j]+"_" + other_area_map[z] +"_mul"

            col_value = train[other_area_map[i]] * train[other_area_map[j]] * train[other_area_map[z]]

            corr_val = col_value.corr(train['SalePrice'])

            print("CORR: {0} ===> {1}".format(col_name, corr_val))

            if corr_val > 0.3:

                to_add_col[col_name] = col_value

            to_draw[col_name] = col_value
fg, ax = plt.subplots(figsize=(18,12))

# fg, ax = plt.subplots()

n_col = 6

n_row = len(to_add_col) // n_col + 1

cnt = 1

for col, value in to_add_col.items():

    fg.add_subplot(n_row, n_col, cnt)

    sns.regplot(x=value, y=train['SalePrice'])

    plt.xlabel(col)

    cnt+=1

plt.tight_layout()

plt.show()
key_df = pd.DataFrame(to_add_col)

key_df
plt.figure(figsize=(12, 12))

sns.heatmap(key_df.corr())

plt.tight_layout()

plt.show()
train['FullBathCount'] = train['BsmtFullBath'] + train['FullBath']

train['HalfBathCount'] = train['HalfBath'] + train['BsmtHalfBath']
test['FullBathCount'] = test['BsmtFullBath'] + test['FullBath']

test['HalfBathCount'] = test['HalfBath'] + test['BsmtHalfBath']
train['OverallMean'] = (train['OverallCond']+ train['OverallQual'])/2

test['OverallMean'] = (test['OverallCond']+ test['OverallQual'])/2
train['GrLivArea_OverallQual_mul'] = train['GrLivArea_TotalBsmtSF_GarageArea_sum'] * train['OverallQual'].astype(int)

test['GrLivArea_OverallQual_mul'] = test['GrLivArea_TotalBsmtSF_GarageArea_sum'] * test['OverallQual'].astype(int)
train['GrLivArea_OverallCond_mul'] = train['GrLivArea_TotalBsmtSF_GarageArea_sum'] * train['OverallCond'].astype(int)

test['GrLivArea_OverallCond_mul'] = test['GrLivArea_TotalBsmtSF_GarageArea_sum'] * test['OverallCond'].astype(int)
compare_box('GrLivArea_OverallQual_mul')
train.isnull().sum().sort_values(ascending=False).head()
test.isnull().sum().sort_values(ascending=False).head()
## Columns to remove

to_remove_cols = [

    'GarageYrBlt',

    'Utilities',

    'Street',

    'SalePrice',

    'PoolQC',

    'Id',

]