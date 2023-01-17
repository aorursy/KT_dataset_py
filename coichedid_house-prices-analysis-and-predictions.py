# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (20, 10)
import seaborn as sns

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import time


# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['Name'] = 'Train'
test['Name'] = 'Test'
combined_set = train.append(test)

print("Train data shape:", combined_set[combined_set.Name == 'Train'].shape)
print("Test data shape:", combined_set[combined_set.Name == 'Test'].shape,"\n\n")
print("Data caracteristics")
print(combined_set.describe())
print("\n\nData types")
print(combined_set.dtypes.value_counts())

## Investigatting null values
num_features = combined_set.shape[0]
null_values = combined_set.columns[combined_set.isnull().any()]
null_features = combined_set[null_values].isnull().sum().sort_values(ascending = False)
null_features_ratio = null_features.apply(lambda x: "{}%".format(round(x/num_features * 100, 3)))
missing_data = pd.DataFrame({'No of Nulls': null_features, 'Ratio': null_features_ratio})
print("\n\nNull values count")
print(missing_data)

## Investigation Skewness of predictor
print("\n\nSkew of Sale Price", combined_set[combined_set.Name == 'Train'].SalePrice.skew())
plt.hist(combined_set[combined_set.Name == 'Train'].SalePrice, color='blue')
plt.show()
## Analysing Pool null values
print('Pool areas of missing pool quality. Zero means house withou pool')
pool_areas = combined_set[combined_set.PoolQC.isnull()]['PoolArea'].value_counts()
print(pool_areas,'\n\n')

## Distinct types of alley
print('Distinct types of alley')
alley_types = train['Alley'].value_counts(dropna=False)
print(alley_types,'\n\n')

## Distinct types of fence
print('Distinct types of fence')
fence_types = train['Fence'].value_counts(dropna=False)
print(fence_types,'\n\n')

## Analysing FireplaceQu null values
print('Fireplace number of missing fireplace quality. Zero means house withou fireplace')
fireplaces = train[train.FireplaceQu.isnull()]['Fireplaces'].value_counts()
print(fireplaces,'\n\n')

## Analysing LotFrontage
print('Trying to figure out why there are missing values of LotFrontage')
## possible related columns
cols = ['Street', 'Alley','LotConfig']
related_values = train[train.LotFrontage.isnull()][cols]
## Replace null values of related columns with "-" 
related_values = related_values.fillna(value="-")
## Concat related columns to get a summary of data
related_values['concat'] = related_values[['Street', 'Alley', 'LotConfig']].apply(lambda x: '::'.join(x), axis=1)
print('Combinations of null values (as "-") on related columns ', cols)
print(related_values.concat.value_counts(), "\n\n")

## Analysing Garage related columns
## possible related columns
cols = ["GarageYrBlt", "GarageType", "GarageFinish", "GarageQual", "GarageCond"]
related_cols = ["GarageCars","GarageArea"]
related_values = train[train[cols].isnull().any(axis=1)][cols + related_cols]
print('Assumption is that garage related columns (', cols, ') makes sense only if house has a garage')
print('Maybe GarageCars and GarageArea are zeroed')
## Replace null values of related columns with "-" 
related_values = related_values.fillna(value="-")
## Concat related collumns to get a summary of data
related_values['GarageNullCols'] = related_values[cols].apply(lambda x: '::'.join(x), axis=1)
related_values['GarageRelatedCols'] = related_values[related_cols].apply(lambda x: '::'.join(x.astype(str)), axis=1)
related_values['GarageCols'] = related_values[['GarageRelatedCols','GarageNullCols']].apply(lambda x: '>>'.join(x), axis=1)
print('Combinations of null values (as "-") on related columns')
print(related_values.GarageCols.value_counts(), "\n\n")

## Analysing Basement related columns
print('Assumption is that basement related columns makes sense only if house has a basement')
print('Maybe TotalBsmtSF is zeroed')
## possible related columns
cols = ["BsmtFinType2", "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual"]
related_cols = ["TotalBsmtSF"]
all_related_cols = ["Id", "TotalBsmtSF","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF"] + cols
related_values = train[train[cols].isnull().any(axis=1)][cols + related_cols]
## Replace null values of related columns with "-" 
related_values = related_values.fillna(value="-")
## Concat related collumns to get a summary of data
related_values['BsmtNullCols'] = related_values[cols].apply(lambda x: '::'.join(x), axis=1)
related_values['BsmtRelatedCols'] = related_values[related_cols].apply(lambda x: '::'.join(x.astype(str)), axis=1)
related_values['BsmtCols'] = related_values[['BsmtRelatedCols','BsmtNullCols']].apply(lambda x: '>>'.join(x), axis=1)
print('Combinations of null values (as "-") on related columns')
print(related_values.BsmtCols.value_counts(), "\n\n")
print('Almost null columns belong to zero basement, but there are two different rows')
rows = train[((train.BsmtFinType2 == 'Unf') & (train.BsmtExposure.isnull())) | ((train.BsmtFinType2.isnull()) & (train.TotalBsmtSF == 3206))]
print("Let's see what's is going on those two rows. Let'ts take a look at all related columns with basement")
print("Ok, Finished Area type 2 is optional and Exposure, probably has a really missing value")
rows[all_related_cols]
## Analysing Masonry veneer columns
print("Maybe, Masonry veneer missing values occurs in the same rows. Maybe, there isn't a masonry veneer")
rows = train[((train.MasVnrType.isnull()) | (train.MasVnrArea.isnull()))]
rows[["Id", "MasVnrType", "MasVnrArea"]]
## Evaluating Object columns
objects = train.select_dtypes(include=[np.object])
desc = objects.describe()
desc
print("Looking at several top frequency values, looks like that there are categories that appears almost everytime.")
top_freq = desc.loc['freq']
count = desc.loc['count']
top = desc.loc['top']
freq_ratio = top_freq/count
print("Let's look at frequency percentage of total count")
d = {'top_cat':top,'ratio':freq_ratio}
freq_comp = pd.DataFrame(d)
freq_comp.sort_values(by='ratio', ascending = False, inplace = True)

sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

plt.figure(figsize= (16, 8))
plt.xticks(rotation='90')
plt.xlabel('Categorical Column')
plt.ylabel('Frequency Ratio')
plt.title('Relevant Categories')
sns.barplot(freq_comp.index, freq_comp.ratio)
freq_comp[(freq_comp.ratio >= .8)]
more_less_relevant_categoricals = ['LandContour', 'BsmtFinType2', 'ExterCond', 'SaleType', 'Condition1',
       'BldgType', 'SaleCondition']

(fig, ax_lst) = plt.subplots(4, 2,figsize=(15,30))
i,j = (0,0)
for cat in more_less_relevant_categoricals:
    ## Apply log to normalize values as most relevant category is much larger then others
    data = np.log(train[cat].value_counts())
    ax_lst[i,j].bar(data.index, data)
    
    ax_lst[i,j].set_ylabel('Count')
    ax_lst[i,j].set_xlabel(cat)
    if j == 1:
        i = i + 1 if i < 3 else 0
        j = 0
    else:
        j = j + 1
numeric = train.select_dtypes(include=[np.number])
print('Evaluating top correlations')
corr = numeric.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:7], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-6:])
sorted_corr = corr['SalePrice'].sort_values(ascending=False)
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

plt.figure(figsize= (16, 8))
plt.xticks(rotation='90')
plt.xlabel('Categorical Column')
plt.ylabel('Frequency Ratio')
plt.title('Relevant Categories')
sns.barplot(sorted_corr.index, sorted_corr)
## Possible values of Overall Quality
print('Overall Quality is a categorical feature, from 1 to 10')
print(train.OverallQual.unique())
print('Also number of cars on garage is a categorical feature')
print(train.GarageCars.unique())
print("Overall Quality and Garage Cars are categorical features.\nLet's visualize them as a bar plot aggregated by median")

quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)

cars_pivot = train.pivot_table(index='GarageCars',
                                  values='SalePrice', aggfunc=np.median)

target = np.log(train.SalePrice) # Eliminate skew
# fig = plt.figure(figsize= (50, 20))  # an empty figure with no axes

(fig, ax_lst) = plt.subplots(3, 2,figsize=(15,15))
#Overall Quality X Sale Price
ax_lst[0,0].bar(quality_pivot.index, quality_pivot.SalePrice)
ax_lst[0,0].set_ylabel('Sale Price')
ax_lst[0,0].set_xlabel('Overall Quality')


#Above Grade Living Area X Sale Price
ax_lst[0,1].scatter(x=train['GrLivArea'], y=target)
ax_lst[0,1].set_ylabel('Sale Price')
ax_lst[0,1].set_xlabel('Above grade living')

#Garage Cars X Sale Price
ax_lst[1,0].bar(cars_pivot.index, cars_pivot.SalePrice)
ax_lst[1,0].set_ylabel('Sale Price')
ax_lst[1,0].set_xlabel('Num of cars on garage')

#Garage Area X Sale Price
ax_lst[1,1].scatter(x=train['GarageArea'], y=target)
ax_lst[1,1].set_ylabel('Sale Price')
ax_lst[1,1].set_xlabel('Area of garage')

#Basement area X Sale Price
ax_lst[2,0].scatter(x=train['TotalBsmtSF'], y=target)
ax_lst[2,0].set_ylabel('Sale Price')
ax_lst[2,0].set_xlabel('Area of basement')

#1st Floor area X Sale Price
ax_lst[2,1].scatter(x=train['1stFlrSF'], y=target)
ax_lst[2,1].set_ylabel('Sale Price')
ax_lst[2,1].set_xlabel('Area of 1st floor')
# plt.subplots_adjust(wspace=1, hspace=1)
plt.tight_layout()
yearSold_pivot = train.pivot_table(index='YrSold',
                                  values='SalePrice', aggfunc=np.median)

overallCond_pivot = train.pivot_table(index='OverallCond',
                                  values='SalePrice', aggfunc=np.median)

MSSubClass_pivot = train.pivot_table(index='MSSubClass',
                                  values='SalePrice', aggfunc=np.median)

KitchenAbvGr_pivot = train.pivot_table(index='KitchenAbvGr',
                                  values='SalePrice', aggfunc=np.median)

# fig = plt.figure(figsize= (16, 8))  # an empty figure with no axes

(fig, ax_lst) = plt.subplots(3, 2, figsize=(15,15))
#Low Quality Finished Area X Sale Price
ax_lst[0,0].scatter(x=train['LowQualFinSF'], y=target)
ax_lst[0,0].set_ylabel('Sale Price')
ax_lst[0,0].set_xlabel('Low Quality Finished Area')


#Year Sold X Sale Price
ax_lst[0,1].bar(yearSold_pivot.index.values, yearSold_pivot.SalePrice.values)
ax_lst[0,1].set_ylabel('Sale Price')
ax_lst[0,1].set_xlabel('Year Sold')

#Overall Condition X Sale Price
ax_lst[1,0].bar(overallCond_pivot.index, overallCond_pivot.SalePrice)
ax_lst[1,0].set_ylabel('Sale Price')
ax_lst[1,0].set_xlabel('Overall Condition')

#type of dwelling X Sale Price
ax_lst[1,1].bar(MSSubClass_pivot.index, MSSubClass_pivot.SalePrice)
ax_lst[1,1].set_ylabel('Sale Price')
ax_lst[1,1].set_xlabel('Type of dwelling')

#Porch area X Sale Price
ax_lst[2,0].scatter(x=train['EnclosedPorch'], y=target)
ax_lst[2,0].set_ylabel('Sale Price')
ax_lst[2,0].set_xlabel('Area of Porch')

#KitchenAbvGr_pivot X Sale Price 
ax_lst[2,1].bar(KitchenAbvGr_pivot.index, KitchenAbvGr_pivot.SalePrice)
ax_lst[2,1].set_ylabel('Sale Price')
ax_lst[2,1].set_xlabel('Kitchen above grade')
# plt.subplots_adjust(wspace=1, hspace=1)
plt.tight_layout()
overallCond_pivot
## Make a new copy of datasets

transf_train = train.copy()
transf_test = test.copy()

#Filling nulls
#PoolQC
transf_train.loc[lambda df: df.PoolArea == 0, 'PoolQC'] = 'No Pool'
transf_test.loc[lambda df: df.PoolArea == 0, 'PoolQC'] = 'No Pool'
# train[train.PoolArea == 0].PoolQC

#MiscFeature
transf_train.MiscFeature.fillna('None',inplace=True)
transf_test.MiscFeature.fillna('None',inplace=True)
# train.MiscFeature.value_counts()

#Alley
transf_train.Alley.fillna('No alley entrace',inplace=True)
transf_test.Alley.fillna('No alley entrace',inplace=True)
# train.Alley.value_counts()

#Fence
transf_train.Fence.fillna('None',inplace=True)
transf_test.Fence.fillna('None',inplace=True)
# train.Fence.value_counts()

#FireplaceQu
transf_train.loc[lambda df: df.Fireplaces == 0, 'FireplaceQu'] = 'No Fireplaces'
transf_test.loc[lambda df: df.Fireplaces == 0, 'FireplaceQu'] = 'No Fireplaces'
# train[train.Fireplaces == 0].FireplaceQu

#LotFrontage
# Get median by neighborhood
med = transf_train.groupby('Neighborhood').LotFrontage.median()
med_test = transf_test.groupby('Neighborhood').LotFrontage.median()
transf_train.loc[lambda df: df.LotFrontage.isnull(), 'LotFrontage'] = transf_train.loc[lambda df: df.LotFrontage.isnull(), 'Neighborhood'].map(med)
transf_test.loc[lambda df: df.LotFrontage.isnull(), 'LotFrontage'] = transf_test.loc[lambda df: df.LotFrontage.isnull(), 'Neighborhood'].map(med_test)

#Garage related columns
garageRelated = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
transf_train.loc[lambda df: df.GarageArea == 0, garageRelated] = 'No garage'
transf_train.loc[lambda df: df.GarageArea == 0, 'GarageYrBlt'] = -1
transf_test.loc[lambda df: df.GarageArea == 0, garageRelated] = 'No garage'
transf_test.loc[lambda df: df.GarageArea == 0, 'GarageYrBlt'] = -1

#Basement related columns
basementNum = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']
basementRelated = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
transf_train.loc[lambda df: df.TotalBsmtSF == 0, basementRelated] = 'No basement'
transf_train.loc[lambda df: df.TotalBsmtSF == 0, basementNum] = 0
transf_train.loc[lambda df: df.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'No extra type'
transf_train.loc[lambda df: df.BsmtExposure.isnull(), 'BsmtExposure'] = 'No'
transf_test.loc[lambda df: df.TotalBsmtSF == 0, basementRelated] = 'No basement'
transf_test.loc[lambda df: df.TotalBsmtSF == 0, basementNum] = 0
transf_test.loc[lambda df: df.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'No extra type'
transf_test.loc[lambda df: df.BsmtExposure.isnull(), 'BsmtExposure'] = 'No'

#Masonry Veneer related
transf_train.loc[lambda df: df.MasVnrArea.isnull(), 'MasVnrArea'] = 0
transf_train.loc[lambda df: df.MasVnrType.isnull(), 'MasVnrType'] = 'None'
transf_test.loc[lambda df: df.MasVnrArea.isnull(), 'MasVnrArea'] = 0
transf_test.loc[lambda df: df.MasVnrType.isnull(), 'MasVnrType'] = 'None'

#Electrical
common_electrical = transf_train.groupby(['Neighborhood','Electrical']).size()
transf_train.loc[lambda df: df.Electrical.isnull(), 'Electrical'] = transf_train.loc[lambda df: df.Electrical.isnull(), 'Neighborhood'].map(lambda n: common_electrical[n].idxmax())
common_electrical_test = transf_test.groupby(['Neighborhood','Electrical']).size()
transf_test.loc[lambda df: df.Electrical.isnull(), 'Electrical'] = transf_test.loc[lambda df: df.Electrical.isnull(), 'Neighborhood'].map(lambda n: common_electrical_test[n].idxmax())

# train.Electrical.value_counts(dropna=False)

## Investigatting null values based on original train dataset
null_values = transf_train.columns[train.isnull().any()]
null_features = transf_train[null_values].isnull().sum().sort_values(ascending = False)
missing_data = pd.DataFrame({'No of Nulls' :null_features})
print("\n\nNull values count")
print(missing_data)

null_values_test = transf_test.columns[test.isnull().any()]
null_features_test = transf_test[null_values_test].isnull().sum().sort_values(ascending = False)
missing_data_test = pd.DataFrame({'No of Nulls' :null_features_test})
print("\n\nNull values count")
print(missing_data_test)
irrelevant_columns = ['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating',
       'GarageCond', 'GarageQual', 'LandSlope', 'CentralAir', 'Functional',
       'BsmtCond', 'PavedDrive', 'Electrical', 'MiscFeature']

transf_train = transf_train.drop(irrelevant_columns, axis=1)
transf_test = transf_test.drop(irrelevant_columns, axis=1)
one_encoded_features = ['LandContour', 'BsmtFinType2', 'BldgType']
two_encoded_features = ['ExterCond', 'SaleType', 'SaleCondition']
all_diff_features = ['Condition1']
other_categorical_features = freq_comp[(freq_comp.ratio < .8)].index

## variable freq_comp has most relevant category of each column
drop_columns = []
# Encoding first case
print('Encoding first case')
def encode_one_encoded(x, default): return 1 if x == default else 0
for f in one_encoded_features:
    if (f in transf_train.columns) & (f in transf_test.columns):
        default_v = freq_comp.loc[f].top_cat
        transf_train['enc_' + f] =  transf_train[f].apply(encode_one_encoded,args=(default_v,))
        transf_test['enc_' + f] =  transf_test[f].apply(encode_one_encoded,args=(default_v,))
        drop_columns.append(f)
        
# Encoding second case
print('Encoding second case')
def encode_two_encoded(x, fs, sc):
    return fs if x == fs else 'sec_category' if x in sc else 'other'
for f in two_encoded_features:
    if (f in transf_train.columns) & (f in transf_test.columns):
        freq = transf_train[f].value_counts()[:3].index # First 3 biggest categories
        default_v = freq[0]
        alternate_v = freq[1:3]
        column_name = 'enc_' + f
        transf_train[column_name] =  transf_train[f].apply(encode_two_encoded,args=(default_v,alternate_v))
        transf_test[column_name] =  transf_test[f].apply(encode_two_encoded,args=(default_v, alternate_v))
        transf_train = pd.get_dummies(transf_train, columns=[column_name])
        transf_test = pd.get_dummies(transf_test, columns=[column_name])
        drop_columns.append(f)

# Encoding third case
print('Encoding third case')
for f in all_diff_features:
    if (f in transf_train.columns) & (f in transf_test.columns):
        column_name = 'enc_' + f
        transf_train = pd.get_dummies(transf_train, prefix=column_name, columns=[f])
        transf_test = pd.get_dummies(transf_test, prefix=column_name, columns=[f])
        
# Encoding other categorical columns
print('Encoding other categorical columns')
for f in other_categorical_features:
    if (f in transf_train.columns) & (f in transf_test.columns):
        column_name = 'enc_' + f
        transf_train = pd.get_dummies(transf_train, prefix=column_name, columns=[f])
        transf_test = pd.get_dummies(transf_test, prefix=column_name, columns=[f])
        
transf_train = transf_train.drop(drop_columns, axis=1)
transf_test = transf_test.drop(drop_columns, axis=1)

outlier_train = transf_train.copy()
outlier_test = transf_test.copy()
## Dropping lines with Above Grade Living Area greater then 4000 feet
outlier_train = outlier_train[outlier_train['GrLivArea'] < 4000]

## Dropping lines with Garage Area greater then 1200 feet
outlier_train = outlier_train[outlier_train['GarageArea'] < 1200]

## Dropping lines with Basement Area greater then 3000 feet
outlier_train = outlier_train[outlier_train['TotalBsmtSF'] < 3000]

## Dropping lines with 1st floor area greater then 2500 feet
outlier_train = outlier_train[outlier_train['1stFlrSF'] < 2500]
## Group GarageCars into 3 clusters: 0, 1, 2 or 4, 3 cars
def encode_garage_cars(x):
    return 'No_cars' if x == 0 else 'One_car' if x == 1 else 'Two_or_four_cars' if (x == 2) | (x == 4) else 'Three_cars'

f = 'GarageCars'
if (f in outlier_train.columns) & (f in outlier_test.columns):
    column_name = 'enc_' + f
    outlier_train[column_name] = outlier_train[f].apply(encode_garage_cars)
    outlier_test[column_name] = outlier_test[f].apply(encode_garage_cars)
    outlier_train = pd.get_dummies(outlier_train, columns=[column_name])
    outlier_test = pd.get_dummies(outlier_test, columns=[column_name])
    outlier_train = outlier_train.drop([f], axis=1)
    outlier_test = outlier_test.drop([f], axis=1)
outlier_train.columns
## Encoding OverallCond into 6 clusters
def encode_overall_condition(x):
    return 'Bellow_6' if x >= 6 else str(x)

f = 'OverallCond'
if (f in outlier_train.columns) & (f in outlier_test.columns):
    column_name = 'enc_' + f
    outlier_train[column_name] = outlier_train[f].apply(encode_overall_condition)
    outlier_test[column_name] = outlier_test[f].apply(encode_overall_condition)
    outlier_train = pd.get_dummies(outlier_train, columns=[column_name])
    outlier_test = pd.get_dummies(outlier_test, columns=[column_name])
    outlier_train = outlier_train.drop([f], axis=1)
    outlier_test = outlier_test.drop([f], axis=1)
outlier_train.columns
print('Dropping other less correlated columns')
cols_to_drop = ['YrSold', 'MSSubClass', 'KitchenAbvGr', 'LowQualFinSF', 'EnclosedPorch']
cols = []
for c in cols_to_drop:
    if (c in outlier_train) & (c in outlier_test):
        cols.append(c)
outlier_train = outlier_train.drop(cols, axis=1)
outlier_test = outlier_test.drop(cols, axis=1)

print('Handle Id columns of both train and test dataset\nDropping Id column from train dataset and making a copy of Id column of test dataset')
outlier_train = outlier_train.drop(['Id'], axis=1)
ids = outlier_test.Id
outlier_test = outlier_test.drop(['Id'], axis=1)
print('Visualizing final shape after transformations\n\n')
print('Train dataset')
print(outlier_train.dtypes.value_counts())
print('\nTest dataset')
print(outlier_test.dtypes.value_counts())
print('\nNull values')
print(missing_data)
tmp = outlier_train.drop('SalePrice',axis=1)
print('Train and test dataset are different\n Train set has more features then test set')
print('Train #features ', tmp.shape[1], ' Test #features ', outlier_test.shape[1])
print("Let's lookup what are those missing features")
missing = list(set(tmp.columns) - set(outlier_test.columns))
print(missing)
for f in missing:
    outlier_test[f] = 0
y = np.log(outlier_train.SalePrice)
X = outlier_train.drop(['SalePrice'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=37, test_size=.33)
lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)
r_2 = model.score(X_test, y_test)

print('r-squared ', r_2)
predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions)

print('RMSE is ', rmse)
actual_values = y_test
plt.figure(figsize= (16, 8))
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.figure(figsize= (16, 8))
    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
submission = pd.DataFrame()
submission['Id'] = ids

## Predict regulated fit
alpha=10
rm = linear_model.Ridge(alpha=alpha)
ridge_model = rm.fit(X_train, y_train)
predictions = ridge_model.predict(outlier_test.interpolate())

## Reverse log transformation
final_predictions = np.exp(predictions)
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])

submission['SalePrice'] = final_predictions
submission.head()

timestr = time.strftime("submission_%Y%m%d_%H%M%S.csv")
print(timestr)
submission.to_csv(timestr, index=False)