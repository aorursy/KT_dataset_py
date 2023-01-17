import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import gc
#bring in the six packs
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
y_reg = df_train['SalePrice']
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]
#saleprice correlation matrix
k = 10 #number of variables for heatmap
corrmat = df_train.corr(method='spearman') # correlation 전체 변수에 대해서 계산
cols = corrmat.nlargest(k, 'OverallQual').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(df_train[cols].values.T) # correlation 특정 컬럼에 대해서
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(16, 10))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
print("Variable","       value  ", "     mean","      ","   0.75Q")
print("YearBuilt:    ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['YearBuilt'].values,
      "",df_train[df_train['OverallQual'] == 4]['YearBuilt'].mean(),
     "",df_train[df_train['OverallQual'] == 4]['YearBuilt'].quantile(0.75))
print("GarageCars:   ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['GarageCars'].values,
      "   ",df_train[df_train['OverallQual'] == 4]['GarageCars'].mean(),
     "",df_train[df_train['OverallQual'] == 4]['GarageCars'].quantile(0.75))
print("GrLivArea:    ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['GrLivArea'].values,
      "",df_train[df_train['OverallQual'] == 4]['GrLivArea'].mean(),
     "",df_train[df_train['OverallQual'] == 4]['GrLivArea'].quantile(0.75))
print("FullBath:     ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['FullBath'].values,
      "   ",df_train[df_train['OverallQual'] == 4]['FullBath'].mean(),
     "",df_train[df_train['OverallQual'] == 4]['FullBath'].quantile(0.75))
print("YearRemodAdd: ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['YearRemodAdd'].values,
      "",df_train[df_train['OverallQual'] == 4]['YearRemodAdd'].mean(),
     "            ",df_train[df_train['OverallQual'] == 4]['YearRemodAdd'].quantile(0.75))
print("GarageArea:   ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['GarageArea'].values,
      " ",df_train[df_train['OverallQual'] == 4]['GarageArea'].mean(),
     " ",df_train[df_train['OverallQual'] == 4]['GarageArea'].quantile(0.75))
print("TotalBsmtSF:  ",df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]['TotalBsmtSF'].values,
      "",df_train[df_train['OverallQual'] == 4]['TotalBsmtSF'].mean(),
     " ",df_train[df_train['OverallQual'] == 4]['TotalBsmtSF'].quantile(0.75))
df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]
df_train = df_train[df_train['Id'] != 457]
df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]
print("Variable","       value  ", "     mean","      ","   0.75Q")
print("YearBuilt:    ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['YearBuilt'].values,
      "",df_train[df_train['OverallQual'] == 8]['YearBuilt'].mean(),
     "",df_train[df_train['OverallQual'] == 8]['YearBuilt'].quantile(0.75))
print("GarageCars:   ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['GarageCars'].values,
      "   ",df_train[df_train['OverallQual'] == 8]['GarageCars'].mean(),
     "",df_train[df_train['OverallQual'] == 8]['GarageCars'].quantile(0.75))
print("GrLivArea:    ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['GrLivArea'].values,
      "",df_train[df_train['OverallQual'] == 8]['GrLivArea'].mean(),
     "",df_train[df_train['OverallQual'] == 8]['GrLivArea'].quantile(0.75))
print("FullBath:     ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['FullBath'].values,
      "   ",df_train[df_train['OverallQual'] == 8]['FullBath'].mean(),
     "",df_train[df_train['OverallQual'] == 8]['FullBath'].quantile(0.75))
print("YearRemodAdd: ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['YearRemodAdd'].values,
      "",df_train[df_train['OverallQual'] == 8]['YearRemodAdd'].mean(),
     "",df_train[df_train['OverallQual'] == 8]['YearRemodAdd'].quantile(0.75))
print("GarageArea:   ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['GarageArea'].values,
      " ",df_train[df_train['OverallQual'] == 8]['GarageArea'].mean(),
     " ",df_train[df_train['OverallQual'] == 8]['GarageArea'].quantile(0.75))
print("TotalBsmtSF:  ",df_train[df_train['OverallQual'] == 8][df_train['SalePrice'] > 500000]['TotalBsmtSF'].values,
      "",df_train[df_train['OverallQual'] == 8]['TotalBsmtSF'].mean(),
     "",df_train[df_train['OverallQual'] == 8]['TotalBsmtSF'].quantile(0.75))
df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]
print("Variable","       value  ", "     mean","      ","   0.25Q")
print("YearBuilt:    ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['YearBuilt'].values,
      "",df_train[df_train['OverallQual'] == 10]['YearBuilt'].mean(),
     " ",df_train[df_train['OverallQual'] == 10]['YearBuilt'].quantile(0.25))
print("GarageCars:   ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['GarageCars'].values,
      "   ",df_train[df_train['OverallQual'] == 10]['GarageCars'].mean(),
     " ",df_train[df_train['OverallQual'] == 10]['GarageCars'].quantile(0.25))
print("GrLivArea:    ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['GrLivArea'].values,
      "",df_train[df_train['OverallQual'] == 10]['GrLivArea'].mean(),
     "",df_train[df_train['OverallQual'] == 10]['GrLivArea'].quantile(0.25))
print("FullBath:     ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['FullBath'].values,
      "   ",df_train[df_train['OverallQual'] == 10]['FullBath'].mean(),
     "",df_train[df_train['OverallQual'] == 10]['FullBath'].quantile(0.25))
print("YearRemodAdd: ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['YearRemodAdd'].values,
      "",df_train[df_train['OverallQual'] == 10]['YearRemodAdd'].mean(),
     "",df_train[df_train['OverallQual'] == 10]['YearRemodAdd'].quantile(0.25))
print("GarageArea:   ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['GarageArea'].values,
      "",df_train[df_train['OverallQual'] == 10]['GarageArea'].mean(),
     " ",df_train[df_train['OverallQual'] == 10]['GarageArea'].quantile(0.25))
print("TotalBsmtSF:  ",df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 180000]['TotalBsmtSF'].values,
      "",df_train[df_train['OverallQual'] == 10]['TotalBsmtSF'].mean(),
     " ",df_train[df_train['OverallQual'] == 10]['TotalBsmtSF'].quantile(0.25))
var = 'SaleCondition'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
var = 'SaleCondition'
data = pd.concat([df_train[df_train['OverallQual'] == 10]['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
var = 'MSZoning'
data = pd.concat([df_train[df_train['OverallQual'] == 10]['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] < 200000]
df_train = df_train[df_train['Id'] != 524][df_train['Id'] != 1299]
var = 'Neighborhood'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
df_train[df_train['Neighborhood'] == 'Edwards']['SalePrice'].describe()
df_train[df_train['OverallQual'] == 10][df_train['SalePrice'] > 700000]
df_train = df_train[df_train['Id'] != 692][df_train['Id'] != 1183]
df_train[df_train['Neighborhood'] == 'NoRidge']['SalePrice'].describe()
#FireplaceQu
var = 'BsmtQual'
data = pd.concat([df_train[df_train['OverallQual'] == 10]['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
xt = plt.xticks(rotation=45)
from scipy import stats
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot('GrLivArea','SalePrice', kind="reg",stat_func=r2, data=data,height =16)
#fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['GarageCars']], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
df_train[df_train['GarageCars'] == 4]
df_test[df_test['GarageCars'] == 4]
df_train[df_train['Neighborhood'] == 'Mitchel']['SalePrice'].describe()
df_train[df_train['Neighborhood'] == 'OldTown']['SalePrice'].describe()
df_train[(df_train['Neighborhood'] == 'OldTown') & (df_train['SalePrice'] > 400000)]
from scipy import stats
data = pd.concat([df_train['SalePrice'], df_train['GarageArea']], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot('GarageArea','SalePrice', kind="reg",stat_func=r2, data=data,height =18)
#fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
#f, ax = plt.subplots(figsize=(8, 6))
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot('GrLivArea','SalePrice', kind="reg",stat_func=r2, data=data,height =18)
#fig.axis(ymin=0, ymax=800000);
df_train[df_train['GrLivArea'] < 3000][df_train["SalePrice"] > 600000]
df_train[df_train['Neighborhood'] == 'NridgHt']['SalePrice'].describe()
df_train = df_train[df_train['Id'] != 899]
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (18,16), fontsize = 15)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Percent of Missing Value (%)", fontsize = 20)
#plt.title("Total Missing Value (%)", fontsize = 20)
import missingno as msno
len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
missingdata_df = df_all.columns[df_all.isnull().any()].tolist()
msno.heatmap(df_all[missingdata_df], figsize=(20,20))
#We impute them by proceeding sequentially through features with missing values

#PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.

df_all["PoolQC"] = df_all["PoolQC"].fillna("None")
df_all["PoolQC"].describe()
df_all[(df_all["PoolQC"] == 'None') & (df_all["PoolArea"] > 0)][["Id","PoolQC","PoolArea","OverallQual"]]
df_all.loc[df_all['Id'] == 2421, ['PoolQC']] = 'TA'
df_all.loc[df_all['Id'] == 2504, ['PoolQC']] = 'Gd'
df_all.loc[df_all['Id'] == 2600, ['PoolQC']] = 'Fa'
df_all[(df_all["PoolQC"] == 'None') & (df_all["PoolArea"] > 0)][["Id","PoolQC","PoolArea","OverallQual"]]
df_all["PoolQC"].describe()
df_all["MiscFeature"] = df_all["MiscFeature"].fillna("None")
df_all["Alley"] = df_all["Alley"].fillna("None")
df_all["Fence"] = df_all["Fence"].fillna("None")
df_all["FireplaceQu"] = df_all["FireplaceQu"].fillna("None")
len(df_all[df_all["LotFrontage"].isnull()])
df_all["LotFrontage"] = df_all.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
len(df_all[df_all["LotFrontage"].isnull()])
df_all['GarageYrBlt'] = df_all.fillna(df_all['YearBuilt'])
df_all[df_all['GarageType'].isnull()][['GarageCond','GarageFinish','GarageQual']].head(10)
df_all[((df_all['GarageType'].isnull()) == False) & ((df_all['GarageFinish'].isnull()) == True)][['Id','GarageCars', 'GarageArea', 'GarageType', 'GarageCond', 'GarageQual', 'GarageFinish']]
print("GarageCond: ", df_all[(df_all['GarageType']=='Detchd') & (df_all['GarageCond'] != "nan")]['GarageCond'].mode().values)
print("GarageQual: ", df_all[(df_all['GarageType']=='Detchd') & (df_all['GarageQual'] != "nan")]['GarageQual'].mode().values)
print("GarageFinish: ", df_all[(df_all['GarageType']=='Detchd') & (df_all['GarageFinish'] != "nan")]['GarageFinish'].mode().values)
df_all.loc[df_all['Id'] == 2127, ['GarageCond']] = 'TA'
df_all.loc[df_all['Id'] == 2127, ['GarageQual']] = 'TA'
df_all.loc[df_all['Id'] == 2127, ['GarageFinish']] = 'Unf'
df_all[df_all["Id"]==2127][['GarageCond','GarageQual','GarageFinish']]
df_all.loc[df_all['Id'] == 2577, ['GarageCars']] = 0
df_all.loc[df_all['Id'] == 2577, ['GarageArea']] = 0
df_all.loc[df_all['Id'] == 2577, ['GarageType']] = 'None'
df_all[df_all["Id"]==2577][['GarageCars','GarageArea','GarageType']]
df_all['GarageType'] = df_all['GarageType'].fillna('None')
df_all['GarageFinish'] = df_all['GarageFinish'].fillna('None')
df_all['GarageQual'] = df_all['GarageQual'].fillna('None')
df_all['GarageCond'] = df_all['GarageCond'].fillna('None')
df_all[((df_all["BsmtFinType1"].isnull())==False) & ((df_all["BsmtCond"].isnull()) | (df_all["BsmtQual"].isnull()) | (df_all["BsmtExposure"].isnull()) | (df_all["BsmtFinType2"].isnull()))][['Id','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']]
print("BsmtFinType2 mode",df_all['BsmtFinType2'].mode().values,"\nBsmtExposure mode",df_all['BsmtExposure'].mode().values,"\nBsmtCond mode",df_all['BsmtCond'].mode().values,"\nBsmtQual mode",df_all['BsmtQual'].mode().values)
df_all.loc[df_all['Id'] == 333, ['BsmtFinType2']] = 'Unf'
df_all.loc[(df_all['Id'] == 949),['BsmtExposure']] = 'No';df_all.loc[(df_all['Id'] == 1488),['BsmtExposure']] = 'No';df_all.loc[(df_all['Id'] == 2349),['BsmtExposure']] = 'No'
df_all.loc[(df_all['Id'] == 2041), ['BsmtCond']] = 'Unf';df_all.loc[(df_all['Id'] == 2186), ['BsmtCond']] = 'Unf';df_all.loc[(df_all['Id'] == 2525), ['BsmtCond']] = 'Unf'
df_all.loc[(df_all['Id'] == 2218), ['BsmtQual']] = 'Unf';df_all.loc[(df_all['Id'] == 2219), ['BsmtQual']] = 'Unf'
df_all['BsmtQual'] = df_all['BsmtQual'].fillna('None')
df_all['BsmtCond'] = df_all['BsmtCond'].fillna('None')
df_all['BsmtExposure'] = df_all['BsmtExposure'].fillna('None')
df_all['BsmtFinType1'] = df_all['BsmtFinType1'].fillna('None')
df_all['BsmtFinType2'] = df_all['BsmtFinType2'].fillna('None')
df_all[(df_all["BsmtFullBath"].isnull()) & ((df_all["BsmtHalfBath"].isnull()) | (df_all["BsmtFinSF1"].isnull()) | (df_all["BsmtFinSF2"].isnull()) | (df_all["BsmtUnfSF"].isnull())| (df_all["TotalBsmtSF"].isnull()) )][['Id','BsmtQual', 'BsmtQual', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']]
df_all['BsmtFullBath'] = df_all['BsmtFullBath'].fillna(0)
df_all['BsmtHalfBath'] = df_all['BsmtHalfBath'].fillna(0)
df_all['BsmtFinSF1'] = df_all['BsmtFinSF1'].fillna(0)
df_all['BsmtFinSF2'] = df_all['BsmtFinSF2'].fillna(0)
df_all['BsmtUnfSF'] = df_all['BsmtUnfSF'].fillna(0)
df_all['TotalBsmtSF'] = df_all['TotalBsmtSF'].fillna(0)
df_all[(df_all['MasVnrType'].isnull()) & (df_all['MasVnrArea'].isnull() == False ) ][['Id','MasVnrType','MasVnrArea']]
#FireplaceQu
var = 'MasVnrArea'
data = pd.concat([df_all['MasVnrType'], df_all[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x='MasVnrType', y="MasVnrArea", data=data)
xt = plt.xticks(rotation=45)
df_all.loc[df_all['Id'] == 2611, ['MasVnrType']] = 'Stone'
df_all[df_all['Id']==2611]['MasVnrType']
df_all[df_all['MasVnrType'].isnull() == True]['MasVnrType'].head()
df_all['MasVnrType'] = df_all['MasVnrType'].fillna('None')
df_all['MasVnrArea'] = df_all['MasVnrArea'].fillna(0)
len(df_all[df_all['MasVnrArea'].isnull()])
df_all['MSZoning'].describe()
df_all['MSZoning'] = df_all['MSZoning'].fillna('RL')
df_all['KitchenQual'].describe()
df_all['KitchenQual'] = df_all['KitchenQual'].fillna('TA')
df_all['Utilities'].describe()
del df_all['Utilities'];
gc.collect()
df_all['Functional'].describe()
df_all['Functional'] = df_all['Functional'].fillna('Typ')
df_all['Exterior1st'].describe()
df_all['Exterior1st'] = df_all['Exterior1st'].fillna('VinylSd')
df_all['Exterior2nd'].describe()
df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna('VinylSd')
df_all['Electrical'].describe()
df_all['Electrical'] = df_all['Electrical'].fillna('SBrkr')
df_all['SaleType'].describe()
df_all['SaleType'] = df_all['SaleType'].fillna('WD')
#missing data
total = df_all.isnull().sum().sort_values(ascending=False)
percent = (df_all.isnull().sum()/df_all.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (18,16), fontsize = 15)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Percent of Missing Value (%)", fontsize = 20)
#plt.title("Total Missing Value (%)", fontsize = 20)
import scipy.stats as st

y = df_train['SalePrice']

#plt.figure(1); plt.title('Johnson SU')
#sns.distplot(y, kde=True, fit=st.johnsonsu)
plt.figure(1); plt.title('Normal')
sns.distplot(y, kde=True, fit=st.norm)
plt.figure(1)
res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.figure(2)
res = stats.probplot(np.log1p(df_train['SalePrice']), plot=plt)
var = 'SaleCondition'
df_train['SaleCondition'].unique()
df_train.shape
from sklearn.preprocessing import OneHotEncoder
one_hot_encoding = df_train.copy()
pd.get_dummies(one_hot_encoding['SaleCondition']).head()
label_encoding = df_train.copy()
label_encoding['SaleCondition'], indexer = pd.factorize(label_encoding['SaleCondition'])
df_test['SaleCondition'] = indexer.get_indexer(df_test['SaleCondition'])
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

len_train_frequency = df_train.shape[0]
df_all_frequency = pd.concat([df_train, df_test])

df_all_frequency_ex = df_all_frequency.copy()
categorical_features = ['SaleCondition']
for col in categorical_features:
    df_all_frequency_ex = frequency_encoding(df_all_frequency, col)
df_all_frequency_ex['SaleCondition_Frequency'].head()
from sklearn.model_selection import KFold

def mean_k_fold_encoding(col, alpha):
    target_name = 'SalePrice'
    target_mean_global = df_train[target_name].mean()
    
    nrows_cat = df_train.groupby(col)[target_name].count()
    target_means_cats = df_train.groupby(col)[target_name].mean()
    target_means_cats_adj = (target_means_cats*nrows_cat + 
                             target_mean_global*alpha)/(nrows_cat+alpha)
    # Mapping means to test data
    encoded_col_test = df_test[col].map(target_means_cats_adj)
    #임의로 추가 한 부분
    encoded_col_test.fillna(target_mean_global, inplace=True)
    encoded_col_test.sort_index(inplace=True)

    kfold = KFold(n_splits=5, shuffle=True, random_state=1989)
    parts = []
    for trn_inx, val_idx in kfold.split(df_train):
        df_for_estimation, df_estimated = df_train.iloc[trn_inx], df_train.iloc[val_idx]
        nrows_cat = df_for_estimation.groupby(col)[target_name].count()
        target_means_cats = df_for_estimation.groupby(col)[target_name].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat + 
                                target_mean_global * alpha) / (nrows_cat + alpha)

        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        parts.append(encoded_col_train_part)
        
    encoded_col_train = pd.concat(parts, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)
    encoded_col_train.sort_index(inplace=True)
    
    return encoded_col_train, encoded_col_test
df_all['GarageYrBlt'] = df_all['GarageYrBlt'].astype('int8')
categorical_features = df_all.select_dtypes(include = ["object"]).columns
numerical_features = df_all.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
df_all.shape
from sklearn.preprocessing import OneHotEncoder
one_hot_encoding = df_all.copy()
one_hot_encoding = pd.get_dummies(one_hot_encoding)
#len_train
one_hot_encoding.head()
one_hot_encoding.shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
%matplotlib inline
import os
one_hot_encoding_train = one_hot_encoding[:len_train]
one_hot_encoding_test = one_hot_encoding[len_train:]

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = one_hot_encoding_train['SalePrice']
del one_hot_encoding_train['SalePrice']

if 'SalePrice' in one_hot_encoding_test.columns:
    del one_hot_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = one_hot_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=one_hot_encoding_train, n_splits=5)

train_features = [_f for _f in one_hot_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(one_hot_encoding_train.shape[0])
sub_reg_preds = np.zeros(one_hot_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = one_hot_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = one_hot_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(one_hot_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
one_hot_encoding_train['SalePrice'] = y_reg
one_hot_encoding_train.to_csv('one_hot_encoding_train.csv', index=False)
one_hot_encoding_test.to_csv('one_hot_encoding_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("one_hot_encoding_model.csv", index=False) # submission
categorical_features
label_encoding = df_all.copy()
for i in categorical_features:
    label_encoding[i], indexer = pd.factorize(label_encoding[i])
label_encoding.head()
label_encoding_train = label_encoding[:len_train]
label_encoding_test = label_encoding[len_train:]

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = label_encoding_train['SalePrice']
del label_encoding_train['SalePrice']

if 'SalePrice' in label_encoding_test.columns:
    del label_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = label_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=label_encoding_train, n_splits=5)

train_features = [_f for _f in label_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(label_encoding_train.shape[0])
sub_reg_preds = np.zeros(label_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = label_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = label_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(label_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
label_encoding_train['SalePrice'] = y_reg
label_encoding_train.to_csv('label_encoding_train.csv', index=False)
label_encoding_test.to_csv('label_encoding_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("label_encoding_model.csv", index=False) # submission
frequency_encoding_all = df_all.copy()
    
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

for col in categorical_features:
    frequency_encoding_all = frequency_encoding(frequency_encoding_all, col)
frequency_encoding_all.head()
frequency_encoding_all = frequency_encoding_all.drop(categorical_features,axis=1, inplace=False)
frequency_encoding_all.head()
frequency_encoding_train = frequency_encoding_all[:len_train]
frequency_encoding_test = frequency_encoding_all[len_train:]

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = frequency_encoding_train['SalePrice']
del frequency_encoding_train['SalePrice']

if 'SalePrice' in frequency_encoding_test.columns:
    del frequency_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = frequency_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=frequency_encoding_train, n_splits=5)

train_features = [_f for _f in frequency_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(frequency_encoding_train.shape[0])
sub_reg_preds = np.zeros(frequency_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = frequency_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = frequency_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(frequency_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
frequency_encoding_train['SalePrice'] = y_reg

frequency_encoding_train.to_csv('frequency_encoding_train.csv', index=False)
frequency_encoding_test.to_csv('frequency_encoding_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("frequency_encoding_model.csv", index=False) # submission
mean_encoding = df_all.copy()
mean_encoding_train = mean_encoding[:len_train]
mean_encoding_test = mean_encoding[len_train:]
del mean_encoding
#del df_all; gc.collect()
for col in categorical_features:
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    mean_encoding_train[new_feat_name] = temp_encoded_tr.values
    mean_encoding_test[new_feat_name] = temp_encoded_te.values
    
mean_encoding_train = mean_encoding_train.drop(categorical_features, axis=1, inplace=False)
mean_encoding_test = mean_encoding_test.drop(categorical_features, axis=1, inplace=False)
mean_encoding_train.head()
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = mean_encoding_train['SalePrice']
del mean_encoding_train['SalePrice']

if 'SalePrice' in mean_encoding_test.columns:
    del mean_encoding_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = mean_encoding_test.Id

sub_reg_preds = 0
folds = get_folds(df=mean_encoding_train, n_splits=5)

train_features = [_f for _f in mean_encoding_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(mean_encoding_train.shape[0])
sub_reg_preds = np.zeros(mean_encoding_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = mean_encoding_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = mean_encoding_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(mean_encoding_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
mean_encoding_train['SalePrice'] = y_reg

mean_encoding_train.to_csv('mean_encoding_train.csv', index=False)
mean_encoding_test.to_csv('mean_encoding_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("mean_encoding_model.csv", index=False) # submission
encoding_all = df_all.copy()
    
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

for col in categorical_features:
    encoding_all = frequency_encoding(encoding_all, col)
encoding_all_train = encoding_all[:len_train]
encoding_all_test = encoding_all[len_train:]
del encoding_all
#del df_all; gc.collect()
for col in categorical_features:
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    encoding_all_train[new_feat_name] = temp_encoded_tr.values
    encoding_all_test[new_feat_name] = temp_encoded_te.values
    
encoding_all = pd.concat([encoding_all_train,encoding_all_test])
encoding_all = pd.get_dummies(encoding_all)
for col in categorical_features:
    encoding_all[col] = list(label_encoding[col].values)
encoding_all_train = encoding_all[:len_train]
encoding_all_test = encoding_all[len_train:]
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = encoding_all_train['SalePrice']
del encoding_all_train['SalePrice']

if 'SalePrice' in encoding_all_test.columns:
    del encoding_all_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = encoding_all_test.Id

sub_reg_preds = 0
folds = get_folds(df=encoding_all_train, n_splits=5)

train_features = [_f for _f in encoding_all_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(encoding_all_train.shape[0])
sub_reg_preds = np.zeros(encoding_all_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = encoding_all_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = encoding_all_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(encoding_all_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
encoding_all_train['SalePrice'] = y_reg

encoding_all_train.to_csv('encoding_all_train.csv', index=False)
encoding_all_test.to_csv('encoding_all_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("encoding_all_model.csv", index=False) # submission
label_columns = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                 'BsmtQual','CentralAir','Electrical','ExterCond','ExterQual','Fence',
                 'FireplaceQu','GarageCond','GarageFinish','GarageQual','GarageYrBlt',
                 'HeatingQC','KitchenQual','LotShape','PoolQC']
for i in label_columns:
    df_all[i], indexer = pd.factorize(df_all[i])
df_all['KitchenQual'].head()
mean_columns = [_f for _f in df_all.columns 
                    if (_f not in label_columns) & (df_all[_f].dtype == 'object')]
print(str(mean_columns))
df_train = df_all[:len_train]
df_test = df_all[len_train:]
del df_all; gc.collect()
for col in mean_columns:
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    df_train[new_feat_name] = temp_encoded_tr.values
    df_test[new_feat_name] = temp_encoded_te.values
df_train = df_train.drop(mean_columns, axis=1, inplace=False)
df_test = df_test.drop(mean_columns, axis=1, inplace=False)
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = df_train['SalePrice']
del df_train['SalePrice']

if 'SalePrice' in df_test.columns:
    del df_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = df_test.Id

sub_reg_preds = 0
folds = get_folds(df=df_train, n_splits=5)

train_features = [_f for _f in df_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(df_train.shape[0])
sub_reg_preds = np.zeros(df_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = df_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = df_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(df_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
df_train['SalePrice'] = y_reg

df_train.to_csv('df_train.csv', index=False)
df_test.to_csv('df_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("label_mean_model.csv", index=False) # submission
fre_label_all = frequency_encoding_all.copy()
for col in categorical_features:
    fre_label_all[col] = list(label_encoding[col].values)
fre_label_all_train = fre_label_all[:len_train]
fre_label_all_test = fre_label_all[len_train:]
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['Id'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['Id'].isin(unique_vis[trn_vis])],
                ids[df['Id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

y_reg = fre_label_all_train['SalePrice']
del fre_label_all_train['SalePrice']

if 'SalePrice' in fre_label_all_test.columns:
    del fre_label_all_test['SalePrice']
    
excluded_features = ['Id','SalePrice'] 
test_idx = fre_label_all_test.Id

sub_reg_preds = 0
folds = get_folds(df=fre_label_all_train, n_splits=5)

train_features = [_f for _f in fre_label_all_train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(fre_label_all_train.shape[0])
sub_reg_preds = np.zeros(fre_label_all_test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = fre_label_all_train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = fre_label_all_train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.005,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(fre_label_all_test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
fre_label_all_train['SalePrice'] = y_reg

fre_label_all_train.to_csv('fre_label_all_train.csv', index=False)
fre_label_all_test.to_csv('fre_label_all_test.csv', index=False)
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["SalePrice"] = sub_reg_preds
test_pred.columns = ["Id", "SalePrice"]
test_pred.to_csv("fre_label_all_model.csv", index=False) # submission
