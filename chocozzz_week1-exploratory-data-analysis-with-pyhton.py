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
print("train.csv. Shape: ",df_train.shape)
print("test.csv. Shape: ",df_test.shape)
#descriptive statistics summary
df_train['SalePrice'].describe()
#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#saleprice correlation matrix
k = 10 #number of variables for heatmap
corrmat = df_train.corr(method='spearman') # correlation 전체 변수에 대해서 계산
cols = corrmat.nlargest(k, 'SalePrice').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(df_train[cols].values.T) # correlation 특정 컬럼에 대해서
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(8, 6))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath','TotRmsAbvGrd','YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='GrLivArea', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['GarageCars']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['GarageArea']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='GarageArea', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([(df_train[df_train['GarageArea'] > 0])['SalePrice'], (df_train[df_train['GarageArea'] > 0])['GarageArea']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='GarageArea', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['TotalBsmtSF']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='TotalBsmtSF', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['1stFlrSF']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='1stFlrSF', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['FullBath']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='FullBath', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['TotRmsAbvGrd']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='TotRmsAbvGrd', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='YearBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['YearRemodAdd'])
data = pd.concat([df_train['SalePrice'], df_train['YearBuilt'], df_train['YearRemodAdd']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
data['YearRemodBuilt'] = data['YearRemodAdd'] - data['YearBuilt']
fig = sns.regplot(x='YearRemodBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
data = data[data['YearRemodBuilt'] > 1]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='YearRemodBuilt', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%)", fontsize = 20)
data = pd.concat([df_train['SalePrice'], df_train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
df_train[df_train['OverallQual'] == 4][df_train['SalePrice'] > 200000]
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
df_train = df_train[df_train['Id'] != 458]
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
data = pd.concat([df_train['SalePrice'], df_train['GarageCars']], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x='GarageCars', y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
df_train[df_train['GarageCars'] == 4]
df_test[df_test['GarageCars'] == 4]
df_train[df_train['GarageCars'] == 4]['Neighborhood'].unique()
df_train[df_train['Neighborhood'] == 'Mitchel']['SalePrice'].describe()
df_train[df_train['Neighborhood'] == 'OldTown']['SalePrice'].describe()
df_train[(df_train['Neighborhood'] == 'OldTown') & (df_train['SalePrice'] > 400000)]
df_train = df_train[df_train['Id']!=186]
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
df_train = df_all[:len_train]
df_test = df_all[len_train:]
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)