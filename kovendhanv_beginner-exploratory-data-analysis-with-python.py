#importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#Load the train data in dataframe
df_train = pd.read_csv('../input/train.csv')
#Display the columns in training set
df_train.columns
df_train.head()
df_train.describe()
df_train.info()
#Analysing Missing Values (NA)
total_na = df_train.isnull().sum().sort_values(ascending=False)
percent_na = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
na_df = pd.concat([total_na, percent_na], axis=1, keys=['No of Missing Values', '% of Missing Values']).sort_values(by= '% of Missing Values', ascending = False)
na_df.head(20)
na_df[na_df['% of Missing Values'] > 0.4]
na_df.loc[['MasVnrType', 'MasVnrArea','Electrical']]
#dealing with missing data
df_train = df_train.drop((na_df[na_df['% of Missing Values'] > 0.4]).index,1)
df_train = df_train.drop((na_df.loc[['MasVnrType', 'MasVnrArea','Electrical']]).index,1)
na_df.isnull().sum().max() #Checking for any missed out NAs
# na_df.head(20)
#Filling NA for other Missing Values with Mean values
df_train['LotFrontage'].fillna(value = df_train['LotFrontage'].mean, inplace = True)
df_train['GarageCond'].fillna(value = df_train['GarageCond'].mean, inplace = True)
df_train['GarageType'].fillna(value = df_train['GarageType'].mean, inplace = True)
df_train['GarageFinish'].fillna(value = df_train['GarageFinish'].mean, inplace = True)
df_train['GarageQual'].fillna(value = df_train['GarageQual'].mean, inplace = True)
df_train['GarageYrBlt'].fillna(value = df_train['GarageYrBlt'].mean, inplace = True)
df_train['BsmtExposure'].fillna(value = df_train['BsmtExposure'].mean, inplace = True)
df_train['BsmtFinType2'].fillna(value = df_train['BsmtFinType2'].mean, inplace = True)
df_train['BsmtFinType1'].fillna(value = df_train['BsmtFinType1'].mean, inplace = True)
df_train['BsmtCond'].fillna(value = df_train['BsmtCond'].mean, inplace = True)
df_train['BsmtQual'].fillna(value = df_train['BsmtQual'].mean, inplace = True)
sns.boxplot(x = df_train['SalePrice'])
#standardizing data
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#bivariate analysis saleprice/grlivarea
sns.jointplot(x = 'GrLivArea', y = 'SalePrice', data = df_train, kind = 'reg');
#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea
sns.jointplot(x = 'TotalBsmtSF', y = 'SalePrice', data = df_train, kind = 'reg');
#descriptive statistics summary
df_train['SalePrice'].describe()
#histogram
sns.distplot(df_train['SalePrice'], fit = norm);

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#Normal probability plot
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#Creating a new column for category variable
#if area>0 then 1, else if area==0 then 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
sns.lmplot(x=var, y='SalePrice', markers = 'x', data = df_train)
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
sns.lmplot(x=var, y='SalePrice', markers = 'x', fit_reg = True, data = df_train)

#box plot overallqual/saleprice
var = 'OverallQual'
f, ax = plt.subplots(figsize=(10, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=df_train)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
f, ax = plt.subplots(figsize=(18, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=df_train)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#Heatmap from Correlation Matrix for all the variables in dataset
corr_mat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, vmax=.8, square=True);
#STRONG POSITIVELY CORRELATED
corr_mat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat[corr_mat > 0.7], vmax=.8, annot = True, square=True);
#STRONG NEGATIVELY CORRELATED
corr_mat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat[corr_mat < -0.3], vmax=.8, annot = True, square=True);
# sns.heatmap(corr_mat, mask = corr_mat < -0.4, vmax=.8, annot = True, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corr_mat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
mask = np.zeros_like(cm)
mask[np.triu_indices_from(mask)] = True
hm = sns.heatmap(cm, cbar=True, mask = mask, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set(palette = 'deep')
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
#applying log transformation
df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])
sns.distplot(df_train['SalePrice_Log'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice_Log'], plot=plt)
#standardizing data
totalBsmtSF_scaled = StandardScaler().fit_transform(df_train['TotalBsmtSF'][:,np.newaxis]);
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
sns.distplot(df_train['TotalBsmtSF'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
