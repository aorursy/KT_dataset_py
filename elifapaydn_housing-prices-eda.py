import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as ss



pd.options.display.max_rows = 1000

pd.options.display.max_columns = 40



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

data.head()
data.info()
#store categorical and numerical features

catcols=['SalePrice','Id','MSSubClass','OverallQual','OverallCond', 'MoSold','YrSold', 'YearBuilt','YearRemodAdd'] #they are not string

quantitative=[col for col in data.columns if data[col].dtype != 'object' and col not in catcols]

qualitative= [col for col in data.columns if data[col].dtype == 'object']

qualitative.extend(['MSSubClass','OverallQual','OverallCond', 'MoSold','YrSold', 'YearBuilt','YearRemodAdd']) 
print('Categorical Features:\n', qualitative)

print('\n')

print('Numerical Features:\n',quantitative)
data.describe().drop('Id', axis=1)
pd.crosstab(data['Neighborhood'], data['OverallQual'], margins=True).style.background_gradient(cmap='summer_r')
cm = sns.light_palette("green", as_cmap=True)

pd.crosstab(data['OverallQual'], data['YearBuilt']).style.background_gradient(cmap=cm)
#profile=ProfileReport(data, title='Profiling Report')

#profile.to_file('report.html')
# Imputing the missing values

def cat_imputation(col, val):

    data.loc[data[col].isnull(),col] = val
data.head()
nullnb=data.isnull().sum()

null=pd.DataFrame( {"Null Number":nullnb, "Null Perc": (nullnb*100/1460)}, index=nullnb.index)

missing_df=null[ null["Null Number"] > 0]

missing_cols=missing_df.index.tolist()

missing_df=missing_df.sort_values(by='Null Perc', ascending=False)

missing_df
missing_df['Null Perc'].plot.bar(color=['red', 'black'])

plt.ylabel('Null Percentage')
print(data[['PoolArea','PoolQC']].groupby(['PoolArea','PoolQC']).size())

print('\n')

print(data['PoolArea'].value_counts())

print('\n')

print('Null PoolQC records: ', data['PoolQC'].isnull().sum())

cat_imputation('PoolQC', 'None')
df=data[['LotFrontage', 'LotArea']].assign(SqrtLotArea = np.sqrt(data.LotArea))

df=df.assign(differ=df.LotFrontage-df.SqrtLotArea)

df['differ'].mean()
data=data.assign(LotFronEst=np.sqrt(data.LotArea)-24)

print('Correlation between our estimate and true value:',data['LotFrontage'].corr(data['LotFronEst']))

#impute 

data.loc[data.LotFrontage.isnull(),'LotFrontage']=data.loc[data.LotFrontage.isnull(),'LotFronEst']
data.drop('LotFronEst', axis=1, inplace=True)
#when MiscVal=0 there exists no MiscFeature.

data['MiscVal'][data['MiscFeature'].isnull()].value_counts()
cat_imputation('MiscFeature', 'None')

cat_imputation('Alley', 'None')

cat_imputation('MasVnrType', 'None')

cat_imputation('MasVnrArea', 0.0)

cat_imputation('Fence', 'None')

cat_imputation('FireplaceQu', 'None')
#garage features

garagefeat=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

for col in garagefeat:

    if data[col].dtype==np.object:

        cat_imputation(col,'None')

    else:

        cat_imputation(col, 0)

        

#basement features

bsmtfeat=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

for col in bsmtfeat:

    if data[col].dtype==np.object:

        cat_imputation(col,'None')

    else:

        cat_imputation(col, 0)
data.loc[data.Electrical.isnull()==True,'Electrical']=data.Electrical.mode().values[0]
#check the overall number of missing vals

data.isnull().sum().sum()

fig=plt.figure(figsize=(15,12))

ax=plt.gca()

sns.heatmap(data.corr(), annot=True,fmt='.1f', cmap='RdYlBu', ax=ax)
labels = [item.get_text() for item in ax.get_xticklabels()]

top10corr=data[labels].corr().abs().nlargest(10, 'SalePrice')['SalePrice']

top10corr_index=top10corr.index

print(top10corr)
f=plt.figure(figsize=(9,9))

sns.heatmap(data[top10corr_index].corr(), mask=np.triu(data[top10corr_index].corr(), k=0), annot=True, cmap='RdYlBu')
#use for multicol. after cleaning nans

from statsmodels.stats.outliers_influence import variance_inflation_factor

x=data[quantitative].drop(['GrLivArea','TotalBsmtSF','GarageCars'], axis=1).assign(cons=1)

colln=pd.Series([variance_inflation_factor(x.values,i) for i in range(x.shape[1])], index=x.columns)

tempcol=colln.index.tolist() + ['SalePrice']

tempcol.remove('cons')

print('Largest 10 Correlations with target:\n',data[tempcol].corr().nlargest(10, 'SalePrice')['SalePrice'])

print('\nVIF:\n',colln)

print('\nFeatures with multicollinearity: ','GrLivArea','TotalBsmtSF','GarageCars')
meanprices=data[['YearBuilt','SalePrice']].groupby('YearBuilt').mean()

meanquals=data[['YearBuilt','OverallQual']].groupby('YearBuilt').mean()

fig=plt.figure(figsize=(20,8))

ax=fig.add_subplot(111)

ax.bar(meanprices.index, meanprices['SalePrice'], label='Sale Price'

    ,alpha=0.7)

ax2=ax.twinx()

ax2.plot(meanquals, label='Overall Quality', c='r')

ax2.legend()

ax.legend()

ax.set_xlabel('Year Built')

ax.set_ylabel('Sale Price(log)')

ax2.set_ylabel('OverallQual')
fig, axs=plt.subplots(1,2, figsize=(8,4))

sns.distplot(data.SalePrice, fit=ss.norm, ax=axs[0])

sns.distplot(data.SalePrice, fit=ss.lognorm, ax=axs[1])

#log transformation

data['SalePrice']=np.log(data['SalePrice'])
data[['SalePrice', 'GrLivArea']].plot.scatter(x='GrLivArea', y='SalePrice', c='tab:red', edgecolor='w', s=50)
sns.pairplot(data[top10corr_index], palette='Set2', kind='scatter', height=3)
d=pd.melt(data, value_vars=quantitative)

grid=sns.FacetGrid(d, col_wrap=3, col='variable', sharex=False, sharey=False)

grid=grid.map(sns.distplot,"value", kde=False)
d2=pd.melt(data, id_vars='SalePrice', value_vars=qualitative)

g2=sns.FacetGrid(d2, col_wrap=3, col='variable', sharex=False, sharey=False, height=5)

g=g2.map(sns.boxplot, 'value', 'SalePrice', palette="Set2")

for ax in g2.axes.ravel():

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    

plt.tight_layout()
#Facet Grid Plot - FirePlace QC vs.SalePrice

g = sns.FacetGrid(data, col = 'FireplaceQu', col_wrap = 3, col_order=data.FireplaceQu.unique()[1:]) #we added col_order so we wont see None column in our facetgrid

g.map(sns.boxplot, 'Fireplaces', 'SalePrice', order = [1, 2, 3], palette = 'Set3')