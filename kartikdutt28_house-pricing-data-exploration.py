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
test=pd.read_csv('../input/test.csv')
train=pd.read_csv('../input/train.csv')
train.columns
train.describe()
train.shape
train.head()
train['SalePrice'].describe()
sns.distplot(train['SalePrice'],fit=norm)
print("Skewness : ")
print(train['SalePrice'].skew())
print("Kurtness: ")
print(train['SalePrice'].kurt())
GrVsSp=pd.concat([train['SalePrice'],train['GrLivArea']],axis=1)
GrVsSp.plot.scatter(x='GrLivArea',y='SalePrice')
TbVsSp=pd.concat([train['TotalBsmtSF'],train['SalePrice']],axis=1)
TbVsSp.plot.scatter(x='TotalBsmtSF',y='SalePrice')
QualVsSp=pd.concat([train['OverallQual'],train['SalePrice']],axis=1)
sns.boxplot(x='OverallQual',y='SalePrice',data=QualVsSp)
corrM=train.corr()
plt.subplots(figsize=(10,8))
sns.heatmap(corrM,xticklabels=corrM.columns,yticklabels=corrM.columns,cmap="YlGnBu")
corrM=train.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corrM,xticklabels=corrM.columns,yticklabels=corrM.columns,annot=True, fmt=".2f",cmap="YlGnBu")
quantitative=list()
for col in train:
    if train[col].dtypes!='object':
        quantitative.append(col)
quantitative
sns.set()
#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(train[cols], size = 5.0)
plt.show();
#Total missing values
total=train.isnull().sum().sort_values(ascending=False)
percent=(train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],keys=['TotalMissingValues','Percentage of missing values'],axis=1)
missing_data.head(20)
train = train.drop((missing_data[missing_data['TotalMissingValues'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()
plt.hist(train['SalePrice'])
Qual = pd.crosstab(index=train["OverallQual"],  # Make a crosstab
                        columns="count")               # Name the count column

Qual 
qualitative=list()
for col in train:
    if train[col].dtypes=='object':
        qualitative.append(col)
qualitative
Mz = pd.crosstab(index=train["MSZoning"],  # Make a crosstab
                        columns="count")               # Name the count column

Mz 
x=list(Mz.index)
y=[10,65,16,1150,218]
plt.bar(x, y, align='center', alpha=0.5)
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();
sns.distplot(train['SalePrice'],fit=norm)
fig=plt.figure()
stats.probplot(train['SalePrice'],plot=plt)
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'],fit=norm)
fig=plt.figure()
stats.probplot(train['SalePrice'],plot=plt)
sns.distplot(train['GrLivArea'],fit=norm)
fig=plt.figure()
stats.probplot(train['GrLivArea'],plot=plt)
train['GrLivArea'] = np.log(train['GrLivArea'])
sns.distplot(train['GrLivArea'],fit=norm)
fig=plt.figure()
stats.probplot(train['GrLivArea'],plot=plt)