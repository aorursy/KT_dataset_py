# NOTE

# tricks learnt from this notebook

#(originally from "https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python")

# Glossory:

#    sns.distplot(data,fit=scipy.stats.norm)

#    data = pd.concat([x,y],axis=1)

#    pd.DataFrame.plot.scatter(x='x',y='y')

#    sns.boxplot(x='x',y='y',data=data)

#    cm = pd.DataFrame.corr()

#    sns.heatmap(cm,vmax=.8,square=True,fmt='.2f',annot=Truen,annot_kws={'size':8})

#    pd.DataFrame.nlargest(n,'column_name')

#    sns.pairplot(data,size=2.5)

#    scipy.stats.probplot(data,plot=matplotlib.pyplot)

#    total = data.isnull().sum()

#    percent = data.isnull().sum()/data.shape[0]

#    missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

#    data.drop(missing_data[missing_data['Percent']>.05].index,1)

# TIPS:

#    big map of numerical vars: sns.heatmap(corr)

#    relation with numerical vars: scatter

#    relation with catogarial vars: sns.boxplot

#    normality visualizaiton: sns.distplot

#    transformation for positive skewness: np.log

#    missingdata: pd.DataFrame.isnull().sum()/pd.DataFrame.shape[0]

#    drop: pd.DataFrame.drop(index); pd.DataFrame.drop()
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import StandardScaler

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_train.columns
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
type(df_train['SalePrice'])
df_train['GrLivArea'].describe()
data = pd.concat([df_train['SalePrice'],df_train['GrLivArea']],axis=1)

data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000))
df_train['TotalBsmtSF'].describe()
data = pd.concat([df_train['TotalBsmtSF'],df_train['SalePrice']],axis=1)

data.plot.scatter(x='TotalBsmtSF',y='SalePrice',ylim=(0,800000))
df_train['OverallQual'].head(10)
data = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis=1)

type(data)
plt.subplots(figsize=(8,7))

fig = sns.boxplot(x='OverallQual',y='SalePrice',data=data)

fig.axis(ymin=0,ymax=800000);
data = pd.concat([df_train['YearBuilt'],df_train['SalePrice']],axis=1)

plt.subplots(figsize=(16,8))

fig = sns.boxplot(x='YearBuilt',y='SalePrice',data=data)

fig.axis(ymin=0,ymax=800000)

plt.xticks(rotation=90);
df_train['Heating'].head(10);

df_train['Heating'].describe()
plt.subplots(figsize=(8,3))

plt.subplot(1,2,1)

data = pd.concat([df_train['Heating'],df_train['SalePrice']],axis=1)

fig = sns.boxplot(x='Heating',y='SalePrice',data=data)

plt.subplot(1,2,2)

data = pd.concat([df_train['LandSlope'],df_train['SalePrice']],axis=1)

fig = sns.boxplot(x='LandSlope',y='SalePrice',data=data)
corrmat = df_train.corr()

corrmat.columns
corrmat.index
corrmat.values.shape
plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,vmax=.8,square=True);
k = 10

cols = corrmat.nlargest(k,'SalePrice').index

#cm = np.corrcoef(df_train[cols].values.T)

cm = df_train[cols].corr()

sns.heatmap(cm,annot=True,square=True,fmt='.2f',annot_kws={'size':10});
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']

sns.pairplot(df_train[cols],size=2.5);
df_train.head(5)

#df_train.tail(5)
total = df_train.isnull().sum(axis=0).sort_values(ascending=False)

total.head(10)

#total.tail(5)
percent = (df_train.isnull().sum()/df_train.shape[0]).sort_values(ascending=False)

missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head(20)
data = pd.concat([df_train['SalePrice'], df_train['MasVnrType']],axis=1)

sns.boxplot(x='MasVnrType',y='SalePrice',data=data)
df_train = df_train.drop(missing_data[missing_data['Total']>1].index,1)

#df_train.loc[df_train['Electrical'].isnull()]

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

plt.plot(saleprice_scaled)

plt.show()
plt.plot(df_train['SalePrice'])

plt.show()
data = pd.concat([df_train['SalePrice'], df_train['GrLivArea']],axis=1)

data.plot.scatter(x='GrLivArea',y='SalePrice',ylim=(0,800000));
sns.distplot(df_train['SalePrice'],fit=stats.norm);

fit = plt.figure()

res = stats.probplot(df_train['SalePrice'],plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice']);

sns.distplot(df_train['SalePrice'],fit=stats.norm);

fig = plt.figure()

stats.probplot(df_train['SalePrice'],plot=plt);
sns.distplot(df_train['GrLivArea'],fit=stats.norm);

plt.figure()

stats.probplot(df_train['GrLivArea'],plot=plt);
df_train['GrLivArea'] = np.log(df_train['GrLivArea']);

sns.distplot(df_train['GrLivArea'],fit=stats.norm);

plt.figure()

stats.probplot(df_train['GrLivArea'],plot=plt);
sns.distplot(df_train['TotalBsmtSF'],fit=stats.norm);

plt.figure()

stats.probplot(df_train['TotalBsmtSF'],plot=plt);
df_train['HasBsmt'] = pd.Series(df_train.shape[0],index=df_train.index)

df_train['HasBsmt'] = 0

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
df_train.loc[df_train['TotalBsmtSF']>0,'TotalBsmtSF'] = np.log(df_train.loc[df_train['TotalBsmtSF']>0,'TotalBsmtSF'])
sns.distplot(df_train.loc[df_train['TotalBsmtSF']>0,'TotalBsmtSF'],fit=stats.norm);

plt.figure()

stats.probplot(df_train.loc[df_train['TotalBsmtSF']>0,'TotalBsmtSF'],plot=plt);