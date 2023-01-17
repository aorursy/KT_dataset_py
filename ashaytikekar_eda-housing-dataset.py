#importing required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as ticker



sns.set(style='darkgrid')
#importing data

df_train = pd.read_csv("../input/train.csv")

df_train.head()
#let's take a look at columns

print(df_train.columns)



#total number of columns

print(len(df_train.columns))

#size of the data at hand

df_train.size
metadata=pd.DataFrame(columns=['Column Name','Type','Missing Values','Unique Values'])



metadata['Column Name'] = df_train.columns



metadata.Type=df_train.dtypes

metadata.Type=list(df_train.dtypes)



misval=df_train.isna().sum()

metadata['Missing Values']=list(misval)



nun=df_train.nunique()

metadata['Unique Values']=list(nun)



metadata['Missing Values %']=metadata['Missing Values']*100/len(df_train)



metadata



#number of columns with missing data

print(len(metadata.loc[metadata['Missing Values %'] >= 1]))



#listing columns with data missing

metadata.loc[metadata['Missing Values %'] >=1]
#dropping columns with more than 30% of the data missing

missing_data = metadata.loc[metadata['Missing Values %'] > 0]["Column Name"]

df_train.drop(missing_data,inplace=True,axis=1)
#dropping column Id as well

df_train.drop("Id",inplace=True,axis=1)
#number of columns we are left with

print("Number of columns left in df_train {}".format(len(df_train.columns)))

#function to create boxplot and distribution plot for columns with numerical data

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

dimens = (20,10)

def create_distribution_plots(column_name,box_log_scale = False,dist_log_scale = False):

    fig, ax = plt.subplots(figsize=dimens,ncols = 2,nrows=1)

    ax[0].set_title('{} Univariate Analysis'.format(column_name))

    ax[0] = sns.boxplot(x=df_train[column_name],orient = 'v',ax=ax[0])

    

    ax[1].set_title('{} Univariate Analysis'.format(column_name))

    ax[1] = sns.distplot(df_train[column_name],ax=ax[1])

    

    if(box_log_scale):

        ax[0].set_yscale('log')

        

    if(dist_log_scale):

        ax[1].set_yscale('log')



# LotArea

create_distribution_plots('LotArea',box_log_scale = True)
# BsmtFinSF1



create_distribution_plots('BsmtFinSF1')
#BsmtFinSF2

create_distribution_plots('BsmtFinSF2')
#BsmtFinSF2



create_distribution_plots('BsmtUnfSF')
#TotalBsmtSF



create_distribution_plots('TotalBsmtSF')
#1stFlrSF



create_distribution_plots('1stFlrSF')
#2ndFlrSF

create_distribution_plots('2ndFlrSF')
#GrLivArea

create_distribution_plots('GrLivArea')
#GarageArea



create_distribution_plots('GarageArea')
#WoodDeckSF

create_distribution_plots('WoodDeckSF')
#OpenPorchSF



create_distribution_plots('OpenPorchSF')
#EnclosedPorch



create_distribution_plots('EnclosedPorch')
#3SsnPorch

create_distribution_plots('3SsnPorch')
#ScreenPorch



create_distribution_plots('ScreenPorch')
#PoolArea



create_distribution_plots('PoolArea')
#SalePrice



create_distribution_plots('SalePrice')
#MSSubClass



sns.countplot(df_train['MSSubClass'])
#OverallQual

sns.countplot(df_train['OverallQual'])
#OverallCond

sns.countplot(df_train['OverallCond'])
#BedroomAbvGr,TotRmsAbvGrd,KitchenAbvGr,Fireplaces

fig, ax = plt.subplots(figsize=(10,10),ncols = 2,nrows=2)

ax[0][0] = sns.countplot(df_train['BedroomAbvGr'],ax=ax[0][0])



ax[0][1] = sns.countplot(df_train['TotRmsAbvGrd'],ax=ax[0][1])



ax[1][0] = sns.countplot(df_train['KitchenAbvGr'],ax=ax[1][0])



ax[1][1] = sns.countplot(df_train['Fireplaces'],ax=ax[1][1])



#FullBath,HalfBath,BsmtFullBath,BsmtHalfBath

fig, ax = plt.subplots(figsize=(10,10),ncols = 2,nrows=2)

ax[0][0] = sns.countplot(df_train['FullBath'],ax=ax[0][0])



ax[0][1] = sns.countplot(df_train['HalfBath'],ax=ax[0][1])



ax[1][0] = sns.countplot(df_train['BsmtFullBath'],ax=ax[1][0])



ax[1][1] = sns.countplot(df_train['BsmtHalfBath'],ax=ax[1][1])
bins = [1872, 1900,1930,1960, 1990, 2000, 2005,2010]

labels = ['before 1900','1900-30','1930-60', '1960-90', '1990-2000', '2000-05', '2005-10']

df_train['YearBuiltBins'] = pd.cut(df_train['YearBuilt'], bins, labels=labels)
#lets plot the countplot for this newly created binned feature

fig, ax = plt.subplots(figsize=(10,5))

ax = sns.countplot(df_train['YearBuiltBins'],ax=ax)
#YrSold

sns.countplot(df_train['YrSold'])
#MoSold

sns.countplot(df_train['MoSold'])
#let's first plot scatter plot of the numerical variables with Sale Price

numerical_features = ['LotArea','TotalBsmtSF','GrLivArea','PoolArea','GarageArea', 'WoodDeckSF','1stFlrSF', '2ndFlrSF','SalePrice']



sns.pairplot(df_train[numerical_features])
print(df_train['PoolArea'].value_counts())

sns.boxplot(df_train['PoolArea'])
fig, ax = plt.subplots(figsize=(10,5))

sns.barplot(x=df_train['YearBuiltBins'], y =df_train['SalePrice'])
sns.barplot(x= df_train['YrSold'],y=df_train['SalePrice'])
sns.barplot(x= 'OverallQual',y='SalePrice',data=df_train)

sns.lineplot(x= 'OverallQual',y='SalePrice',data=df_train)
fig, ax = plt.subplots(figsize=(20,20))

plot = sns.catplot(x= 'OverallQual',y='SalePrice',data=df_train,ax=ax,kind='swarm',sharey=False)

plt.close(plot.fig)
sns.barplot(x= 'OverallCond',y='SalePrice',data=df_train)

sns.lineplot(x= 'OverallCond',y='SalePrice',data=df_train)
fig, ax = plt.subplots(figsize=(20,20))

plot = sns.catplot(x= 'OverallCond',y='SalePrice',data=df_train,ax=ax,kind='swarm',sharey=False)

plt.close(plot.fig)


#Let's bin the GrLivArea.

bins = [300,500,700,1000,1500,2000,2500,3000, 3500,4000,4500,5000,6000]



labels = ['< 500','300-500 ','500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000','3000-3500','3500-4000','4000-4500','4500-5000','5000+']



df_train['GrLivAreaBins'] = pd.cut(df_train['GrLivArea'], bins, labels=labels)





df_train['Age'] = df_train['YrSold'] - df_train['YearBuilt']



#Newly derived Age feature                                             

df_train[['YrSold','YearBuilt','Age']]
#Let's further create bins for Age.

#Considering -1 as well in bin so that houses with YrSold and YearBuilt same will have Age as 0 and must be accomodated in some bin.

bins = [-1,0, 10,20,30, 40, 50, 60,80,100,120]

labels = ['0','0-10 ','10-20', '20-30', '30-40', '40-50', '50-60','60-80','80-100','100+']

df_train['AgeBins'] = pd.cut(df_train['Age'], bins, labels=labels)
fig, ax = plt.subplots(figsize=(20,20))

sns.barplot(x='AgeBins',y='SalePrice',hue='OverallQual',data=df_train,ax=ax)
fig, ax = plt.subplots(figsize=(20,20))

sns.barplot(x='AgeBins',y='SalePrice',hue='GrLivAreaBins',data=df_train,ax=ax)
fig, ax = plt.subplots(figsize=(20,20))

plot = sns.catplot(x= 'OverallQual',y='SalePrice',hue='GrLivAreaBins',data=df_train,ax=ax,kind='swarm',sharey=False)

plt.close(plot.fig)
df_train.loc[(df_train['OverallQual'] ==10) & (df_train['SalePrice']  < 200000)][['GrLivArea','OverallCond','OverallQual','SaleCondition']]
df_train.loc[(df_train['OverallQual'] ==10) & (df_train['SalePrice']  < 200000)][['GrLivArea','OverallCond','OverallQual','SaleCondition','1stFlrSF','2ndFlrSF']]
fig, ax = plt.subplots(figsize=(20,20))

plot = sns.catplot(x= 'OverallCond',y='SalePrice',data=df_train,ax=ax,kind='swarm',hue='GrLivAreaBins',sharey=False)

plt.close(plot.fig)
df_train.loc[(df_train['OverallCond'] == 5) & (df_train['SalePrice']  > 700000)][['GrLivArea','OverallCond','OverallQual','1stFlrSF','2ndFlrSF','GarageArea']]
fig, ax = plt.subplots(figsize=(20,20))

sns.heatmap(df_train.corr(),ax=ax)