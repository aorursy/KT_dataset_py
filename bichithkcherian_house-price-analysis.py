import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plotting Tools
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#print(os.listdir("../input"))
#fuction to show more rows and columns
def show_all(df):
    #This fuction lets us view the full dataframe
    with pd.option_context('display.max_rows', 100, 'display.max_columns', 100):
        display(df)
# Bring test data into the environment
train = pd.read_csv('/kaggle/input/house-price/train.csv', index_col='Id')
#train=pd.read_csv('train.csv')
show_all(train.head())
#gives info regarding data types 
train.info()
#basic representative values of dataset
train.describe().T
# Plot missing values of each column in the given dataset 
def plot_missing(df):
    # Find columns having missing values and count
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    # Plot missing values by count 
    missing.plot.bar(figsize=(12,8))
    plt.xlabel('Columns with missing values')
    plt.ylabel('Count')
    
    #search for missing data
    import missingno as msno
    msno.matrix(df=df, figsize=(16,8), color=(0,0.2,1))
    
plot_missing(train)
# Drop columns which has very high number of missing values
df=train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
#checking for missing values
plot_missing(df)
#checking the approximate distribution of 'LotFrontage' to substitute median
plt.figure(figsize=(10,4))
sns.distplot(df['LotFrontage'].dropna())
plt.xlabel('LotFrontage')
#dataframe for each 'Neighborhood' is created for future analysis(groupby was not used)
NAmes=df.loc[df.Neighborhood =='NAmes',:]
Gilbert=df.loc[df.Neighborhood =='Gilbert',:]
StoneBr=df.loc[df.Neighborhood =='StoneBr',:]
BrDale=df.loc[df.Neighborhood =='BrDale',:]
NPkVill=df.loc[df.Neighborhood =='NPkVill',:]
NridgHt=df.loc[df.Neighborhood =='NridgHt',:]
Blmngtn=df.loc[df.Neighborhood =='Blmngtn',:]
NoRidge=df.loc[df.Neighborhood =='NoRidge',:]
Somerst=df.loc[df.Neighborhood =='Somerst',:]
SawyerW=df.loc[df.Neighborhood =='SawyerW',:]
Sawyer=df.loc[df.Neighborhood =='Sawyer',:]
NWAmes=df.loc[df.Neighborhood =='NWAmes',:]
OldTown=df.loc[df.Neighborhood =='OldTown',:]
BrkSide=df.loc[df.Neighborhood =='BrkSide',:]
ClearCr=df.loc[df.Neighborhood =='ClearCr',:]
SWISU=df.loc[df.Neighborhood =='SWISU',:]
Edwards=df.loc[df.Neighborhood =='Edwards',:]
CollgCr=df.loc[df.Neighborhood =='CollgCr',:]
Crawfor=df.loc[df.Neighborhood =='Crawfor',:]
Blueste=df.loc[df.Neighborhood =='Blueste',:]
IDOTRR=df.loc[df.Neighborhood =='IDOTRR',:]
Mitchel=df.loc[df.Neighborhood =='Mitchel',:]
Timber=df.loc[df.Neighborhood =='Timber',:]
MeadowV=df.loc[df.Neighborhood =='MeadowV',:]
Veenker=df.loc[df.Neighborhood =='Veenker',:]
#we are using the above created Neighborhood dataframes to fill NaN based on each Neighborhood. It wont be accurate if we
#use the orginal whole dataframe to fillna.
def fill_missing_values(df):
    df['MSZoning'].fillna(df['MSZoning'].mode(), inplace=True)
    df['MSSubClass'].fillna(df['MSSubClass'].mode(), inplace=True)
    df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    df['Utilities'].fillna(df['Utilities'].mode(), inplace=True)
    df['Exterior1st'].fillna(df['Exterior1st'].mode(), inplace=True)
    df['Exterior2nd'].fillna(df['Exterior2nd'].mode(), inplace=True)
    df['MasVnrType'].fillna(df['MasVnrType'].mode(), inplace=True)
    df['MasVnrArea'].fillna(df['MasVnrArea'].mode(), inplace=True)
    df['BsmtQual'].fillna('no_bsmt', inplace=True)
    df['BsmtCond'].fillna('no_bsmt', inplace=True)
    df['BsmtExposure'].fillna('no_bsmt', inplace=True)
    df['BsmtFinType1'].fillna('no_bsmt', inplace=True)
    df['BsmtFinSF1'].fillna(0, inplace=True)
    df['BsmtFinType2'].fillna('no_bsmt', inplace=True)
    df['BsmtFinSF2'].fillna(0, inplace=True)
    df['Electrical'].fillna(df['Electrical'].mode(), inplace=True)
    df['FireplaceQu'].fillna('no_fp', inplace=True)
    df['GarageType'].fillna('no_Gar', inplace=True) 
    df['GarageYrBlt'].fillna(0, inplace=True)
    df['GarageFinish'].fillna('no_Gar', inplace=True) 
    df['GarageCars'].fillna(0, inplace=True)
    df['GarageArea'].fillna(0, inplace=True)
    df['GarageQual'].fillna('no_Gar', inplace=True)                        
    df['GarageCond'].fillna('no_Gar', inplace=True)                         
                             
a=[CollgCr, Veenker, Crawfor, NoRidge, Mitchel, Somerst,
       NWAmes, OldTown, BrkSide, Sawyer, NridgHt, NAmes,
       SawyerW, IDOTRR, MeadowV, Edwards, Timber, Gilbert,
       StoneBr, ClearCr, NPkVill, Blmngtn, BrDale, SWISU,
       Blueste]
for i in a:
    fill_missing_values(i)
#By this step we have filled our Nan values based on each Neighborhood.
#we are concatenating the Neighborhood dataframes to form the orginal dataframe.
#This new dataframe will be used for further analysis.
df1=pd.concat([CollgCr, Veenker, Crawfor, NoRidge, Mitchel, Somerst,
       NWAmes, OldTown, BrkSide, Sawyer, NridgHt, NAmes,
       SawyerW, IDOTRR, MeadowV, Edwards, Timber, Gilbert,
       StoneBr, ClearCr, NPkVill, Blmngtn, BrDale, SWISU,
       Blueste], axis=0, join='outer', ignore_index=False, keys=None,
          levels=None, names=None, verify_integrity=False, copy=True)
#here we are checking the correlation between saleprice of house and other variables
corr_mat = df1[["SalePrice","MSSubClass","MSZoning","LotFrontage","LotArea", "BldgType",
                       "OverallQual", "OverallCond","YearBuilt", "BedroomAbvGr", "PoolArea", "GarageArea",
                       "SaleType", "MoSold","YearRemodAdd",'TotalBsmtSF', '1stFlrSF',
                'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars']].corr()
f, ax = plt.subplots(figsize=(30, 15))
sns.heatmap(corr_mat, vmax=1 , square=True,annot=True,linewidths=.5);
ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='YearBuilt', y='SalePrice', data=df1);
ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x='OverallQual', y='SalePrice', color='green',data=df1);
plt.figure(figsize=(18,4))
sns.barplot(x='SaleType',y='SalePrice', data=df1);

ax = plt.subplots(figsize=(18, 4))
sns.lineplot(x='GarageArea', y='SalePrice', data=df1);
ax = plt.subplots(figsize=(25, 4))
sns.lineplot(x='TotalBsmtSF', y='SalePrice', data=df1);
ax = plt.subplots(figsize=(18, 4))
sns.lineplot(x='1stFlrSF', y='SalePrice', data=df1);

ax = plt.subplots(figsize=(16, 4))
sns.lineplot(x='GrLivArea', y='SalePrice', data=df1);
x=df1
b=df1['1stFlrSF']+df1['2ndFlrSF']+df1['TotalBsmtSF']
b
t=sns.jointplot(x=b, y=df1['GrLivArea'], kind="hex", color="#4CB391")
t.set_axis_labels('total SF(1st+2d+Tbsmt)', 'GrLivArea', fontsize=16)
#we can observe the level of influence of each variable in the following graphs.
cat1=["MSSubClass","MSZoning", "BldgType","OverallQual", "OverallCond", "BedroomAbvGr","SaleType", "MoSold",'FullBath','TotRmsAbvGrd','GarageCars']

plt.figure(figsize=(16,90))
plt.subplots_adjust(hspace=0.5)

i = 1
for j in cat1:
    plt.subplot(5,3,i)
    sns.boxplot(x=df1[j],y=df1['SalePrice'])
    plt.xlabel(j)
    i+=1
cat2=["LotFrontage","LotArea","YearBuilt","GarageArea","YearRemodAdd",'TotalBsmtSF','1stFlrSF','GrLivArea']
plt.figure(figsize=(16,90))

i = 1
for j in cat2:
    plt.subplot(8,1,i)
    sns.scatterplot(x=df1[j],y=df1['SalePrice'])
    plt.xlabel(j)
    i+=1

from numpy import median
fig=plt.figure(figsize=(15,5))
fig.suptitle('SalePrice vs Neighborhood')

ax = sns.barplot(data=df1,x="Neighborhood",y='SalePrice',estimator=median)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);

plt.figure(figsize=(25,90))
i = 1
for j in a:
    plt.subplot(13,2,i)
    
    b=sns.scatterplot(x=j['BldgType'],y=j['SalePrice'],hue=j['MSZoning'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Building Type', fontsize=18)
    plt.ylabel('SalePrice', fontsize=18)
    plt.setp(b.get_legend().get_texts(), fontsize=15)
    plt.setp(b.get_legend().get_title(), fontsize=15)

    i+=1
plt.figure(figsize=(25,90))

a=[CollgCr, Veenker, Crawfor, NoRidge, Mitchel, Somerst,
       NWAmes, OldTown, BrkSide, Sawyer, NridgHt, NAmes,
       SawyerW, IDOTRR, MeadowV, Edwards, Timber, Gilbert,
       StoneBr, ClearCr, NPkVill, Blmngtn, BrDale, SWISU,
       Blueste]
i = 1
for j in a:
    plt.subplot(13,2,i)
    sns.stripplot(x=j['YearBuilt'],y=j['SalePrice'],hue=j['OverallQual'],data=j);
    plt.xticks(rotation=90)
    plt.xlabel('YearBuilt')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('construction year', fontsize=18)
    plt.ylabel('salesprice', fontsize=18)
    plt.setp(b.get_legend().get_texts(), fontsize=15)
    plt.setp(b.get_legend().get_title(), fontsize=15)
    i+=1
    
#plt.suptitle('Neighborhood-SalePrice vs YearBuilt by OverallQuality');
#we are comparing square feet above ground to foundation materials used with Electrical system
plt.figure(figsize=(25,90))
i = 1
for j in a:
    plt.subplot(13,2,i)
    
    sns.scatterplot(x=j['Foundation'],y=j['GrLivArea'],hue=j['Electrical'])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Foudation', fontsize=18)
    plt.ylabel('SF above basement', fontsize=18)
    plt.setp(b.get_legend().get_texts(), fontsize=15)
    plt.setp(b.get_legend().get_title(), fontsize=15)

    i+=1
# Here we are comparing the proximity to various conditions ad sale price of house
plt.figure(figsize=(25,90))
i = 1
for j in a:
    plt.subplot(13,2,i)
    sns.boxenplot(x=j['Condition1'],y=j['SalePrice']);
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('condition1', fontsize=18)
    plt.ylabel('saleprice', fontsize=18)
    i+=1
#plt.suptitle("boxen plot");
#here we are comparing type of house and condition of house to sale price of the house
plt.figure(figsize=(25,90))
i = 1
for j in a:
    plt.subplot(13,2,i)
    sns.swarmplot(x=j['MSSubClass'],y=j['SalePrice'],hue=j['SaleCondition'],data=j)
    plt.xticks(rotation=90)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('saleprice', fontsize=18)
    plt.xlabel('MSSubClass',fontsize=18)
    i+=1
 #further analysis will be updated in the future do upvote