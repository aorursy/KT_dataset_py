# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# This homework write by hadi sarvari For Course "advance machin learning with python"

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer

from sklearn.neighbors import KNeighborsRegressor

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")
df_sample_sub = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

#Creating a copy of the train and test datasets
df_test.head()
df_train.head()
#descriptive statistics summary

df_train['SalePrice'].describe()


#histogram

sb.distplot(df_train['SalePrice']);

#----------------------------Evaluate relationship with SalePrice and GrLivArea------------------------------------
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#---------------------------Evaluate relationship with SalePrice and TotalBsmtSF-----------------------------
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#----------------------------Evaluate relationship with SalePrice and LotArea------------------------------------
#scatter plot LotArea/saleprice

var = 'LotArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# It seems that 'LotArea' and 'TotalBsmtSF' haven't a strong  relationship
#---------------------------Evaluate relationship with SalePrice and OverallQual-----------------------------
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sb.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#box plot overallqual/saleprice

var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(20, 16))

fig = sb.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

sb.swarmplot(x='YearBuilt', y="SalePrice", data=df_train, color=".25")

plt.xticks(weight='bold',rotation=90)
#---------------------------Evaluate relationship with SalePrice and OverallQual-----------------------------
#box plot MSZoning/saleprice

var = 'MSZoning'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data=data.sort_values(by='SalePrice', ascending=True)

f, ax = plt.subplots(figsize=(8, 6))

fig = sb.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#---------------------------Evaluate relationship with SalePrice and OverallQual-----------------------------
#box plot Neighborhood/saleprice

var = 'Neighborhood'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data=data.sort_values(by='SalePrice', ascending=True)

f, ax = plt.subplots(figsize=(8, 6))

fig = sb.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(weight='bold',rotation=90)
#In my opinion, this heatmap is the best way to get a quick overview of  columns relationships
correlation_train=df_train.corr()

sb.set(font_scale=2)

plt.figure(figsize = (45,70))

ax = sb.heatmap(correlation_train, annot=True,annot_kws={"size": 25},fmt='.1f',cmap='PiYG', linewidths=.5)
corr_dict=correlation_train['SalePrice'].sort_values(ascending=False).to_dict()

important_columns=[]

for key,value in corr_dict.items():

    if ((value>=0.5) & (value<0.999)) | ((value<=-0.5)& (value>-0.999)):

        important_columns.append(key)

important_columns    
#scatterplot

sb.set()

sb.set(font_scale=2)

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF','1stFlrSF', 'FullBath','TotRmsAbvGrd', 'YearBuilt','YearRemodAdd']

sb.pairplot(df_train[cols], size = 3.5)

plt.show()
#------------------------------combine train and test------------------------------------------

#combine train and test datasets.Because This processes are must be carried out together.

pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 81)

train_test=pd.concat([df_train,df_test],axis=0,sort=False)

train_test.head()
#missing data

total = train_test.isnull().sum().sort_values(ascending=False)

percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]
train_test.PoolQC.unique()
PoolArea=pd.DataFrame(train_test['PoolArea'],columns=['PoolArea'])

PoolQC=pd.DataFrame(train_test['PoolQC'],columns=['PoolQC'])

Df_Pool=pd.concat([PoolArea,PoolQC],axis=1)

Df_Pool[(Df_Pool['PoolArea'] !=0) & (Df_Pool['PoolArea'].isna())]
Df_Pool[(Df_Pool['PoolQC'].isna())].shape

Df_Pool[(Df_Pool['PoolArea']==0)].shape
#we haven't any recorde for pair (PoolArea,PoolQC) that PoolArea>0 and PoolQC is null, 

#and number of recorde with ['PoolArea']==0 is equal with number of record with 'PoolQC' ==null 

#so in we replace recordes  PoolQC==null with PoolQC==NA



#note :data description says NA means "No Pool". That make sense, given the huge ratio of missing value (100%) 

#and majority of houses have no Pool at all in general.
train_test["PoolQC"] = train_test["PoolQC"].fillna("NA")
#--------------------------------------------MiscFeature------------------------------------------------------
train_test[(train_test['MiscFeature'].isna())].shape
train_test["MiscFeature"] = train_test["MiscFeature"].fillna("NA")
#--------------------------------------------Alley------------------------------------------------------
train_test[(train_test['Alley'].isna())].shape
train_test["Alley"] = train_test["Alley"].fillna("NA")
#--------------------------------------------Fence------------------------------------------------------
train_test[(train_test['Fence'].isna())].shape
train_test["Fence"] = train_test["Fence"].fillna("NA")
#--------------------------------------------Fence------------------------------------------------------
train_test.FireplaceQu.unique()
#FireplaceQu      Fireplaces

FireplaceQu=pd.DataFrame(train_test['FireplaceQu'],columns=['FireplaceQu'])

Fireplaces=pd.DataFrame(train_test['Fireplaces'],columns=['Fireplaces'])

Df_Fireplace=pd.concat([FireplaceQu,Fireplaces],axis=1)

Df_Fireplace[(Df_Fireplace['Fireplaces'] !=0) & (Df_Fireplace['FireplaceQu'].isna())]
Df_Fireplace[(Df_Fireplace['FireplaceQu'].isna())].shape
Df_Fireplace[(Df_Fireplace['Fireplaces']==0)].shape
train_test['FireplaceQu'][(Df_Fireplace['FireplaceQu'].isna())]=train_test['FireplaceQu'][(Df_Fireplace['FireplaceQu'].isna())].fillna('NA')
#we haven't any recorde for pair (Fireplaces,FireplaceQu) that PoolArea>0 and FireplaceQu is null, 

#and number of recorde with ['Fireplaces']==0 is equal with number of record with 'FireplaceQu' ==null 

#so in we replace recordes  FireplaceQu==null with FireplaceQu==NA



#note :data description says NA means "No Fireplace". That make sense, given the huge ratio of missing value (100%) 

#and majority of houses have no Fireplace at all in general.
#--------------------------------------------LotFrontage------------------------------------------------------
train_test.LotFrontage.unique()
train_test['LotFrontage'][(train_test.LotFrontage=='NA')]=0

#Street  LotFrontage

pd.set_option('display.max_rows', 5000)           

idD=pd.DataFrame(train_test['Id'],columns=['Id'])

LotArea=pd.DataFrame(train_test['LotArea'],columns=['LotArea'])

LotFrontage=pd.DataFrame(train_test['LotFrontage'],columns=['LotFrontage'])

Neighborhood=pd.DataFrame(train_test['Neighborhood'],columns=['Neighborhood'])

Df_LotFrontage=pd.concat([LotArea,LotFrontage,Neighborhood,idD],axis=1)

Df_LotFrontage.head()



#KNN form k=7 
#for name,group in Df_LotFrontage.groupby('Neighborhood'):

#    for index,row in group[group["LotFrontage"].isna()].iterrows():

#            print(index,row)

#            print('-------------------------------------------')

#    print('*******************************************')
KNN = KNeighborsRegressor(n_neighbors=3)
 

for name,group in Df_LotFrontage.groupby('Neighborhood'):

    

       if group[group["LotFrontage"].isna()& (group['LotFrontage']!=0)].shape[0] >=3 :

            DF_LotFrontage_train=group[(group['LotFrontage'].notna()) & (group['LotFrontage']!=0)]

            X = DF_LotFrontage_train.drop(['LotFrontage','Id'], axis=1)

            Y = DF_LotFrontage_train['LotFrontage']

            X['Neighborhood']=5000000

            KNN.fit(X, Y)

            for index,row in group[group["LotFrontage"].isna()].iterrows():

                DF_One_LotArea=pd.DataFrame([row['LotArea']],columns=['LotArea'])

                DF_One_Neighborhood=pd.DataFrame([5000000],columns=['Neighborhood'])

                D_test=pd.concat([DF_One_LotArea,DF_One_Neighborhood],axis=1)

                pred=KNN.predict(D_test)

                train_test["LotFrontage"][train_test.Id==row['Id']]=round(pred[0], 0)

                
 #pred=KNN.predict(D_test)
train_test[(train_test['LotFrontage'].isna())].shape
train_test['LotFrontage'] = train_test['LotFrontage'].fillna(train_test.groupby('1stFlrSF')['LotFrontage'].transform('mean'))
#----------------------------------Garage-------------------------------------------
GarageFinish=pd.DataFrame(train_test['GarageFinish'],columns=['GarageFinish'])

GarageCars=pd.DataFrame(train_test['GarageCars'],columns=['GarageCars'])

GarageArea=pd.DataFrame(train_test['GarageArea'],columns=['GarageArea'])

GarageQual=pd.DataFrame(train_test['GarageQual'],columns=['GarageQual'])

GarageCond=pd.DataFrame(train_test['GarageCond'],columns=['GarageCond'])

GarageType=pd.DataFrame(train_test['GarageType'],columns=['GarageType'])

GarageYrBlt=pd.DataFrame(train_test['GarageYrBlt'],columns=['GarageYrBlt'])

Df_Garage=pd.concat([GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,GarageType,GarageYrBlt],axis=1)

DF_Garage_Cond=Df_Garage[(Df_Garage['GarageType'].isna()) | (Df_Garage['GarageCond'].isna()) | (Df_Garage['GarageQual'].isna())

                        | (Df_Garage['GarageFinish'].isna())]

DF_Garage_Cond.head()
DF_Garage_Cond.shape
train_test.GarageFinish.unique()
train_test['GarageFinish'][train_test['GarageFinish'].isna()].shape

train_test['GarageFinish'][(train_test['GarageFinish'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))]=train_test['GarageFinish'][(train_test['GarageFinish'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))].fillna('NA')
train_test.GarageQual.unique()
train_test['GarageQual'][train_test['GarageQual'].isna()].shape
train_test['GarageQual'][(train_test['GarageQual'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))]=train_test['GarageQual'][(train_test['GarageQual'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))].fillna('NA')
train_test.GarageCond.unique()
train_test['GarageCond'][train_test['GarageCond'].isna()].shape
train_test['GarageCond'][(train_test['GarageCond'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))]=train_test['GarageCond'][(train_test['GarageCond'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))].fillna('NA')
train_test.GarageType.unique()
train_test['GarageType'][train_test['GarageType'].isna()].shape
train_test['GarageType'][(train_test['GarageType'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))]=train_test['GarageType'][(train_test['GarageType'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))].fillna('NA')
train_test.GarageYrBlt.unique()
train_test['GarageYrBlt'][train_test['GarageYrBlt'].isna()].shape
train_test['GarageYrBlt'][(train_test['GarageYrBlt'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))]=train_test['GarageYrBlt'][(train_test['GarageYrBlt'].isna()) & ((train_test['GarageCars']==0) | (train_test['GarageArea']==0))].fillna(0)
GarageFinish=pd.DataFrame(train_test['GarageFinish'],columns=['GarageFinish'])

GarageCars=pd.DataFrame(train_test['GarageCars'],columns=['GarageCars'])

GarageArea=pd.DataFrame(train_test['GarageArea'],columns=['GarageArea'])

GarageQual=pd.DataFrame(train_test['GarageQual'],columns=['GarageQual'])

GarageCond=pd.DataFrame(train_test['GarageCond'],columns=['GarageCond'])

GarageType=pd.DataFrame(train_test['GarageType'],columns=['GarageType'])

GarageYrBlt=pd.DataFrame(train_test['GarageYrBlt'],columns=['GarageYrBlt'])

Df_Garage=pd.concat([GarageFinish,GarageCars,GarageArea,GarageQual,GarageCond,GarageType,GarageYrBlt],axis=1)

DF_Garage_Cond=Df_Garage[(Df_Garage['GarageType'].isna()) | (Df_Garage['GarageCond'].isna()) | (Df_Garage['GarageQual'].isna())

                        | (Df_Garage['GarageFinish'].isna())]

DF_Garage_Cond.head()
#BsmtFinSF2,TotalBsmtSF

BsmtExposure=pd.DataFrame(train_test['BsmtExposure'],columns=['BsmtExposure'])

BsmtCond=pd.DataFrame(train_test['BsmtCond'],columns=['BsmtCond'])

BsmtQual=pd.DataFrame(train_test['BsmtQual'],columns=['BsmtQual'])

BsmtFinType2=pd.DataFrame(train_test['BsmtFinType2'],columns=['BsmtFinType2'])

BsmtFinType1=pd.DataFrame(train_test['BsmtFinType1'],columns=['BsmtFinType1'])

BsmtFinSF2=pd.DataFrame(train_test['BsmtFinSF2'],columns=['BsmtFinSF2'])

TotalBsmtSF=pd.DataFrame(train_test['TotalBsmtSF'],columns=['TotalBsmtSF'])

BsmtUnfSF=pd.DataFrame(train_test['BsmtUnfSF'],columns=['BsmtUnfSF'])

Df_Bsmt=pd.concat([BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF],axis=1)

DF_Bsmt_Cond=Df_Bsmt[(Df_Bsmt['BsmtExposure'].isna()) | (Df_Bsmt['BsmtCond'].isna()) | (Df_Bsmt['BsmtQual'].isna())

                        | (Df_Bsmt['BsmtFinType2'].isna())| (Df_Bsmt['BsmtFinType1'].isna())| (Df_Bsmt['BsmtFinSF2'].isna())

                        | (Df_Bsmt['BsmtUnfSF'].isna())| (Df_Bsmt['TotalBsmtSF'].isna())]

DF_Bsmt_Cond.head()





train_test['BsmtQual'][(train_test['BsmtQual'].isna()) & ((train_test['TotalBsmtSF']==0) )]=train_test['BsmtQual'][(train_test['BsmtQual'].isna()) & ((train_test['TotalBsmtSF']==0) )].fillna('NA')
train_test['BsmtCond'][(train_test['BsmtCond'].isna()) & ((train_test['TotalBsmtSF']==0) )]=train_test['BsmtCond'][(train_test['BsmtCond'].isna()) & ((train_test['TotalBsmtSF']==0) )].fillna('NA')
train_test['BsmtFinType1'][(train_test['BsmtFinType1'].isna()) & ((train_test['TotalBsmtSF']==0) )]=train_test['BsmtFinType1'][(train_test['BsmtFinType1'].isna()) & ((train_test['TotalBsmtSF']==0) )].fillna('NA')
train_test['BsmtExposure'][(train_test['BsmtExposure'].isna()) & ((train_test['TotalBsmtSF']==0) )]=train_test['BsmtExposure'][(train_test['BsmtExposure'].isna()) & ((train_test['TotalBsmtSF']==0) )].fillna('NA')
train_test['BsmtFinType2'][(train_test['BsmtFinType2'].isna()) & ((train_test['TotalBsmtSF']==0) )]=train_test['BsmtFinType2'][(train_test['BsmtFinType2'].isna()) & ((train_test['TotalBsmtSF']==0) )].fillna('NA')
#--------------------------------------------------------------
#BsmtFinSF2,TotalBsmtSF

BsmtExposure=pd.DataFrame(train_test['BsmtExposure'],columns=['BsmtExposure'])

BsmtCond=pd.DataFrame(train_test['BsmtCond'],columns=['BsmtCond'])

BsmtQual=pd.DataFrame(train_test['BsmtQual'],columns=['BsmtQual'])

BsmtFinType2=pd.DataFrame(train_test['BsmtFinType2'],columns=['BsmtFinType2'])

BsmtFinType1=pd.DataFrame(train_test['BsmtFinType1'],columns=['BsmtFinType1'])

BsmtFinSF2=pd.DataFrame(train_test['BsmtFinSF2'],columns=['BsmtFinSF2'])

TotalBsmtSF=pd.DataFrame(train_test['TotalBsmtSF'],columns=['TotalBsmtSF'])

BsmtUnfSF=pd.DataFrame(train_test['BsmtUnfSF'],columns=['BsmtUnfSF'])

Df_Bsmt=pd.concat([BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF],axis=1)

DF_Bsmt_Cond=Df_Bsmt[(Df_Bsmt['BsmtExposure'].isna()) | (Df_Bsmt['BsmtCond'].isna()) | (Df_Bsmt['BsmtQual'].isna())

                        | (Df_Bsmt['BsmtFinType2'].isna())| (Df_Bsmt['BsmtFinType1'].isna())| (Df_Bsmt['BsmtFinSF2'].isna())

                        | (Df_Bsmt['BsmtUnfSF'].isna())| (Df_Bsmt['TotalBsmtSF'].isna())]

DF_Bsmt_Cond.head()

Df_Bsmt[Df_Bsmt['BsmtCond']=='TA'].groupby('BsmtExposure').agg({ 'BsmtExposure':np.size})
#mode for  'BsmtCond'=='TA'   equal No
train_test['BsmtExposure'][(train_test['BsmtExposure'].isna()) & (train_test['BsmtCond']=='TA' )]=train_test['BsmtExposure'][(train_test['BsmtExposure'].isna()) & (train_test['BsmtCond']=='TA') ].fillna('No')
#-------------------------------------------
Df_Bsmt[Df_Bsmt['BsmtCond']=='TA'].groupby('BsmtQual').agg({ 'BsmtQual':np.size})
#So mode for 'BsmtQual' when 'BsmtCond']=='TA' equal 'TA'
train_test['BsmtQual'][(train_test['BsmtQual'].isna()) & (train_test['BsmtCond']=='TA' )]=train_test['BsmtQual'][(train_test['BsmtQual'].isna()) & (train_test['BsmtCond']=='TA') ].fillna('TA')
Df_Bsmt[Df_Bsmt['BsmtCond']=='Fa'].groupby('BsmtQual').agg({ 'BsmtQual':np.size})
#So mode for 'BsmtQual' when 'BsmtCond']=='Fa' equal 'TA'
train_test['BsmtQual'][(train_test['BsmtQual'].isna()) & (train_test['BsmtCond']=='Fa' )]=train_test['BsmtQual'][(train_test['BsmtQual'].isna()) & (train_test['BsmtCond']=='Fa') ].fillna('TA')
#-----------------------------------------------------------
Df_Bsmt[(Df_Bsmt['BsmtCond']=='TA') & (Df_Bsmt['BsmtFinType1']=='GLQ') & (Df_Bsmt['BsmtQual']=='Gd')].groupby('BsmtFinType2').agg({ 'BsmtFinType2':np.size})
# because  BsmtUnfSF=not null  so 
train_test['BsmtFinType2'][(train_test['BsmtFinType2'].isna()) &(Df_Bsmt['BsmtCond']=='TA') & (Df_Bsmt['BsmtFinType1']=='GLQ') & (Df_Bsmt['BsmtQual']=='Gd')]=train_test['BsmtFinType2'][(train_test['BsmtFinType2'].isna()) & (Df_Bsmt['BsmtCond']=='TA') & (Df_Bsmt['BsmtFinType1']=='GLQ') & (Df_Bsmt['BsmtQual']=='Gd') ].fillna('Unf')
#BsmtFinSF2,TotalBsmtSF

BsmtExposure=pd.DataFrame(train_test['BsmtExposure'],columns=['BsmtExposure'])

BsmtCond=pd.DataFrame(train_test['BsmtCond'],columns=['BsmtCond'])

BsmtQual=pd.DataFrame(train_test['BsmtQual'],columns=['BsmtQual'])

BsmtFinType2=pd.DataFrame(train_test['BsmtFinType2'],columns=['BsmtFinType2'])

BsmtFinType1=pd.DataFrame(train_test['BsmtFinType1'],columns=['BsmtFinType1'])

BsmtFinSF2=pd.DataFrame(train_test['BsmtFinSF2'],columns=['BsmtFinSF2'])

TotalBsmtSF=pd.DataFrame(train_test['TotalBsmtSF'],columns=['TotalBsmtSF'])

BsmtUnfSF=pd.DataFrame(train_test['BsmtUnfSF'],columns=['BsmtUnfSF'])

Df_Bsmt=pd.concat([BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF],axis=1)

DF_Bsmt_Cond=Df_Bsmt[(Df_Bsmt['BsmtExposure'].isna()) | (Df_Bsmt['BsmtCond'].isna()) | (Df_Bsmt['BsmtQual'].isna())

                        | (Df_Bsmt['BsmtFinType2'].isna())| (Df_Bsmt['BsmtFinType1'].isna())| (Df_Bsmt['BsmtFinSF2'].isna())

                        | (Df_Bsmt['BsmtUnfSF'].isna())| (Df_Bsmt['TotalBsmtSF'].isna())]

DF_Bsmt_Cond.head()



Df_Bsmt[(Df_Bsmt['BsmtQual']=='Gd') & (Df_Bsmt['BsmtFinType1']=='GLQ') & (Df_Bsmt['BsmtFinType2']=='Rec') & (Df_Bsmt['BsmtExposure']=='Mn')].groupby('BsmtCond').agg({ 'BsmtCond':np.size})
train_test['BsmtCond'][(train_test['BsmtCond'].isna()) &  (Df_Bsmt['BsmtQual']=='Gd') & (Df_Bsmt['BsmtFinType1']=='GLQ') & (Df_Bsmt['BsmtFinType2']=='Rec') & (Df_Bsmt['BsmtExposure']=='Mn')]=train_test['BsmtCond'][(train_test['BsmtCond'].isna()) & (Df_Bsmt['BsmtQual']=='Gd') & (Df_Bsmt['BsmtFinType1']=='GLQ') & (Df_Bsmt['BsmtFinType2']=='Rec') & (Df_Bsmt['BsmtExposure']=='Mn') ].fillna('TA')
#------------------------------------------------------
Df_Bsmt[(Df_Bsmt['BsmtQual']=='TA') & (Df_Bsmt['BsmtFinType1']=='BLQ') & (Df_Bsmt['BsmtFinType2']=='Unf') & (Df_Bsmt['BsmtExposure']=='No')].groupby('BsmtCond').agg({ 'BsmtCond':np.size})
train_test['BsmtCond'][(train_test['BsmtCond'].isna()) &  (train_test['BsmtQual']=='TA') & (train_test['BsmtFinType1']=='BLQ') & (train_test['BsmtFinType2']=='Unf') & (train_test['BsmtExposure']=='No')]=train_test['BsmtCond'][(train_test['BsmtCond'].isna()) & (train_test['BsmtQual']=='TA') & (train_test['BsmtFinType1']=='BLQ') & (train_test['BsmtFinType2']=='Unf') & (Df_Bsmt['BsmtExposure']=='No') ].fillna('TA')
#-------------------------------------------
Df_Bsmt[(Df_Bsmt['BsmtQual']=='TA') & (train_test['BsmtFinType1']=='ALQ') & (train_test['BsmtFinType2']=='Unf') & (train_test['BsmtExposure']=='Av')].groupby('BsmtCond').agg({ 'BsmtCond':np.size})
train_test['BsmtCond'][(train_test['BsmtCond'].isna()) &  (train_test['BsmtQual']=='TA') & (train_test['BsmtFinType1']=='ALQ') & (train_test['BsmtFinType2']=='Unf') & (train_test['BsmtExposure']=='Av')]=train_test['BsmtCond'][(train_test['BsmtCond'].isna()) & (train_test['BsmtQual']=='TA') & (train_test['BsmtFinType1']=='ALQ') & (train_test['BsmtFinType2']=='Unf') & (train_test['BsmtExposure']=='Av') ].fillna('TA')
#--------------------------MasVnrType      MasVnrArea
#MasVnrType      MasVnrArea

MasVnrArea=pd.DataFrame(train_test['MasVnrArea'],columns=['MasVnrArea'])

MasVnrType=pd.DataFrame(train_test['MasVnrType'],columns=['MasVnrType'])

MasVnr=pd.concat([MasVnrType,MasVnrArea],axis=1)

Df_MasVnr=MasVnr[(MasVnr['MasVnrType'].isna()) & (MasVnr['MasVnrArea'].isna())]

Df_MasVnr.shape

#23 samilar 1 difernce 

#None replace MasVnrType

#0 replace in MasVnrArea
MasVnr=pd.concat([MasVnrType,MasVnrArea],axis=1)

MasVnr[(MasVnr['MasVnrType'].isna())]
train_test['MasVnrArea'][(train_test['MasVnrArea'].isna()) &  (train_test['MasVnrType'].isna()) ]=train_test['MasVnrArea'][(train_test['MasVnrArea'].isna()) &  (train_test['MasVnrType'].isna()) ].fillna(0)
train_test['MasVnrType'][(train_test['MasVnrArea']==0) &  (train_test['MasVnrType'].isna()) ]=train_test['MasVnrType'][(train_test['MasVnrArea']==0) &  (train_test['MasVnrType'].isna()) ].fillna('None')
train_test['MasVnrType'][ train_test['MasVnrType'].isna() ]=train_test['MasVnrType'][(train_test['MasVnrType'].isna()) ].fillna('None')
train_test[train_test.index==1150]
train_test.groupby('MasVnrType').agg({'MasVnrType':np.size})

#train_test['MasVnrType'][(train_test['MasVnrArea']>0) &  (train_test['MasVnrType'].isna()) ]=train_test['MasVnrType'][(train_test['MasVnrArea']>0) &  (train_test['MasVnrType'].isna()) ].fillna('None')
train_test[train_test['MSZoning'].isna()].head()
train_test[train_test['LotArea']>14000].groupby('MSZoning').agg({'MSZoning':np.size})
train_test.groupby('MSZoning').agg({'MSZoning':np.size})
train_test['MSZoning'][ (train_test['MSZoning'].isna()) ]=train_test['MSZoning'][ (train_test['MSZoning'].isna()) ].fillna('RL')


train_test[(train_test['MasVnrType'].isna())].head()
total = train_test.isnull().sum().sort_values(ascending=False)

percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]
train_test.FireplaceQu.unique()
#FireplaceQu      Fireplaces

FireplaceQu=pd.DataFrame(train_test['FireplaceQu'],columns=['FireplaceQu'])

Fireplaces=pd.DataFrame(train_test['Fireplaces'],columns=['Fireplaces'])

Df_Fireplace=pd.concat([FireplaceQu,Fireplaces],axis=1)

Df_Fireplace[(Df_Fireplace['Fireplaces'] !=0) & (Df_Fireplace['FireplaceQu'].isna())]

#بنابراین جایگزینی 

#nan  with NA
train_test['FireplaceQu'][ (train_test['FireplaceQu'].isna()) ]=train_test['FireplaceQu'][ (train_test['FireplaceQu'].isna()) ].fillna('NA')
#-------------------------------------------------------------------------


#Utilities

Df_Utilities_Non=train_test[(train_test['Utilities'].isna())  ].head()



Df_Utilities_Non.head()
train_test.groupby('Utilities').agg({'Utilities':np.size})
train_test['Utilities'][ (train_test['Utilities'].isna()) ]=train_test['Utilities'][ (train_test['Utilities'].isna()) ].fillna('AllPub')
#BsmtFullBath

#BsmtHalfBath



BsmtFullBath=pd.DataFrame(train_test['BsmtFullBath'],columns=['BsmtFullBath'])

BsmtHalfBath=pd.DataFrame(train_test['BsmtHalfBath'],columns=['BsmtHalfBath'])

FullBath=pd.DataFrame(train_test['FullBath'],columns=['FullBath'])

HalfBath=pd.DataFrame(train_test['HalfBath'],columns=['HalfBath'])

Df_Bath=pd.concat([BsmtFullBath,BsmtHalfBath,FullBath,HalfBath],axis=1)

DF_Bath_All=Df_Bath[(Df_Bath['BsmtFullBath'].isna()) & (Df_Bath['BsmtHalfBath'].isna())  ]

DF_BathF_All=Df_Bath[(Df_Bath['BsmtFullBath'].isna())]

DF_BathH_All=Df_Bath[(Df_Bath['BsmtHalfBath'].isna())]

print(DF_BathF_All.shape)

print(DF_BathH_All.shape)

#DF_BathH_All

DF_Bath_All.head()
train_test.groupby('BsmtHalfBath').agg({'BsmtHalfBath':np.size})
train_test.groupby('BsmtFullBath').agg({'BsmtFullBath':np.size})
train_test['BsmtFullBath'][ (train_test['BsmtFullBath'].isna()) ]=train_test['BsmtFullBath'][ (train_test['BsmtFullBath'].isna()) ].fillna('0')
train_test['BsmtHalfBath'][ (train_test['BsmtHalfBath'].isna()) ]=train_test['BsmtHalfBath'][ (train_test['BsmtHalfBath'].isna()) ].fillna('0')
#-------------------------------------------------------------------------
train_test[train_test['Functional'].isna()].head()
train_test[ (train_test['OverallQual']==4)].groupby('Functional').agg({'Functional':np.size})
train_test['Functional'][ (train_test['Functional'].isna()) & (train_test['OverallQual']==4) ]=train_test['Functional'][ (train_test['Functional'].isna()) & (train_test['OverallQual']==4) ].fillna('Typ')
train_test[ (train_test['OverallQual']==1)].groupby('Functional').agg({'Functional':np.size})
train_test['Functional'][ (train_test['Functional'].isna()) & (train_test['OverallQual']==1) ]=train_test['Functional'][ (train_test['Functional'].isna()) & (train_test['OverallQual']==1) ].fillna('Mod')
total = train_test.isnull().sum().sort_values(ascending=False)

percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]
train_test[train_test['Exterior1st'].isna()].head()
train_test[(train_test['YearBuilt']>1935) & (train_test['YearBuilt'] < 1945)].groupby('Exterior1st').agg({'Exterior1st':np.size})
#Wd Sdng
total = train_test.isnull().sum().sort_values(ascending=False)

percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]
train_test['Exterior1st'][ (train_test['Exterior1st'].isna())  ]=train_test['Exterior1st'][ (train_test['Exterior1st'].isna()) ].fillna('Wd Sdng')
train_test[train_test['Exterior2nd'].isna()].head()
train_test[(train_test['YearBuilt']>1935) & (train_test['YearBuilt'] < 1945)].groupby('Exterior2nd').agg({'Exterior2nd':np.size})
train_test['Exterior2nd'][ (train_test['Exterior2nd'].isna())  ]=train_test['Exterior2nd'][ (train_test['Exterior2nd'].isna()) ].fillna('Wd Sdng')
#------------------------------------
train_test[(train_test['KitchenQual'].isna())].head()
train_test['KitchenQual'][(train_test['KitchenQual'].isna())]=train_test['KitchenQual'][(train_test['KitchenQual'].isna())].fillna('TA')
#GarageArea

train_test['GarageArea'][(train_test['GarageArea'].isna())]=train_test['GarageArea'][(train_test['GarageArea'].isna())].fillna(0)

train_test['GarageCars'][(train_test['GarageCars'].isna())]=train_test['GarageCars'][(train_test['GarageCars'].isna())].fillna(0)

#Electrical

train_test['Electrical'][(train_test['Electrical'].isna())]=train_test['Electrical'][(train_test['Electrical'].isna())].fillna('SBrkr')
train_test[train_test.isna().any(axis=1)].head()
#SaleType

train_test.groupby('SaleType').agg({'SaleType':np.size})
train_test['SaleType'][(train_test['SaleType'].isna())]=train_test['SaleType'][(train_test['SaleType'].isna())].fillna('WD')
train_test.head(5)


train_test['GarageCond'][(train_test['GarageCond'].isna())]=train_test['GarageCond'][(train_test['GarageCond'].isna())].fillna('NA')

train_test['BsmtFinSF2'][(train_test['BsmtFinSF2'].isna())]=train_test['BsmtFinSF2'][(train_test['BsmtFinSF2'].isna())].fillna(0)

train_test['BsmtFinType2'][(train_test['BsmtFinType2'].isna())]=train_test['BsmtFinType2'][(train_test['BsmtFinType2'].isna())].fillna('NA')

train_test['BsmtFinSF1'][(train_test['BsmtFinSF1'].isna())]=train_test['BsmtFinSF1'][(train_test['BsmtFinSF1'].isna())].fillna(0)

train_test['BsmtFinType1'][(train_test['BsmtFinType1'].isna())]=train_test['BsmtFinType1'][(train_test['BsmtFinType1'].isna())].fillna('NA')

train_test['BsmtExposure'][(train_test['BsmtExposure'].isna())]=train_test['BsmtExposure'][(train_test['BsmtExposure'].isna())].fillna('NA')

train_test['BsmtCond'][(train_test['BsmtCond'].isna())]=train_test['BsmtCond'][(train_test['BsmtCond'].isna())].fillna('NA')

train_test['BsmtQual'][(train_test['BsmtQual'].isna())]=train_test['BsmtQual'][(train_test['BsmtQual'].isna())].fillna('NA')

train_test['BsmtUnfSF'][(train_test['BsmtUnfSF'].isna())]=train_test['BsmtUnfSF'][(train_test['BsmtUnfSF'].isna())].fillna(0)

train_test['GarageFinish'][(train_test['GarageFinish'].isna())]=train_test['GarageFinish'][(train_test['GarageFinish'].isna())].fillna('NA')

train_test['GarageQual'][(train_test['GarageQual'].isna())]=train_test['GarageQual'][(train_test['GarageQual'].isna())].fillna('NA')

train_test['GarageYrBlt'][(train_test['GarageYrBlt'].isna())]=train_test['GarageYrBlt'][(train_test['GarageYrBlt'].isna())].fillna(0)

train_test['TotalBsmtSF'][(train_test['TotalBsmtSF'].isna())]=train_test['TotalBsmtSF'][(train_test['TotalBsmtSF'].isna())].fillna(0)

total = train_test.isnull().sum().sort_values(ascending=False)

percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]
total = train_test.isnull().sum().sort_values(ascending=False)

percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total']>0]

#ADD a column   Age_House


train_test['Age_House']=train_test['YrSold']-train_test['YearBuilt']

train_test[train_test['Age_House']<0]
#year is not suit so change with "what year ago?"
train_test['YrSold']=2010-train_test['YrSold']

train_test['YearBuilt']=2010-train_test['YearBuilt']

train_test['YearRemodAdd']=2010-train_test['YearRemodAdd']

train_test['GarageYrBlt']=2010-train_test['GarageYrBlt']
# what is suit?(label encoding or onehot encoding or replace )

#I choose replace for following feature
bin_map_Street  = {'Grvl':2,'Pave':1}

train_test['Street'] = train_test['Street'].replace(bin_map_Street)



bin_map_Alley  = {'Grvl':2,'Pave':1,'NA':0}

train_test['Alley'] = train_test['Alley'].replace(bin_map_Alley)



bin_map_LotShape  = {'Reg':4,'IR1':3,'IR2':2,'IR3':1}

train_test['LotShape'] = train_test['LotShape'].replace(bin_map_LotShape)



bin_map_Utilities  = {'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1}

train_test['Utilities'] = train_test['Utilities'].replace(bin_map_Utilities)



bin_map_Exter  = {'Ex':5,'Gd':4, 'TA':3,'Fa':2,'Po':1 ,'NA':0}

train_test['ExterQual'] = train_test['ExterQual'].replace(bin_map_Exter)

train_test['ExterCond'] = train_test['ExterCond'].replace(bin_map_Exter)

train_test['HeatingQC'] = train_test['HeatingQC'].replace(bin_map_Exter)

train_test['KitchenQual'] = train_test['KitchenQual'].replace(bin_map_Exter)

train_test['FireplaceQu'] = train_test['FireplaceQu'].replace(bin_map_Exter)

train_test['GarageQual'] = train_test['GarageQual'].replace(bin_map_Exter )

train_test['GarageCond'] = train_test['GarageCond'].replace(bin_map_Exter)

train_test['PoolQC'] = train_test['PoolQC'].replace(bin_map_Exter)



bin_map_Bsmt  = {'Ex':5,'Gd':4, 'TA':3,'Fa':2,'Po':1 ,'NA':0}

train_test['BsmtCond'] = train_test['BsmtCond'].replace(bin_map_Bsmt)

train_test['BsmtQual'] = train_test['BsmtQual'].replace(bin_map_Bsmt)





bin_map_BsmtExposure  = {'Gd':4,'Av':3, 'Mn':2,'No':1,'NA':0}

train_test['BsmtExposure'] = train_test['BsmtExposure'].replace(bin_map_BsmtExposure )



bin_map_BsmtFin = {'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3, 'LwQ':2,'Unf':1,'NA':0}

train_test['BsmtFinType1'] = train_test['BsmtFinType1'].replace(bin_map_BsmtFin )

train_test['BsmtFinType2'] = train_test['BsmtFinType2'].replace(bin_map_BsmtFin )



bin_map_CentralAir  = { 'Y':1,'N':0}

train_test['CentralAir'] = train_test['CentralAir'].replace(bin_map_CentralAir)

#----------------------------------------------------------------------------------------------------------

#LotFrontage, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, GarageYrBlt, GarageCars, GarageArea

bin_map_NA  = {'NA':0}

train_test['MasVnrArea'][train_test['MasVnrArea']=='NA'] = train_test['MasVnrArea'][train_test['MasVnrArea']=='NA'].replace(bin_map_NA)

train_test['BsmtFinSF1'][train_test['BsmtFinSF1']=='NA'] = train_test['BsmtFinSF1'][train_test['BsmtFinSF1']=='NA'].replace(bin_map_NA)

train_test['BsmtFinSF2'][train_test['BsmtFinSF2']=='NA'] = train_test['BsmtFinSF2'][train_test['BsmtFinSF2']=='NA'].replace(bin_map_NA)

train_test['BsmtUnfSF'][train_test['BsmtUnfSF']=='NA'] = train_test['BsmtUnfSF'][train_test['BsmtUnfSF']=='NA'].replace(bin_map_NA)

train_test['TotalBsmtSF'][train_test['TotalBsmtSF']=='NA'] = train_test['TotalBsmtSF'][train_test['TotalBsmtSF']=='NA'].replace(bin_map_NA)

train_test['BsmtFullBath'][train_test['BsmtFullBath']=='NA'] = train_test['BsmtFullBath'][train_test['BsmtFullBath']=='NA'].replace(bin_map_NA)

train_test['BsmtHalfBath'][train_test['BsmtHalfBath']=='NA'] = train_test['BsmtHalfBath'][train_test['BsmtHalfBath']=='NA'].replace(bin_map_NA)

train_test['GarageYrBlt'][train_test['GarageYrBlt']=='NA'] = train_test['GarageYrBlt'][train_test['GarageYrBlt']=='NA'].replace(bin_map_NA)

train_test['GarageCars'][train_test['GarageCars']=='NA'] = train_test['GarageCars'][train_test['GarageCars']=='NA'].replace(bin_map_NA)

train_test['GarageArea'][train_test['GarageArea']=='NA'] = train_test['GarageArea'][train_test['GarageArea']=='NA'].replace(bin_map_NA)
#transfer with  labelencoder for following feature
# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()

categorical_cols=['MSZoning','LandSlope','LandContour','LotConfig','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','Electrical','Functional','GarageType','GarageFinish','PavedDrive','Fence','MiscFeature','SaleType','SaleCondition']

#categorical_cols=[]

# apply le on categorical feature columns

train_test[categorical_cols] =train_test[categorical_cols].apply(lambda col: le.fit_transform(col))

# ADD 3 column 
train_test['TotalArea'] = train_test['TotalBsmtSF'] + train_test['1stFlrSF'] + train_test['2ndFlrSF'] + train_test['GrLivArea'] +train_test['GarageArea']



train_test['Bathrooms'] = train_test['FullBath'] + train_test['HalfBath']*0.5 



train_test['Year average']= (train_test['YearRemodAdd']+train_test['YearBuilt'])/2
#change type of collumnes for Memory and ... 
for col in train_test:

    if col !='SalePrice':

        train_test[col]=train_test[col].astype('int32')
#Outleir Detection base on two factors  1- corolation matrix 2-Test Data
a=train_test[0:1460]
import math

colors=['yellowgreen','red','deepskyblue','deepskyblue','gold','orchid','tan']

Num=0

NumCol=len(a.columns)

RowNum=int(math.floor((NumCol/2))+1)







for col in a.columns:

    #ax1 = plt.subplot2grid((RowNum, 2),(math.floor((Num/2)),Num%2))

    fig = plt.figure(figsize=(5,5))

    plt.scatter(x=a[col], y=a['SalePrice'], color=(colors[Num%7]),alpha=0.9)

    Num=Num+1

    #plt.axvline(x=13, color='r', linestyle='-')

    plt.title(col+' - Price scatter plot', fontsize=15, weight='bold' )

    plt.show()
sb.set(font_scale=1)

fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((5,2),(0,0))

plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'), alpha=0.5)

plt.axvline(x=3600, color='r', linestyle='-')

plt.title('Ground living Area- Price scatter plot', fontsize=10, weight='bold' )



ax1 = plt.subplot2grid((5,2),(0,1))

plt.scatter(x=a['TotalBsmtSF'], y=a['SalePrice'], color=('red'),alpha=0.5)

plt.axvline(x=3000, color='r', linestyle='-')

plt.title('Basement Area - Price scatter plot', fontsize=10, weight='bold' )



ax1 = plt.subplot2grid((5,2),(1,0))

plt.scatter(x=a['1stFlrSF'], y=a['SalePrice'], color=('deepskyblue'),alpha=0.5)

plt.axvline(x=2800, color='r', linestyle='-')

plt.title('First floor Area - Price scatter plot', fontsize=10, weight='bold' )



ax1 = plt.subplot2grid((5,2),(1,1))

plt.scatter(x=a['TotalArea'], y=a['SalePrice'], color=('gold'),alpha=0.9)

plt.axvline(x=10000, color='r', linestyle='-')

plt.title('TotalArea - Price scatter plot', fontsize=10, weight='bold' )







ax1 = plt.subplot2grid((5,2),(2,0))

plt.scatter(x=a['LotFrontage'], y=a['SalePrice'], color=('orchid'),alpha=0.5)

plt.axvline(x=230, color='r', linestyle='-')

plt.title('LotFrontage - Price scatter plot', fontsize=10, weight='bold' )



ax1 = plt.subplot2grid((5,2),(2,1))

plt.scatter(x=a['BsmtFinSF2'], y=a['SalePrice'], color=('tan'),alpha=0.9)

plt.axvline(x=1400, color='r', linestyle='-')

plt.title('BsmtFinSF2 - Price scatter plot', fontsize=10, weight='bold' )





ax1 = plt.subplot2grid((5,2),(3,0))

plt.scatter(x=a['BsmtFinSF1'], y=a['SalePrice'], color=('gold'),alpha=0.9)

plt.axvline(x=2100, color='r', linestyle='-')

plt.title('BsmtFinSF1 - Price scatter plot', fontsize=10, weight='bold' )



ax1 = plt.subplot2grid((5,2),(3,1))

plt.scatter(x=a['LotArea'], y=a['LotArea'], color=('tan'),alpha=0.9)

plt.axvline(x=100000, color='r', linestyle='-')

plt.title('LotArea - Price scatter plot', fontsize=10, weight='bold' )

#MasVnrArea



ax1 = plt.subplot2grid((5,2),(4,0))

plt.scatter(x=a['MasVnrArea'], y=a['MasVnrArea'], color=('red'),alpha=0.9)

plt.axvline(x=1300, color='r', linestyle='-')

plt.title('MasVnrArea - Price scatter plot', fontsize=10, weight='bold' )

plt.show()
from sklearn.model_selection import train_test_split

train=train_test[0:1460]



Test=train_test[1460:2919]



print(Test.shape)
Train=train.copy()

train=train[(train['LotFrontage'] < 230) ] #--209=>  get value from max in "Test['LotFrontage'].describe()" 

train=train[(train['BsmtFinSF1'] < 2100)] #-----1972 => get value from max in "Test['BsmtFinSF1'].describe()" 

train=train[(train['BsmtFinSF2'] < 1400)] #-----1393 =>...

train=train[(train['TotalBsmtSF'] < 3000)] #----2846

train=train[(train['1stFlrSF'] < 2800) ] #--2696

train=train[(train['GrLivArea'] < 3600)] #---3500

train=train[(train['TotalArea'] < 10000)] #----9692

train=train[(train['LotArea'] < 100000)] #---57000

train=train[(train['MasVnrArea'] < 1300)] #---1290



print(f'We removed {Train.shape[0]- train.shape[0]} outliers')
Test['LotFrontage'].describe()
#histogram and normal probability plot

from scipy.stats import norm

from scipy import stats

sb.distplot(train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
train['SalePrice'] = np.log1p(train['SalePrice'])

y= train['SalePrice']

x = train.drop(['SalePrice','Id'],axis=1)



#----------------------------------------Split Data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = .3, random_state=2019)

#----------------------------------------prepar Final Test Data
X_test = Test.drop(['SalePrice','Id'],axis=1)
#-------------------------------------------------------------------------------------------------
from sklearn.metrics import fbeta_score, make_scorer,mean_squared_error

def scorer(y,yhat):

    return np.sqrt(mean_squared_error(y,yhat))
my_score=make_scorer(scorer, greater_is_better=False)
def score(y_pred):

    return str(np.sqrt(mean_squared_error(y_test, y_pred)))
import xgboost as xgb

import lightgbm as lgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.55, gamma=0.0468, 

                             learning_rate=0.05, max_depth=4, 

                             min_child_weight=1.73, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.6,

                             subsample=0.5, silent=1,

                             random_state =7, nthread = -1)
#from sklearn.model_selection import GridSearchCV

#par = {'gamma':[0.0450,0.0500,0.0550],

#       'colsample_bytree':[0.4,0.45,0.5,0.55,0.6],

#       'learning_rate':[0.008,0.01,0.03,0.05],

#        'min_child_weight':[1.3,1.5,1.7,1.8],

#       'n_estimators':[1000,1200,1500,2000]

#      }

#GS = GridSearchCV(model_xgb, param_grid=par, cv=5, scoring='neg_mean_squared_error')

#GS.fit(x,y)
#GS.best_params_
#GS.best_score_
model_xgb.fit(x_train,y_train)

y_xgb_train = model_xgb.predict(x_train)

y_xgb_test = model_xgb.predict(x_test)



print(f'Root Mean Square Error train  {str(np.sqrt(mean_squared_error(y_train, y_xgb_train)))}')

print(f'Root Mean Square Error test  {score(y_xgb_test)}')

#xgb_pred = np.expm1(model_xgb.predict(test))

#print(rmsle(y_train, xgb_train_pred))
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=1000,

                              max_bin = 55, bagging_fraction = 0.70,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 9)
model_lgb.fit(x_train,y_train)

y_lgb_train = model_lgb.predict(x_train)

y_lgb_test = model_lgb.predict(x_test)

print(f'Root Mean Square Error train  {str(np.sqrt(mean_squared_error(y_train, y_lgb_train)))}')

print(f'Root Mean Square Error test  {score(y_lgb_test)}')
Pro=0.54

print(f'Root Mean Square Error train  {str(np.sqrt(mean_squared_error(y_train, (y_lgb_train*Pro + y_xgb_train*(1-Pro)))))}')

print(f'Root Mean Square Error test  {str(np.sqrt(mean_squared_error(y_test, (y_lgb_test*Pro + y_xgb_test*(1-Pro)))))}')
model_lgb.fit(x,y)

y_lgb_F = model_lgb.predict(X_test)

#-------------------------------------------------------

model_xgb.fit(x,y)

y_xgb_F = model_xgb.predict(X_test)

#-------------------------------------------------------

ensemble=np.expm1(y_xgb_F*(1-Pro)+ y_lgb_F*Pro)

#-------------------------------------------------------

sub = pd.DataFrame()

sub['Id'] = Test["Id"]

sub['SalePrice'] = ensemble

sub.to_csv('submission_F.csv',index=False)
