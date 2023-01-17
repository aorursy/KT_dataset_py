#!pip install pypng
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from collections import Counter
import math 
warnings.filterwarnings('ignore')
%matplotlib inline
df= pd.read_csv('../input/train.csv')
print (df.shape)
df.head()
df=df.iloc[:,1:81]
# check the columns
names=df.columns
print(names)
df.isna().sum().sort_values(ascending=False).head(20)
#descriptive statistics summary
df['SalePrice'].describe()
#histogram
#histogram and normal probability plot
sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
#applying log transformation
df['SalePrice'] = np.log(df['SalePrice'])
#histogram and normal probability plot
sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)
df['FrstFlrSF'] = np.log(df['1stFlrSF'])
df=df.drop(columns=['1stFlrSF'])
#histogram and normal probability plot
sns.distplot((df['FrstFlrSF']), fit=norm);
fig = plt.figure()
res = stats.probplot(df['FrstFlrSF'], plot=plt)
df=df.drop(columns=['2ndFlrSF'])
df[df['LowQualFinSF']==0]['LowQualFinSF'].count()
df=df.drop(columns=['LowQualFinSF'])
df['LotFrontage'].isna().sum()
sns.boxplot(df[~np.isnan(df['LotFrontage'])]['LotFrontage'], orient='v' )
print (df['LotFrontage'].describe())
print(df['LotFrontage'].median())
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].median())
df[['LotArea']]=np.sqrt(df[['LotArea']])
sns.distplot(df['LotArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['LotArea'], plot=plt)
ax = sns.regplot(x="SalePrice", y="LotArea", data=df)
sns.distplot(np.sqrt(df['GrLivArea']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.sqrt(df['GrLivArea']), plot=plt)
df['GrLivArea']=np.sqrt(df['GrLivArea'])
print ('missing Values')
print(df['MasVnrArea'].isna().sum())
print ('Values=0')
print(df[df['MasVnrArea']==0]['MasVnrArea'].count())
df[df['MasVnrArea']==0]['MasVnrArea'].count()
#box plot overallqual/saleprice
var = 'MasVnrType'
data = pd.concat([df['MasVnrArea'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y='MasVnrArea', data=df)
fig.axis(ymin=0, ymax=1600);
df=df.drop(columns=['MasVnrArea'])
print("na Valuest")
print(df['MasVnrType'].isna().sum())
# I will fill na as 0
df['MasVnrType']=df['MasVnrType'].fillna(0)
pd.Series(df['MasVnrType']).value_counts().plot('bar')
#box plot overallqual/saleprice
var = 'MasVnrType'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y='SalePrice', data=df)
cleanup_nums = {"MasVnrType":     {"None": 1, "BrkFace": 2, "Stone":3, "BrkCmn":4}}
df.replace(cleanup_nums, inplace=True)  
#box plot 
var = 'MasVnrType'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y='SalePrice', data=df)
#box plot 
var = 'BldgType'
data = pd.concat([df['BsmtFinSF1'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y='BsmtFinSF1', data=df)
#box plot 
var = 'BldgType'
data = pd.concat([df[df['BsmtFinSF1']>0]['BsmtFinSF1'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y=df[df['BsmtFinSF1']>0]['BsmtFinSF1'], data=df)
print ('missing Values')
print(df['BsmtFinSF1'].isna().sum())
print ('Values=0')
print(df[df['BsmtFinSF1']==0]['BsmtFinSF1'].count())
#histogram and normal probability plot
sns.distplot(np.sqrt(df[df['BsmtFinSF1']>0]['BsmtFinSF1']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.sqrt(df[df['BsmtFinSF1']>0]['BsmtFinSF1']), plot=plt)
ax = sns.regplot(x=df[df.BsmtFinSF1>0]["SalePrice"], y=np.sqrt(df[df.BsmtFinSF1>0]['BsmtFinSF1']), data=df)
df['BsmtFinSF1']=np.sqrt(df['BsmtFinSF1'])
print ('missing Values')
print(df['BsmtFinSF2'].isna().sum())
print ('Values=0')
print(df[df['BsmtFinSF2']==0]['BsmtFinSF2'].count())
df['BsmtFinSF2_Flag']=np.sqrt(df['BsmtFinSF2'])
dfTrain=df.drop(columns=['BsmtFinSF2'])
print ('missing Values')
print(df['BsmtUnfSF'].isna().sum())
print ('Values=0')
print(df[df['BsmtUnfSF']==0]['BsmtUnfSF'].count())
#box plot 
var = 'BldgType'
data = pd.concat([df['BsmtUnfSF'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y=df['BsmtUnfSF'], data=df)
print((df[df['BldgType']=='Duplex']['BsmtUnfSF']==0).count())
ax = sns.regplot(x=df[df.BsmtUnfSF>0]["SalePrice"], y=np.sqrt(df[df.BsmtUnfSF>0]['BsmtUnfSF']), data=df)
df['BsmtUnfSF']=np.sqrt(df['BsmtUnfSF'])
print ('missing Values')
print(df['TotalBsmtSF'].isna().sum())
print ('Values=0')
print(df[df['TotalBsmtSF']==0]['TotalBsmtSF'].count())
#box plot 
var = 'BldgType'
data = pd.concat([df['TotalBsmtSF'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y=df['TotalBsmtSF'], data=df)
#histogram and normal probability plot
sns.distplot(np.sqrt(df[df['TotalBsmtSF']>0]['TotalBsmtSF']), fit=norm);
fig = plt.figure()
res = stats.probplot(np.sqrt(df[df['TotalBsmtSF']>0]['TotalBsmtSF']), plot=plt)
ax = sns.regplot(x=df[df.TotalBsmtSF>0]["SalePrice"], y=np.sqrt(df[df.TotalBsmtSF>0]['TotalBsmtSF']), data=df)
df['TotalBsmtSF']=np.sqrt(df['TotalBsmtSF'])
print ('missing Values')
print(df['BsmtHalfBath'].isna().sum())
pd.Series(df['BsmtHalfBath']).value_counts().plot('bar')
print ('missing Values')
print(df['MSZoning'].isna().sum())
pd.Series(df['MSZoning']).value_counts().plot('bar')
#box plot overallqual/saleprice
var = 'MSZoning'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(10,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)

cleanup_nums = {"MSZoning":     {"C (all)": 1, "RM": 2, "RH":3, "RL":4, "FV":5}}
df.replace(cleanup_nums, inplace=True)  
print ('missing Values')
print(df['Street'].isna().sum())
pd.Series(df['Street']).value_counts().plot('bar')
cleanup_nums = {"Street":     {"Pave": 1, "Grvl": 2}}
df.replace(cleanup_nums, inplace=True) 
print(df['Alley'].isna().sum())
pd.Series(df['Alley']).value_counts().plot('bar')
df=df.drop(columns=['Alley'])
print(df['LotShape'].isna().sum())
pd.Series(df['LotShape']).value_counts().plot('bar')
#box plot overallqual/saleprice
var = 'LotShape'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=dfTrain)
cleanup_nums = {"LotShape":     {"Reg": 1, "IR1": 2, "IR2": 3, "IR3": 4}}
df.replace(cleanup_nums, inplace=True) 
#box plot overallqual/saleprice
var = 'LotShape'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['LandContour'].isna().sum())
pd.Series(df['LandContour']).value_counts().plot('bar')
#box plot 
var = 'LandContour'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
cleanup_nums = {"LandContour":     {"Lvl": 1, "Bnk": 2, "HLS": 3, "Low": 4}}
df.replace(cleanup_nums, inplace=True) 
#box plot 
var = 'LandContour'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['Utilities'].isna().sum())
pd.Series(df['Utilities']).value_counts().plot('bar')
df=df.drop(columns=['Utilities'])
print("na Values")
print(df['LotConfig'].isna().sum())
pd.Series(df['LotConfig']).value_counts().plot('bar')
cleanup_nums = {"LotConfig":     {"Inside": 1, "Corner": 2, "CulDSac": 3, "FR2": 4, "FR3": 5}}
df.replace(cleanup_nums, inplace=True)
#box plot 
var = 'LotConfig'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['LandSlope'].isna().sum())
pd.Series(df['LandSlope']).value_counts().plot('bar')
var = 'LandSlope'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
df = df.copy()
df = pd.get_dummies(df, columns=['LandSlope'], prefix = ['LandSlope'])
df.head()
print("na Values")
print(df['Neighborhood'].isna().sum())
pd.Series(df['Neighborhood']).value_counts().plot('bar')
df = df.copy()
df = pd.get_dummies(df, columns=['Neighborhood'], prefix = ['Neighborhood'])

df.head()
print("na Values")
print(df['Condition1'].isna().sum())
pd.Series(df['Condition1']).value_counts().plot('bar')
cleanup_nums = {"Condition1":     {"Norm": 1, "Feedr": 2, "Artery":3, "RRAn":4,"PosN":5, "RRAe":6, "PosA":7, "RRNn":8, "RRNe":9}}
df.replace(cleanup_nums, inplace=True)
#box plot 
var = 'Condition1'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['Condition2'].isna().sum())
pd.Series(df['Condition2']).value_counts().plot('bar')
cleanup_nums = {"Condition2":     {"Norm": 1, "Feedr": 2, "Artery":3, "RRAn":4,"PosN":5, "RRAe":6, "PosA":7, "RRNn":8, "RRNe":9}}
df.replace(cleanup_nums, inplace=True)
#box plot 
var = 'Condition2'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['BldgType'].isna().sum())
pd.Series(df['BldgType']).value_counts().plot('bar')
#Lets asign 1Fam=1, 2FmCon=2, Duplx=3, TwnhsE=4, TwnhsI=5       
# For each row in the column,
BldgType=[]
for row in df['BldgType']:
        if row =='1Fam':
            BldgType.append(1)
        elif row =='2fmCon':
            BldgType.append(2)
        elif row =='Duplex':
            BldgType.append(3)
        elif row =='TwnhsE':
            BldgType.append(4)
        elif row =='Twnhs':
            BldgType.append(5)
#change asign numbers to column values
df['BldgType']=pd.DataFrame(BldgType)
df['BldgType'].head()
var = 'BldgType'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['HouseStyle'].isna().sum())
pd.Series(df['HouseStyle']).value_counts().plot('bar')
cleanup_nums = {"HouseStyle":     {"1Story": 1, "1.5Unf": 1.5, "1.5Fin":1.8, "2Story":2,"2.5Unf":2.5, "2.5Fin":2.8, "SFoyer":3, "SLvl":4}}
df.replace(cleanup_nums, inplace=True)
#box plot 
var = 'HouseStyle'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['OverallQual'].isna().sum())
pd.Series(df['OverallQual']).value_counts().plot('bar')
#box plot 
var = 'OverallQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['OverallCond'].isna().sum())
pd.Series(df['OverallCond']).value_counts().plot('bar')
#box plot 
var = 'OverallCond'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
(df['YearRemodAdd']-df['YearBuilt']).head()
# Create a list to store the data
YY=df['YearRemodAdd']-df['YearBuilt']
# For each row in the column,
RemodAdd=[]
for row in YY:
        if row == 0:
            RemodAdd.append(0)
        elif row>0:
            RemodAdd.append(1)
df['RemodAdd']=pd.DataFrame(RemodAdd)
df.head()
ax = sns.regplot(x="SalePrice", y="YearBuilt", data=df)
df['YearBuilt'].describe()
bins = [1872, 1954, 1973, 1990, 2000, 2010]
df['binned'] = pd.cut(df['YearBuilt'], bins)
df = df.copy()
df = pd.get_dummies(df, columns=['binned'], prefix = ['binned'])
df.head()
df=df.rename(columns={"binned_(1872, 1954]":"BuiltYear1", "binned_(1954, 1973]":"BuiltYear2", 
                       "binned_(1973, 1990]":"BuiltYear3","binned_(1990, 2000]":"BuiltYear4",
                       "binned_(2000, 2010]":"BuiltYear5"})
df.head()
print("na Values")
print(df['RoofStyle'].isna().sum())
pd.Series(df['RoofStyle']).value_counts().plot('bar')
#I will assign numbers for each type of roof
cleanup_nums = {"RoofStyle":     {"Gable": 1, "Hip": 2, "Flat":3, "Gambrel":4,"Mansard":5, "Shed":6 }}
df.replace(cleanup_nums, inplace=True)
#box plot 
var = 'RoofStyle'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['RoofMatl'].isna().sum())
pd.Series(df['RoofMatl']).value_counts().plot('bar')
#I will assign numbers for each type of roof matarial
cleanup_nums = {"RoofMatl":     {"CompShg": 1, "Tar&Grv": 2, "WdShngl":3, "WdShake":4,"ClyTile":4, "Roll":4, "Membran":4, "Metal":4 }}
df.replace(cleanup_nums, inplace=True)
#box plot 
var = 'RoofMatl'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
df=df.drop(columns=['Exterior2nd'])
print("na Values for Exterior1st")
print(df['Exterior1st'].isna().sum())

pd.Series(df['Exterior1st']).value_counts().plot('bar')

#I will assign numbers for each type of 
cleanup_nums = {"Exterior1st":     {"VinylSd": 1, "HdBoard": 2, "MetalSd":3, "Wd Sdng":4,"Plywood":5, "CemntBd":6, "BrkFace":7, "WdShing":8, "Stucco":9, "AsbShng":10,"Stone":0,"BrkComm":0, "ImStucc":0, "AsphShn":0, "CBlock":0}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'Exterior1st'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['ExterQual'].isna().sum())

pd.Series(df['ExterQual']).value_counts().plot('bar')
#I will assign numbers 
cleanup_nums = {"ExterQual":     {"Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'ExterQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['ExterCond'].isna().sum())

pd.Series(df['ExterCond']).value_counts().plot('bar')
#I will assign numbers 
cleanup_nums = {"ExterCond":     {"Po":1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'ExterCond'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['Foundation'].isna().sum())

pd.Series(df['Foundation']).value_counts().plot('bar')
#I will assign numbers
cleanup_nums = {"Foundation":     {"BrkTil":3, "CBlock":2, "PConc":1, "Slab":4, "Stone":5, "Wood":6}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'Foundation'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['BsmtQual'].isna().sum())
# I will fill na as 0
df['BsmtQual']=df['BsmtQual'].fillna(0)
pd.Series(df['BsmtQual']).value_counts().plot('bar')
#I will assign numbers
cleanup_nums = {"BsmtQual":     {"Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'BsmtQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['BsmtCond'].isna().sum())
# I will fill na as 0
df['BsmtCond']=df['BsmtCond'].fillna(0)
pd.Series(df['BsmtCond']).value_counts().plot('bar')
#I will assign numbers
cleanup_nums = {"BsmtCond":     {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'BsmtCond'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['BsmtExposure'].isna().sum())
# I will fill na as 0
df['BsmtExposure']=df['BsmtExposure'].fillna(0)
pd.Series(df['BsmtExposure']).value_counts().plot('bar')
#I will assign numbers
cleanup_nums = {"BsmtExposure":     {"No": 1, "Mn": 2, "Av": 3, "Gd":4}}
df.replace(cleanup_nums, inplace=True)

#box plot 'Exterior1st'
var = 'BsmtExposure'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
df[['BsmtFinType2']].isna().sum()
# For each row in the column,
BsmtFTM_Flag=[]
x=df['BsmtFinType1']
y=df['BsmtFinType2']
ZZ=x == y
YY=pd.DataFrame(ZZ)
for row in YY[0]:
    if row==True:
        BsmtFTM_Flag.append(1)
    else:
            BsmtFTM_Flag.append(0)
df['BsmtFTM_Flag']=pd.DataFrame(BsmtFTM_Flag)
df.head()
df=df.drop(columns=['BsmtFinType2'])
print("na Values")
print(df['BsmtFinType1'].isna().sum())
# I will fill na as 0
df['BsmtFinType1']=df['BsmtFinType1'].fillna(0)
pd.Series(df['BsmtFinType1']).value_counts().plot('bar')
#I will assign numbers
cleanup_nums = {"BsmtFinType1":     {"Unf": 1, "GLQ": 2, "ALQ": 3, "BLQ":4, "Rec": 5, "LwQ":5}}
df.replace(cleanup_nums, inplace=True)
#box plot 'BsmtFinType1'
var = 'BsmtFinType1'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['Heating'].isna().sum())
pd.Series(df['Heating']).value_counts().plot('bar')
#box plot overallqual/saleprice
var = 'Heating'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
#I will assign numbers
#cleanup_nums = {"Heating":     {"GasA":6, 'Floor':2, 'GasW':5,'Grav':1,  'OthW':4, 'Wall':3}}
#df.replace(cleanup_nums, inplace=True)
#box plot 'BsmtFinType1'
var = 'Heating'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
df = df.copy()
df = pd.get_dummies(df, columns=['Heating'], prefix = ['Heating'])
df.head()
df=df.drop(columns=['Heating_Floor', 'Heating_GasW','Heating_Grav',  'Heating_OthW', 'Heating_Wall'])
df=df.rename(columns = {'Heating_GasA':'GasA_Flag'})
df=df.rename(columns = {'Heating_GasA':'GasA_Flag'})
print("na Values")
print(df['HeatingQC'].isna().sum())
pd.Series(df['HeatingQC']).value_counts().plot('bar')
#I will assign numbers
cleanup_nums = {"HeatingQC":     {"Po": 1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)
#box plot 'BsmtFinType1'
var = 'HeatingQC'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['CentralAir'].isna().sum())
pd.Series(df['CentralAir']).value_counts().plot('bar')
# For each row in the column,
CentralAir=[]
for row in df['CentralAir']:
        if row == 'Y':
            CentralAir.append(1)
        elif row=='N':
            CentralAir.append(0)
df['CentralAir']=pd.DataFrame(CentralAir)
df['CentralAir'].head()
print("na Values")
print(df['Electrical'].isna().sum())
df['Electrical']=df['Electrical'].fillna(0)
pd.Series(df['Electrical']).value_counts().plot('bar')
cleanup_nums = {"Electrical":     {"SBrkr": 5, "FuseA": 4, "FuseF": 3, "FuseP":2, "Mix":1}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'Electrical'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("na Values")
print(df['KitchenQual'].isna().sum())
pd.Series(df['KitchenQual']).value_counts().plot('bar')
cleanup_nums = {"KitchenQual":     {"Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'KitchenQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[2,3,4,5])
print("na Values")
print(df['Functional'].isna().sum())
pd.Series(df['Functional']).value_counts().plot('bar')
cleanup_nums = {"Functional":     {"Typ": 1, "Min1": 2, "Min2": 3, "Mod":4, "Maj1":5, "Maj2":6, "Sev":7}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'Functional'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[1,2,3,4,5,6,7])
print("na Values")
print(df['FireplaceQu'].isna().sum())
print("na Values")
print(df['Fireplaces'].isna().sum())
pd.Series(df['Fireplaces']).value_counts().plot('bar')
df[['Fireplaces', 'FireplaceQu']].head()
df['FireplaceQu']=df['FireplaceQu'].fillna(0)

cleanup_nums = {"FireplaceQu":     {"Po":1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'FireplaceQu'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1,2,3,4,5])
df[df['GarageArea']==0]['GarageArea'].count()
ax = sns.regplot(x="SalePrice", y=np.sqrt(df["GarageArea"]), data=df)
df['GarageArea']=np.sqrt(df['GarageArea'])
df[["GarageFinish","GarageType" ]].isna().sum()
df[df.GarageType.isnull()][['GarageType', "GarageFinish"]].head()
df['GarageFinish']=df['GarageFinish'].fillna(0)
df['GarageType']=df['GarageType'].fillna(0)
pd.Series(df['GarageType']).value_counts().plot('bar')
cleanup_nums = {"GarageType":     {"Attchd":1, "Detchd": 2, "BuiltIn": 3, "CarPort":4, "Basment":5, "2Types":6}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'GarageType'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1,2,3,4,5,6])
pd.Series(df['GarageFinish']).value_counts().plot('bar')
cleanup_nums = {"GarageFinish":     {"Unf":1, "RFn": 2, "Fin": 3}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'GarageFinish'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1,2,3])
df[["GarageQual", "GarageCond" ]].isna().sum()
df['GarageQual']=df['GarageQual'].fillna(0)
df['GarageCond']=df['GarageCond'].fillna(0)
pd.Series(df['GarageQual']).value_counts().plot('bar')
pd.Series(df['GarageCond']).value_counts().plot('bar')
cleanup_nums = {"GarageQual":     {"Po":1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'GarageQual'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1,2,3, 4, 5])
cleanup_nums = {"GarageCond":     {"Po":1, "Fa": 2, "TA": 3, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'GarageCond'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1,2,3, 4, 5])
ax = sns.regplot(x="GarageCond", y="GarageQual", data=df)
df=df.drop(columns=['GarageQual'])
df[["GarageYrBlt" ]].isna().sum()
df['GarageYrBlt']=df['GarageYrBlt'].fillna(0)
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df['HasGrg'] = pd.Series(len(df['GarageYrBlt']), index=df.index)
df['HasGrg'] = 0 
df.loc[df['GarageYrBlt']>0,'HasGrg'] = 1
df=df.drop(columns=['GarageYrBlt']) #ekledim
df[["PavedDrive" ]].isna().sum()
pd.Series(df["PavedDrive"]).value_counts().plot('bar')
cleanup_nums = {"PavedDrive":     {"Y":3, "N": 1, "P": 2}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'PavedDrive'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[1,2,3])
print("NA Pools")
print(df[["PoolQC" ]].isna().sum())
print("0 area Pools")
print(df[df.PoolArea==0]["PoolArea"].count())
df[["PoolQC", "PoolArea" ]].head()
df['PoolQC']=df['PoolQC'].fillna(0)
pd.Series(df["PoolQC"]).value_counts().plot('bar')
cleanup_nums = {"PoolQC":     {"Fa": 2, "Gd":4, "Ex":5}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'PoolQC'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 2, 4, 5])
df=df.drop(columns=['PoolQC'])
print("NA Pools")
print(df[["PoolArea" ]].isna().sum())
print("0 area Pools")
print(df[df.PoolArea==0]["PoolArea"].count())
df[df.PoolArea==0]["PoolArea"].count()
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df['HasPool'] = pd.Series(len(df['PoolArea']), index=df.index)
df['HasPool'] = 0 
df.loc[df['PoolArea']>0,'HasPool'] = 1
#df=df.drop(columns=['PoolArea']) #ekledim
df=df.drop(columns=['PoolArea'])
print("NA")
print(df[["Fence"]].isna().sum())
#I will asign NA as 0
df['Fence']=df['Fence'].fillna(0)
pd.Series(df["Fence"]).value_counts().plot('bar')
cleanup_nums = {"Fence":     {"MnPrv": 1,"GdPrv":2, "GdWo":4,"MnWw":3}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'Fence'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1, 2, 3, 4])
print("NA")
print(df[["MiscFeature"]].isna().sum())
#I will asign NA as 0
df['MiscFeature']=df['MiscFeature'].fillna(0)
pd.Series(df["MiscFeature"]).value_counts().plot('bar')
cleanup_nums = {"MiscFeature":     {"Shed": 2,"Gar2":3, "Othr":1,"TenC":4}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'MiscFeature'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df, order=[0, 1, 2, 3, 4])
print("NA")
print(df[["SaleType"]].isna().sum())

pd.Series(df["SaleType"]).value_counts().plot('bar')
cleanup_nums = {"SaleType":     {"WD": 1,"New":2, "COD":3,"ConLD":4,"ConLw":5, "ConLI":6, "CWD":7, "Oth":8, "Con":9}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'SaleType'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
print("NA")
print(df[["SaleCondition"]].isna().sum())

pd.Series(df["SaleCondition"]).value_counts().plot('bar')
cleanup_nums = {"SaleCondition":     {"Normal": 1,"Partial":2, "Abnorml":3,"Family":4,"Alloca":5, "AdjLand":6}}
df.replace(cleanup_nums, inplace=True)
#box plot overallqual/saleprice
var = 'SaleCondition'
data = pd.concat([df['SalePrice'], df[var]], axis=1)
f, ax = plt.subplots(figsize=(7,5))
fig = sns.boxplot(x=var, y="SalePrice", data=df)
df['OpenPorchSF']=np.sqrt(df['OpenPorchSF'])
df[df.OpenPorchSF==0]['OpenPorchSF'].count()
ax = sns.regplot(x="SalePrice", y="OpenPorchSF", data=df)
df['EnclosedPorch']=np.sqrt(df['EnclosedPorch'])
print("NA")
print(df[["EnclosedPorch"]].isna().sum())
df[df.EnclosedPorch==0]['EnclosedPorch'].count()
df[df.EnclosedPorch>0]['EnclosedPorch'].count()
df=df.drop(columns=['EnclosedPorch'])
print("NA")
print(df[["3SsnPorch"]].isna().sum())
df[df['3SsnPorch']==0]['3SsnPorch'].count()
df=df.drop(columns=['3SsnPorch'])
print("NA")
print(df[["ScreenPorch"]].isna().sum())
df[df['ScreenPorch']==0]['ScreenPorch'].count()
df=df.drop(columns=['ScreenPorch'])
print("NA")
print(df[["WoodDeckSF"]].isna().sum())
df[df['WoodDeckSF']==0]['WoodDeckSF'].count()
df['WoodDeckSF']=np.sqrt(df['WoodDeckSF'])
XX=df['SalePrice']
df=df.drop(columns=['SalePrice'])
df['SalePrice']=pd.DataFrame(XX)
df.head()
#df.to_csv('HPP.csv', index=False)