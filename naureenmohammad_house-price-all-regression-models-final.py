import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split



import os

iowa_file_path1 = '../input/home-data-for-ml-course/train.csv'

iowa_file_path2 ="../input/home-data-for-ml-course/test.csv"



home_data = pd.read_csv(iowa_file_path1)

test_data=pd.read_csv(iowa_file_path2)



print(home_data.isnull().sum().sum())


train_data=home_data.drop(['SalePrice'], axis=1)

target= home_data['SalePrice']

Id=test_data['Id']


fig = plt.figure(figsize = (15,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.hist(home_data['SalePrice'])

home_data['SalePrice'].describe()
from matplotlib import pyplot as plt

inputs= pd.concat([train_data,target], axis=1)

corr = inputs[inputs.SalePrice>1].corr()

top_corr_cols = corr[abs((corr.SalePrice)>=0.5)].SalePrice.sort_values(ascending=False).keys()



# print(top_corr_cols)



top_corr = corr.loc[top_corr_cols, top_corr_cols]

dropSelf = np.zeros_like(top_corr)

# print(dropSelf)



dropSelf[np.triu_indices_from(dropSelf)] = True

plt.figure(figsize=(20,20))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),

            annot=True, fmt=".2f", mask=dropSelf)

sns.set(font_scale=0.5)

plt.show()

del corr, dropSelf, top_corr
plt.figure(figsize=(20,10))

plt.xlabel('OverallQual', fontsize=20)

plt.ylabel('SalePrice', fontsize=20)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)



sns.boxplot(x= home_data['OverallQual'], y= target)



plt.figure(figsize=(20,10))

plt.xlabel('GrLivArea', fontsize=20)

plt.ylabel('SalePrice', fontsize=20)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.scatter(x= home_data['GrLivArea'], y= target)
data= home_data.loc[(home_data['GrLivArea']>4000) & (home_data['SalePrice']< 200000)]

print(data['Id'])            

           
# print(home_data.describe())

print('Info of input:\n',home_data.info())

print('Info of test data:\n',test_data.info())
# Quality and condition columns are to be numerically encoded 



cleanup_nums = {"ExterQual":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},

            "ExterCond":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},

            "HeatingQC":       {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},

            "KitchenQual":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1},

            "CentralAir":     {"Y": 1, "N": 0},

# below features also have missing values that can be replaced by 0

    "BsmtQual":      {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},

    "BsmtCond":      {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},

    "BsmtExposure":    {"Gd":4, "Av":3 , "Mn":2, "No":1, "NA":0},

    "BsmtFinType1":    {"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6},

    "BsmtFinType2":    {"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6},

    "FireplaceQu":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},

    "GarageQual":      {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},

    "GarageCond":      {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},

    'GarageFinish':    {'Fin':3,'RFn':2, 'Unf':1, 'None':0},

    "PoolQC":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}}

train_data.replace(cleanup_nums, inplace=True)

test_data.replace(cleanup_nums, inplace=True)



train_data[['PoolQC','FireplaceQu','GarageQual','GarageCond',

            'GarageFinish','BsmtQual','BsmtCond',"BsmtExposure","BsmtFinType1",

            "BsmtFinType2"]]=train_data[['PoolQC','FireplaceQu', 'GarageQual',

                        'GarageCond','GarageFinish','BsmtQual',

                        'BsmtCond', "BsmtExposure","BsmtFinType1",

                                         "BsmtFinType2"]].fillna(0)



test_data[['PoolQC','FireplaceQu','GarageQual','GarageCond',

           'GarageFinish','BsmtQual','BsmtCond', "BsmtExposure","BsmtFinType1",

           "BsmtFinType2"]]=test_data[['PoolQC','FireplaceQu', 'GarageQual',

                                'GarageCond','GarageFinish',

                                'BsmtQual','BsmtCond',"BsmtExposure",

                                "BsmtFinType1","BsmtFinType2"]].fillna(0)



print(train_data.info())

print(test_data.info())
print(train_data['Alley'].unique(),'\n',train_data['Alley'].describe(),'\n')



print(train_data['Utilities'].unique(),'\n',train_data['Utilities'].describe(),'\n')



print(train_data['MasVnrType'].unique(),'\n',train_data['MasVnrType'].describe(),'\n')



print(train_data['Fence'].unique(),'\n',train_data['Fence'].describe(),'\n')



print(train_data['MiscFeature'].unique(),'\n',train_data['MiscFeature'].describe(),'\n')



print(train_data['Functional'].unique(),'\n',train_data['Functional'].describe(),'\n')



print(train_data["GarageType"].unique(),'\n',train_data["GarageType"].describe(),'\n')





train_data.drop(['Utilities'], axis=1, inplace=True)

test_data.drop(['Utilities'], axis=1, inplace=True)

# print(train_data['LotFrontage'].head())



train_data[['Alley','MiscFeature','Fence','MasVnrType',"GarageType"]]=train_data[['Alley', 'MiscFeature',

                        'Fence', 'MasVnrType',"GarageType"]].fillna("None")



test_data[['Alley', 'MiscFeature','Fence', 'MasVnrType',"GarageType"]]=test_data[['Alley', 'MiscFeature',

                     'Fence', 'MasVnrType',"GarageType"]].fillna("None")





data=train_data[['Alley', 'MiscFeature','Fence', 'MasVnrType', 'GarageFinish']]

j=1

fig=plt.figure(figsize=(15,10))

for i in data.columns:

    if j<=5:

        ax1 = fig.add_subplot(2,3,j)

        sns.barplot(data= data, x=i, y=target, ax = ax1, label=i)

        plt.xlabel(i,fontsize=14)

        ax1.xaxis.set_ticks_position('top')

        plt.xticks(rotation='vertical', fontsize=12) 

        plt.yticks(fontsize=8)

        

    j+=1

ax1 = fig.add_subplot(2,3,6)  

sns.boxplot(x= home_data['GarageFinish'], y= target)

plt.xticks(fontsize=14)

plt.yticks(fontsize=8)

plt.xlabel('Barplot of GarageFinish', fontsize=14)





print(test_data['Exterior1st'].unique(),'\n',test_data['Exterior1st'].describe(),'\n')



print(test_data['Exterior2nd'].unique(),'\n',test_data['Exterior2nd'].describe(),'\n')



print(test_data['SaleType'].unique(),'\n',test_data['SaleType'].describe())
train_data.loc[train_data['GarageYrBlt'].isnull(),

               'GarageYrBlt']=train_data['YearBuilt']



test_data.loc[test_data['GarageYrBlt'].isnull(),

              'GarageYrBlt']=test_data['YearBuilt']
# looking for id with no garage

data1= test_data.loc[(test_data['GarageArea'].isnull())]

print(data1['Id'])

test_data.loc[1116,'GarageCars']=0

print(test_data.loc[1116,'GarageType'],'\n')

#looking for id with no basement

data2= test_data.loc[((test_data['BsmtFinSF1'].isnull())&(test_data['BsmtFinSF2'].isnull())&

                      (test_data['BsmtFullBath'].isnull())&(test_data['BsmtHalfBath'].isnull()))]



print(data2[['Id','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

        'BsmtFinType2','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','BsmtUnfSF',

             'BsmtFullBath','BsmtHalfBath']],'\n')



## updating values for no basement condition

test_data.loc[660 ,'BsmtFullBath']=0

test_data.loc[660 ,'BsmtHalfBath']=0

test_data.loc[660 ,'BsmtFinSF1']=0

test_data.loc[660 ,'BsmtFinSF2']=0

test_data.loc[660 ,'TotalBsmtSF']=0

test_data.loc[660 ,'BsmtUnfSF']=0
data3= test_data.loc[(test_data['MasVnrType']=='None')]

print(data3['MasVnrArea'].isnull().sum(),data3[['MasVnrType','MasVnrArea']].shape )


test_data.loc[test_data['MasVnrType']=='None', 'MasVnrArea']=0

test_data['BsmtFullBath']=test_data['BsmtFullBath'].fillna(0)

test_data['BsmtHalfBath']=test_data['BsmtHalfBath'].fillna(0)
print(train_data.info())

print(test_data.info())
#collecting categorical data columns



category=(train_data.select_dtypes(include=['object']).copy()).columns

non_category=list( set(train_data.columns) - set(category))



print('non_categorical columns: \n',non_category)

print('categorical columns: \n',category )



train_data[category]=train_data[category].fillna(train_data[category].mode().iloc[0])

test_data[category]=test_data[category].fillna(test_data[category].mode().iloc[0])



train_data[non_category]= train_data[non_category].fillna(train_data[non_category].mean().iloc[0])

test_data[non_category]= test_data[non_category].fillna(test_data[non_category].mean().iloc[0])



print(train_data.info())

print(test_data.info())
data1 = train_data[['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

            'Foundation', 'Heating', 'Electrical', 'PavedDrive',

                'SaleType', 'SaleCondition','HouseStyle','GarageType','Functional']]



data1=pd.concat([data1,target],axis=1)

# print(data.info())



i = 1

fig = plt.figure(figsize = (20,50))

for c in list(data1.columns):

    if i <= 3:

        if c != 'SalePrice':

            ax1 = fig.add_subplot(8,3,i)

            sns.countplot(data = data1, x=c, ax = ax1)

            plt.xlabel(c,fontsize=12)

            ax1.xaxis.set_ticks_position('top')

            plt.xticks(rotation='vertical',fontsize=14)

            plt.yticks(fontsize=14)



            

            ax2 = fig.add_subplot(8,3,i+3)

            sns.boxplot(data=data1, x=c, y='SalePrice', ax=ax2)

            plt.xlabel(c,fontsize=12)

            ax1.xaxis.set_ticks_position('top')

            plt.xticks(rotation='vertical',fontsize=14)

            plt.yticks(fontsize=14)



       

    i = i +1

    if i == 4: 

        fig = plt.figure(figsize = (20, 50))

        i =1
train_data[['MSSubClass_o','YrSold_o','MoSold_o']]=train_data[['MSSubClass',

                                             'YrSold','MoSold']].astype('O')

test_data[['MSSubClass_o','YrSold_o','MoSold_o']]=test_data[['MSSubClass',

                                            'YrSold','MoSold']].astype('O')



final_categorical_cols=list(category) +['MSSubClass_o','YrSold_o','MoSold_o']

print(final_categorical_cols)



print(train_data.info())

print(test_data.info())
print(train_data['MSSubClass_o'].unique(),'\n',train_data['MSSubClass_o'].describe())

print(train_data['YrSold_o'].unique(),'\n',train_data['YrSold_o'].describe())

print(train_data['MoSold_o'].unique(),'\n',train_data['MoSold_o'].describe())

total_data =pd.concat([train_data,test_data],axis=0)

len=train_data.shape[0]

print('total length:',len)

non_object_data=pd.get_dummies(total_data, columns=final_categorical_cols,drop_first=True)



train_data_dummy= non_object_data[0:len]

test_data_dummy=non_object_data[len:]

print('all final columns after OneHotEncoding:\n',train_data.columns)



# data.head()



print(train_data_dummy.shape,test_data_dummy.shape)

test_data.isnull().sum().sum()
from matplotlib import pyplot as plt



inputs=pd.concat([home_data['SalePrice'],train_data_dummy],axis=1)

corr = inputs[inputs.SalePrice>1].corr()

top_corr_cols = corr[abs((corr.SalePrice)>=0.5)].SalePrice.sort_values(ascending=False).keys()



# print(top_corr_cols)



top_corr = corr.loc[top_corr_cols, top_corr_cols]

dropSelf = np.zeros_like(top_corr)

# print(dropSelf)



dropSelf[np.triu_indices_from(dropSelf)] = True

plt.figure(figsize=(20,20))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(top_corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)

sns.set(font_scale=1)

plt.show()

del corr, dropSelf, top_corr
from sklearn.ensemble import RandomForestRegressor



input_data= train_data_dummy.drop(['Id'], axis=1)

test_data_dummy=test_data_dummy.drop(['Id'], axis=1)



rfr_imp= RandomForestRegressor(n_estimators=100)

rfr_imp.fit(input_data, target)



importance =pd.Series(rfr_imp.feature_importances_, index=input_data.columns)

importance.nlargest(15).plot(kind='bar')
area=['GrLivArea','GarageArea','GarageCars','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF',

     '2ndFlrSF','FullBath','HalfBath','WoodDeckSF','OpenPorchSF','EnclosedPorch','Fireplaces',

      'LowQualFinSF','TotRmsAbvGrd']

corr = train_data[area].corr()

plt.figure(figsize=(10,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f")

sns.set(font_scale=1)

plt.show()

plt.figure(figsize= (100, 50))

train_data['LowQualFinSF'].hist(bins= int(180/5),grid=True,alpha=1,color='gray',label='LowQualFinSF')

train_data['WoodDeckSF'].hist(bins =int(180/5), grid=True,alpha=1,color= 'k',label='WoodDeckSF')

train_data['OpenPorchSF'].hist(bins =int(180/5),grid=True,alpha=1,color='purple',label= 'OpenPorchSF')

train_data['MasVnrArea'].hist(bins =int(180/5),grid=True,alpha=0.5,color='Yellow',label='MasVnrArea')

train_data['GrLivArea'].hist(bins = int(180/5), grid=True, alpha=1, color= 'Red', label= 'GrLivArea')

train_data['1stFlrSF'].hist(bins = int(180/5),grid=True, alpha=0.5, color= 'Blue', label= '1stFlrSF')

train_data['2ndFlrSF'].hist(bins = int(180/5),grid=True, alpha=0.7, color= 'green', label= '2ndFlrSF')

plt.legend(fontsize= 80)

plt.xticks(fontsize= 80)

plt.yticks(fontsize= 80)

train_data.loc[train_data['LowQualFinSF']>0,['GrLivArea','1stFlrSF','2ndFlrSF','LowQualFinSF']].head()
train_data['2ndFlr']=np.where(train_data['2ndFlrSF']==0,0,1)

test_data['2ndFlr']=np.where(test_data['2ndFlrSF']==0,0,1)

print(train_data[['2ndFlr','2ndFlrSF']])
train_data=train_data.drop(['1stFlrSF','2ndFlrSF','LowQualFinSF'],axis=1)



test_data=test_data.drop(['1stFlrSF','2ndFlrSF','LowQualFinSF'],axis=1)
train_data=train_data.loc[(train_data['GrLivArea'] < 4500)]

target1=home_data.loc[(home_data['GrLivArea'] < 4500)]

target=target1['SalePrice']

print(train_data.shape,target.shape)
quality=['OverallQual','OverallCond','YearBuilt','ExterQual','BsmtQual',

        'KitchenQual','FireplaceQu','GarageQual','PoolQC']

corr = (pd.concat([train_data[quality], target],axis=1)).corr()

plt.figure(figsize=(10,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f")

sns.set(font_scale=1)

plt.show()



data1=train_data[quality]

i = 1

fig = plt.figure(figsize = (20,10))

for c in list(quality):

    if c!='YearBuilt':

        if i <= 4:

            ax1 = fig.add_subplot(2,4,i)

            sns.countplot(data = data1, x=c, ax = ax1, label= c)

            plt.xticks(rotation='vertical')            

        i = i +1

        if i == 5: 

            fig = plt.figure(figsize = (20, 10))

            i =1
train_data['Pool']=np.where(train_data['PoolArea']==0, 0,1)

test_data['Pool']=np.where(test_data['PoolArea']==0, 0,1)

train_data=train_data.drop(['PoolQC'],axis=1)

test_data=test_data.drop(['PoolQC'],axis=1)
basement=['BsmtHalfBath','BsmtFullBath','TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinType2',

          'BsmtFinSF1','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual']

corr = (pd.concat([train_data[basement],home_data['SalePrice']], axis=1)).corr()

plt.figure(figsize=(10,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f")

sns.set(font_scale=1)

plt.show()



data1=train_data[basement]

i = 1

fig = plt.figure(figsize = (20,10))

for c in list(basement):

    if c not in list(['TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1']):

        if i <= 4:

            ax1 = fig.add_subplot(2,4,i)

            sns.countplot(data = data1, x=c, ax = ax1, label= c)

            plt.xticks(rotation='vertical')            

        i = i +1

        if i == 5: 

            fig = plt.figure(figsize = (20, 10))

            i =1
train_data[['TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1']].head()
train_data['BsmtFin2']=np.where(train_data['BsmtFinSF2']==0,0,1)

test_data['BsmtFin2']=np.where(test_data['BsmtFinSF2']==0,0,1)





train_data['BsmtFin1']=np.where(train_data['BsmtFinSF1']==0,0,1)

test_data['BsmtFin1']=np.where(test_data['BsmtFinSF1']==0,0,1)



print(train_data[['BsmtFin2','BsmtFinSF2','BsmtFin1','BsmtFinSF1']].head())
train_data=train_data.drop(['BsmtUnfSF', 'BsmtFinSF2','BsmtFinSF1'],axis=1)

test_data=test_data.drop(['BsmtUnfSF', 'BsmtFinSF2','BsmtFinSF1'],axis=1)
garage= ['GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','GarageYrBlt']

data1=pd.concat([train_data[garage],train_data['GarageType'],home_data['SalePrice']],axis=1)

corr = data1.corr()

plt.figure(figsize=(10,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f")

sns.set(font_scale=1)

plt.show()



i = 1

fig = plt.figure(figsize = (20,5))

for c in data1.columns:

    if c not in list(['GarageArea','GarageYrBlt']):

        if i <= 5:

            ax1 = fig.add_subplot(1,5,i)

            sns.countplot(data = data1, x=c, ax = ax1, label= c)

            plt.xticks(rotation='vertical')  

        i = i +1

train_data=train_data.drop(['GarageCond', 'GarageArea'],axis=1)

test_data=test_data.drop(['GarageCond', 'GarageArea'],axis=1)
train_data['TotalBathrooms']= (train_data['FullBath']+0.5*train_data['HalfBath']+

                                train_data['BsmtFullBath']+0.5*train_data['BsmtHalfBath'])

print(train_data['TotalBathrooms'].head())

plt.scatter(x= train_data['TotalBathrooms'], y= target )

plt.show()
test_data['TotalBathrooms']= (test_data['FullBath']+0.5*test_data['HalfBath']+

                    test_data['BsmtFullBath']+0.5*test_data['BsmtHalfBath'])

train_data['Age']= train_data['YrSold']-train_data['YearRemodAdd']

train_data['Remodeled']= np.where(train_data['YearRemodAdd']-train_data['YearBuilt']==0, 0, 1)

train_data['Isnew']=np.where(train_data['YrSold']-train_data['YearBuilt']==0, 1, 0)

# print(input_data[['Remodeled','Age']].head())



correlation=pd.concat([train_data['Age'],target],axis=1).corr()

print(correlation)



plt.figure(figsize=(15,10))

plt.subplot(221)

plt.scatter(x=train_data['Age'], y= target,s=1)

plt.subplot(222)

sns.barplot(x=train_data['Remodeled'],y= target)

plt.axhline(y=178000,linewidth=1, color='k')

plt.subplot(223)

sns.barplot(x=train_data['Isnew'],y= target)

plt.axhline(y=178000,linewidth=1, color='k')

plt.show()
test_data['Age']= test_data['YrSold']-test_data['YearRemodAdd']

test_data['Remodeled']= np.where(test_data['YearRemodAdd']-test_data['YearBuilt']==0, 0, 1)

test_data['Isnew']=np.where(test_data['YrSold']-test_data['YearBuilt']==0, 1, 0)



print(test_data[['Remodeled','Age']])
train_data=train_data.drop(['YearBuilt','YearRemodAdd'],axis=1)

test_data=test_data.drop(['YearBuilt','YearRemodAdd'],axis=1)
data4=pd.concat([target,train_data['Neighborhood']],axis=1)



data_41 = data4.groupby('Neighborhood', as_index=False)['SalePrice'].mean()

data_41=data_41.sort_values(['SalePrice','Neighborhood'], ascending=[1,0])



data_42 = data4.groupby('Neighborhood', as_index=False)['SalePrice'].median()

data_42=data_42.sort_values(['SalePrice','Neighborhood'], ascending=[1,0])



# print(data_41.head(),data_42.head()) 

plt.figure(figsize=(30,20))

plt.subplot(211)

plt.xlabel('mean sale price for each neighborhood', fontsize=18)

sns.barplot(x=data_41['Neighborhood'],y= data_41['SalePrice'])

plt.axhline(y=110000,linewidth=1, color='k')

plt.axhline(y=240000,linewidth=1, color='k')

plt.xticks(rotation='vertical', fontsize=18) 

plt.yticks(fontsize=18) 



plt.subplot(212)

sns.barplot(x=data_42['Neighborhood'],y= data_42['SalePrice'])

plt.axhline(y=110000,linewidth=1, color='k')

plt.axhline(y=240000,linewidth=1, color='k')

plt.xticks(rotation='vertical', fontsize=18) 

plt.xlabel('median sale price', fontsize=18)

plt.yticks(fontsize=18) 

plt.show()

print(data_42['Neighborhood'].unique(),'\n',train_data['Neighborhood'].unique())
mapping= {'MeadowV':0, 'IDOTRR':0, 'BrDale':0,'StoneBr':2, 'NoRidge':2, 'NridgHt':2}

train_data['Neighborhood_bin']=train_data['Neighborhood'].map(mapping)

train_data['Neighborhood_bin']=train_data['Neighborhood_bin'].fillna('1')



test_data['Neighborhood_bin']=test_data['Neighborhood'].map(mapping)

test_data['Neighborhood_bin']=test_data['Neighborhood_bin'].fillna('1')

print(train_data['Neighborhood_bin'].value_counts(),'\n', test_data['Neighborhood_bin'].value_counts())
train_data['TotalPorchArea']= (train_data['WoodDeckSF']+train_data['OpenPorchSF']+

         train_data['EnclosedPorch']+ train_data['3SsnPorch']+train_data['ScreenPorch'])

plt.figure(figsize=(10,10))

plt.scatter(x=train_data['TotalPorchArea'], y= target,s=3)

print(train_data[['TotalPorchArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch',

                 'ScreenPorch']].head()) 

correlation=(pd.concat([train_data[['TotalPorchArea','WoodDeckSF','OpenPorchSF',

            'EnclosedPorch','3SsnPorch','ScreenPorch']],home_data['SalePrice']],

                       axis=1)).corr()

print(correlation)

# x=train_data.loc[(train_data['TotalPorchArea'])==0]



test_data['TotalPorchArea']=(test_data['WoodDeckSF']+test_data['OpenPorchSF']+

  test_data['EnclosedPorch']+ test_data['3SsnPorch']+test_data['ScreenPorch'])


train_data['Has3SSNPorch']=np.where(train_data['3SsnPorch']==0,0,1)

test_data['Has3SSNPorch']=np.where(test_data['3SsnPorch']==0,0,1)



# print(train_data[['TotalPorchArea','Has3SSNPorch','3SsnPorch']].head())



train_data.drop(['3SsnPorch'],axis=1)

test_data.drop(['3SsnPorch'],axis=1)
train_data['TotalArea']=train_data['GrLivArea']+train_data['TotalBsmtSF']

plt.figure(figsize=(10,5))

plt.scatter(x=train_data['TotalArea'],y=target,s=3)
 

correlation=(pd.concat([train_data['TotalArea'],target],axis=1)).corr()

print(correlation)
test_data['TotalArea']=test_data['GrLivArea']+test_data['TotalBsmtSF']



drop=['YrSold','MoSold','MSSubClass', 'GarageYrBlt','TotalBsmtSF',

      'TotRmsAbvGrd']

train_data=train_data.drop(drop,axis=1)

test_data=test_data.drop(drop,axis=1)

print(train_data.columns, train_data.shape)
category=(train_data.select_dtypes(include=['object']).copy()).columns

category=list(category)+['OverallQual', 'OverallCond']

dummy=list(set(category)-set(['MoSold','YrSold']))

non_category=list( set(train_data.columns) - set(category))



print((category))
skewValue = train_data[non_category].skew(axis=0)

data=[]

for i,index in enumerate(skewValue):

    if abs(index)>0.8:

        data.append(non_category[i])

print(print(skewValue.head()),data)

print(train_data.isnull().sum().sum(),test_data.isnull().sum().sum())



train_data[data]=np.log1p(train_data[data])

target=np.log1p(target)

target.hist(bins = 40)

print(train_data[data].head())
test_data[data]=np.log1p(test_data[data])
total_data =pd.concat([train_data,test_data],axis=0)

len=train_data.shape[0]

print('total length:',len)

total_dummy=pd.get_dummies(total_data, columns=dummy,drop_first=True)



train_data_final= total_dummy[0:len]

test_data_final=total_dummy[len:]

print(train_data_final.shape, target.shape)


from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import xgboost as xgb

from xgboost import XGBRegressor



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Ridge, Lasso,ElasticNet

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, make_scorer  

from sklearn import metrics 

from sklearn.model_selection import cross_val_score,KFold

 



kfolds = KFold(n_splits=5, shuffle=True, random_state=42)



def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X,y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)

scaler = RobustScaler()

robust_scaled_df = scaler.fit_transform(train_data_final)

robust_scaled_test = scaler.fit_transform(test_data_final)
# # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1,10,20,50,100,150,200]

# param_rcv={'alpha':np.arange(1,5,1)}

# gs_rcv=GridSearchCV(Ridge(),param_rcv,cv=kfolds,

#                     scoring='neg_mean_squared_error',verbose=3)

# gs_rcv.fit(robust_scaled_df,target)  

# print('optimal estimator: %s' % gs_rcv.best_estimator_)

# rmse1 = cv_rmse(gs_rcv.best_estimator_, robust_scaled_df, target)

# print("Ridge score: {:.4f} ({:.4f})\n".format(rmse1.mean(),rmse1.std()))





# # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1,10,20,50,100,150,200],

# param_lcv={'max_iter':np.arange(12000,15000,1000),

#            'alpha':[1e-5, 1e-4, 1e-3, 1e-2]}

# gs_lcv=GridSearchCV(Lasso(),param_lcv,scoring='neg_mean_squared_error',

#                     verbose=3,cv=kfolds)

# gs_lcv.fit(robust_scaled_df,target)     

# print('optimal estimator: %s' % gs_lcv.best_estimator_)

# rmse2 = cv_rmse(gs_lcv.best_estimator_, robust_scaled_df, target)

# print("Lasso score: {:.4f} ({:.4f})\n".format(rmse2.mean(),rmse2.std()))

# # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1,10,20,50,100,150,200]

# param_encv={'max_iter':[5e5,1e6,5e6,1e7,5e7,1e8],

#             'alpha':[1e-5, 1e-4, 1e-3, 1e-2],

#             'l1_ratio':np.arange(0.1,0.5,0.1)}

# gs_encv=GridSearchCV(ElasticNet(),param_encv,scoring='neg_mean_squared_error',

#                      verbose=3,cv=kfolds)

# gs_encv.fit(robust_scaled_df,target)  

# print('optimal estimator: %s' % gs_encv.best_estimator_)

# rmse3 = cv_rmse(gs_encv.best_estimator_, robust_scaled_df, target)

# print("ElasticNet score: {:.4f} ({:.4f})\n".format(rmse3.mean(),rmse3.std()))
# param_svr={'C':np.arange(20,30,1),

#            'epsilon':[1e-5,1e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1],

#             'gamma':[1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1]}

# gs_svr=GridSearchCV(SVR(),param_svr,scoring='neg_mean_squared_error',

#                     cv=kfolds,verbose=3)

# gs_svr.fit(robust_scaled_df,target)  

# print('optimal estimator: %s' % gs_svr.best_estimator_)

# rmse4 = cv_rmse(gs_svr.best_estimator_, robust_scaled_df, target)

# print("svr score: {:.4f} ({:.4f})\n".format(rmse4.mean(),rmse4.std()))
# # 'n_estimators':np.arange(100,2000,200),'l_rate':[1e-5,1e-4,1e-3,1e-2,1e-1],

# # 'colsample_bytree':np.arange(0.1,1, 0.2)}

# param_xgb={'n_estimators':[5000],'booster': ["gbtree"],

#            'objective':['reg:squarederror'],           

#            'learning_rate':[0.01],'max_depth':[7],

#           'min_child_weight':[5],'subsample':[0.7],'reg_alpha':[0],

#           'colsample_bytree':[0.7],'gamma':[0],

#            'reg_lambda':[0.8]}

# #            

# gs_xgb=GridSearchCV(XGBRegressor(),param_grid=param_xgb,

#                     scoring='neg_mean_squared_error',cv=3,verbose=3)

# gs_xgb.fit(robust_scaled_df,target)  

# print('optimal estimator: %s' % gs_xgb.best_estimator_)

# rmse5 = cv_rmse(gs_xgb.best_estimator_, robust_scaled_df, target)

# print("xgb score: {:.4f} ({:.4f})\n".format(rmse5.mean(),rmse5.std()))
# param_gbr = {

#              'learning_rate': [0.1],'n_estimators': [400],'verbose':[0],

#             'min_samples_split' :[41],'min_samples_leaf': [8],

#             'subsample': [0.6],'max_depth':[4],

#             'max_features':[9]  

#              }

# #  



# gs_gbr=GridSearchCV(GradientBoostingRegressor(),param_grid=param_gbr,

#                     scoring='neg_mean_squared_error',cv=5,verbose=3)

# gs_gbr.fit(robust_scaled_df,target)  

# print('optimal estimator: %s' % gs_gbr.best_estimator_)



# rmse6 = cv_rmse(gs_gbr.best_estimator_, robust_scaled_df, target)

# print("gbr score: {:.4f} ({:.4f})\n".format(rmse6.mean(),rmse6.std()))
ridge=Ridge(alpha=4.0)

lasso=Lasso(alpha=0.0001, max_iter=12000)

elasticnet=ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=500000.0)

svr=SVR(C=21, epsilon=0.01, gamma=0.001)



xgboost=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

        colsample_bynode=1,validate_parameters=1, verbosity=None,

            colsample_bytree=0.7, gamma=0, gpu_id=-1,importance_type='gain',

        learning_rate=0.01, max_delta_step=0, max_depth=7,min_child_weight=5,

n_estimators=5000, n_jobs=0, num_parallel_tree=1, random_state=0,reg_alpha=0,

    reg_lambda=0.8,scale_pos_weight=1, subsample=0.7,tree_method='exact')



gbr=GradientBoostingRegressor(max_depth=4, max_features=9, min_samples_leaf=8,

                        min_samples_split=41, n_estimators=400,subsample=0.6)

elastic_model = elasticnet.fit(robust_scaled_df,target)



lasso_model = lasso.fit(robust_scaled_df,target)



ridge_model = ridge.fit(robust_scaled_df,target)



svr_model = svr.fit(robust_scaled_df,target)



xgb_model = xgboost.fit(robust_scaled_df,target)



gbr_model=gbr.fit(robust_scaled_df,target)

print(target)
from mlxtend.regressor import StackingCVRegressor

stack_gen=StackingCVRegressor(regressors=[ridge,lasso,elasticnet,

                                          xgboost,svr,gbr],

                 meta_regressor=xgboost,use_features_in_secondary=True)



stack_model=stack_gen.fit(robust_scaled_df,target)



scores = cross_val_score(stack_gen,robust_scaled_df,target, cv=10)



print(scores.mean(), scores.std())
average_output= (0.1*elastic_model.predict(robust_scaled_test)+

                 0.02*lasso_model.predict(robust_scaled_test)+

                    0.01*ridge_model.predict(robust_scaled_test)+

                       0.25* svr_model.predict(robust_scaled_test)+

                         0.6* stack_model.predict(robust_scaled_test)+

                            0.01*xgb_model.predict(robust_scaled_test)+

                            0.01*gbr_model.predict(robust_scaled_test))

final_output= np.expm1(average_output)

print(final_output,average_output)
test_preds=np.around(final_output,1)

print(test_preds)



output = pd.DataFrame({'Id':Id.astype('Int32'),

                      'SalePrice': test_preds})

output.to_csv('house_pred.csv', index=False)
