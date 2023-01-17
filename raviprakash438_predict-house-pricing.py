import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')

print(train_df.shape)
print(test_df.shape)
train_df.head()
train_df.info()
#Lets find the corelation of columns againts SalePrice column and arrange it in ascending order.
plt.figure(figsize=(20,8))
sns.heatmap(train_df.corr())#['SalePrice'].sort_values(ascending=False)
plt.show()
#Lets take all the columns which can help in predicting the sale price based on corelation cofficient.
numericalCols=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt',
     'MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF',
     'BedroomAbvGr','ScreenPorch','PoolArea','KitchenAbvGr','EnclosedPorch','OverallCond']
y=train_df['SalePrice']
#numericalCols=list(train_df.select_dtypes(include=['int64','float64']).columns.drop(['Id','SalePrice']))

#Please refer data_description.txt file to understand the categorical columns.
categoricalCols=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
'RoofStyle','RoofMatl','Exterior1st','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
'BsmtFinType2','Electrical','Heating','HeatingQC','CentralAir','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
                 'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition','MSSubClass']
#categoricalCols=list(train_df.select_dtypes(include=['object']).columns)
#Putting train and test dataset in an array so that data cleaning will be easy.
dataset=[train_df,test_df]
#Plotting top six significant columns which are helping in predition Sale price.
fig, ax=plt.subplots(3,2,figsize=(20,20))
ax[0][0].scatter(x='OverallQual',y='SalePrice',data=train_df)
ax[0][0].set_title('OverallQual vs Sale Price')

ax[0][1].scatter(x='GrLivArea',y='SalePrice',data=train_df)
ax[0][1].set_title('GrLivArea vs SalePrice')

ax[1][0].scatter(x='GarageCars',y='SalePrice',data=train_df)
ax[1][0].set_title('GarageCars vs SalePrice')

ax[1][1].scatter(x='GarageArea',y='SalePrice',data=train_df)
ax[1][1].set_title('GarageArea vs SalePrice')

ax[2][0].scatter(x='TotalBsmtSF',y='SalePrice',data=train_df)
ax[2][0].set_title('TotalBsmtSF vs Sale Price')

ax[2][1].scatter(x='1stFlrSF',y='SalePrice',data=train_df)
ax[2][1].set_title('1stFlrSF vs Sale Price')

plt.show()
#Lets fill the null value.
for data in dataset:
    print('---filling null value')
    #data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].median())
    data.loc[data.LotFrontage.isnull(),'LotFrontage']=data.LotFrontage.median()
    #I am converting GarageYrBlt column to feature column. If year is present then its 1 else 0
    data['GarageYrBlt']= data['GarageYrBlt'].apply(lambda x: 1 if pd.notnull(x) else 0)
    #I am assigning the null value to 0 for MasVnrArea.
    data['MasVnrArea']=data['MasVnrArea'].apply(lambda x: x if pd.notnull(x) else 0)
    #Please refer data_description.txt file to understand with what value we should replace nan in each categorical columns.
    data['Alley']=data['Alley'].apply(lambda x: x if pd.notnull(x) else 'No Alley')
    data['MasVnrType']=data['MasVnrType'].apply(lambda x: x if pd.notnull(x) else 'None')
    data['BsmtQual']=data['BsmtQual'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['BsmtCond']=data['BsmtCond'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['BsmtExposure']=data['BsmtExposure'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['BsmtFinType1']=data['BsmtFinType1'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['BsmtFinType2']=data['BsmtFinType2'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['Electrical']=data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['FireplaceQu']=data['FireplaceQu'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['GarageType']=data['GarageType'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['GarageFinish']=data['GarageFinish'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['GarageQual']=data['GarageQual'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['GarageCond']=data['GarageCond'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['PoolQC']=data['PoolQC'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['Fence']=data['Fence'].apply(lambda x: x if pd.notnull(x) else 'No')
    data['MiscFeature']=data['MiscFeature'].apply(lambda x: x if pd.notnull(x) else 'No')    

#Lets check null value in the columns to insure that we have replace all the null values.
train_df.isnull().sum().max()
#Lets plot graph for all categorical columns agains sale price to understand how significant they are.
#There are 42 categorical columns I have listed in categoricalCols array.
#So I am creating subplots which contains 21 rows and 2 columns
fig,axi=plt.subplots(21,2,figsize=(20,100))
indx=0
leng=len(categoricalCols)
for ax in axi:
    for a in ax:
        if leng>indx:
            a.scatter(x= categoricalCols[indx],y='SalePrice',data=train_df)
            a.set_title('{} vs Sale Price'.format(categoricalCols[indx]))
            indx +=1
plt.show()
#Lets normalize it.
for data in dataset:
    data.replace({'Alley':{'Grvl':0,'No Alley':1,'Pave':0},
                 'BldgType':{'1Fam':1,'2fmCon':0,'Duplex':0,'TwnhsE':0,'Twnhs':0},
                 'RoofStyle':{'Flat':0,'Gable':1,'Gambrel':0,'Hip':1,'Mansard':0,'Shed':0},
                 'BsmtQual':{'Ex':2,'Fa':0,'Gd':1,'No':0,'TA':1},
                 'LotShape':{'IR2':0,'IR3':0,'Reg':1,'IR1':2},
                 'LandContour':{'Bnk':0,'HLS':1,'Low':0,'Lvl':2},
                  'LandSlope':{'Gtl':1,'Mod':0,'Sev':0},
                  'ExterQual':{'Fa':0,'Gd':1,'TA':1,'Ex':2},
                  'Electrical':{'FuseA':0,'FuseF':0,'FuseP':0,'Mix':0,'SBrkr':1},
                  'Heating':{'Floor':0,'GasW':0,'Grav':0,'OthW':0,'Wall':0,'GasA':1},
                  'CentralAir':{'N':0,'Y':1},
                  'KitchenQual':{'Ex':2,'Fa':0,'Gd':1,'No':0,'TA':1},
                  'Functional':{'Maj1':0,'Maj2':0,'Min1':0,'Min2':0,'Mod':0,'Sev':0,'Typ':1},
                  'GarageQual':{'Ex':2,'Fa':0,'Gd':1,'No':0,'Po':0,'TA':1},
                 },inplace=True)
feaCols=['Alley','BsmtQual']
#train_df.BldgType.unique()
#removing the outlier in GrLivArea.    
train_df=train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)
#So after analysing the plots I selected the below columns.
featureCols=['MSZoning','Neighborhood','Condition1','MSSubClass','Foundation']

#Lets take out all the selected numerial columns and feature columns from train dataset.
trainData=train_df[numericalCols+featureCols+feaCols]
#Lets do feature scaling on categorical columns for train data.
trainData=pd.get_dummies(trainData,columns=featureCols+feaCols,drop_first=True)

#Lets take out all the selected numerial columns and feature columns from test dataset.
testData=test_df[numericalCols+featureCols+feaCols]
#Lets do feature scaling on categorical columns for test data.
testData=pd.get_dummies(testData,columns=featureCols+feaCols,drop_first=True)

y=train_df.SalePrice
X=trainData

#Lets print out the shape of train and test dataset after feature scaling.
print('train Data Shape-->{}'.format(trainData.shape))
print('test Data Shape-->{}'.format(testData.shape))
trainCols=trainData.columns
testCols=testData.columns
#find test columns not present in train data
c1=[col for col in testCols if col not in trainCols]
print('Missing column in train data-->',c1)
#MSSubClass_150 is the columns missing in the train database.
#Lets see the values of MSSubClass in train and test dataset.
print(train_df.MSSubClass.unique())
print(test_df.MSSubClass.unique())
#Lets see how many records are there for MSSubClass=150
test_df.loc[test_df['MSSubClass']==150]
#Since there is only one record we will update MSSubClass value. But with what value?
#Understand know that we need to analyse MSSubClass vs SalePrice Plot. I have already plotted it in categorical data visualisation.
#But lets Plot it here.
plt.figure(figsize=(8,6))
plt.scatter(x='MSSubClass',y='SalePrice',data=train_df)
plt.show()
#If we see here there is no value for 150.The nearest category having value is 160. So we will update 150 by  160.
test_df.loc[test_df['MSSubClass'] == 150, 'MSSubClass'] = 160
#Lets check if the data is updated.
test_df.loc[test_df['MSSubClass']==150]
#We need to do the below step again
trainData=train_df[numericalCols+featureCols+feaCols]
trainData=pd.get_dummies(trainData,columns=featureCols+feaCols,drop_first=True)
testData=test_df[numericalCols+featureCols+feaCols]
testData=pd.get_dummies(testData,columns=featureCols+feaCols,drop_first=True)

y=train_df.SalePrice
X=trainData
#Lets print out the shape of train and test dataset after feature scaling.
print('train Data Shape-->{}'.format(trainData.shape))
print('test Data Shape-->{}'.format(testData.shape))
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=0)
y_train=np.log1p(y_train)
#Lets take Liner Regression for predition
lnmodel=LinearRegression()
lnmodel.fit(x_train,y_train)
pred=lnmodel.predict(x_test)
print(mean_absolute_error(y_test,np.expm1(pred)))
#Lets take Ramdom forest for predition
model=RandomForestRegressor(n_estimators=100)
model.fit(x_train,y_train)
pred=model.predict(x_test)
print(mean_absolute_error(y_test,np.expm1(pred)))
#Lets take XGBoots for predition
xmodel=XGBRegressor(n_estimators=2200,learning_rate=0.035,n_jobs=-1,max_depth=2)
xmodel.fit(x_train,y_train,)
xpred=xmodel.predict(x_test)
print(mean_absolute_error(y_test,np.expm1(xpred)))
test_pred=np.expm1(xmodel.predict(testData))
test_pred
sub=pd.DataFrame({'Id':test_df.Id,'SalePrice':test_pred})
sub.to_csv('Submission.csv',index=False)
