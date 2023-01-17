import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', 500)
#read the data

data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
data.head(5)
#Univariate Analysis

data.describe(include='all')


sns.lineplot(x=data['YrSold'],y=data['SalePrice'])
sns.lineplot(x=data['YearBuilt'],y=data['SalePrice'])
sns.lineplot(x=data['MoSold'],y=data['SalePrice'])
sns.lineplot(x=data['OverallQual'],y=data['SalePrice'])
sns.lineplot(x=data['SaleCondition'],y=data['SalePrice'])
sns.distplot(data['SalePrice'])

a=data['SalePrice'].skew()

plt.title("Skew:"+str(a))
#SalePrice(target Variable right skewed hence performed log transform)

sns.distplot(np.log(data['SalePrice']+1))

a=np.log(data['SalePrice']+1).skew()

plt.title("Skew:"+str(a))
num_feat=set(data._get_numeric_data().columns)

feat=set(data.columns)

cat_feat=list(feat-num_feat)

print("total categoricalfeatures : "+str(len(cat_feat)))
y='SalePrice'

for i,j in enumerate(cat_feat):

    

    sns.catplot(x=j, y=y, data=data,alpha=0.5)

    plt.xticks(rotation=90)
#Heatmap to check correlation between variables

corr=data.corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corr)
plt.scatter(data['GrLivArea'],data['SalePrice'],alpha=0.2)

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')
plt.scatter(data['MSSubClass'],data['SalePrice'],alpha=0.2)

plt.xlabel('MSSubClass')

plt.ylabel('SalePrice')
a=data._get_numeric_data().columns

for i in a:

    plt.figure()

    plt.scatter(data[i],np.log(data['SalePrice']),alpha=0.2)

    plt.title(i)
plt.scatter(data['LotFrontage'],data['SalePrice'],alpha=0.2)

plt.xlabel('LotFrontage')

plt.ylabel('SalePrice')

# Observed Outliers
plt.scatter((data['LotArea']),np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('LotArea')

plt.ylabel('SalePrice')

#observed Outliers and non-linear relationship
plt.scatter(np.log(data['LotArea']),np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('log_LotArea')

plt.ylabel('log_SalePrice')
plt.scatter(data['GarageArea'],data['SalePrice'],alpha=0.2)

plt.xlabel('GarageArea')

plt.ylabel('SalePrice')
plt.scatter(data['OverallQual'],data['SalePrice'],alpha=0.2)

plt.xlabel('OverallQual')

plt.ylabel('SalePrice')
plt.scatter(data['TotRmsAbvGrd'],data['SalePrice'],alpha=0.2)

plt.xlabel('TotRmsAbvGrd')

plt.ylabel('SalePrice')
plt.scatter((data['MasVnrArea']),np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('MasVnrArea')

plt.ylabel('SalePrice')
plt.scatter(np.sqrt(data['MasVnrArea']),np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('log_MasVnrArea')

plt.ylabel('log_SalePrice')
plt.scatter(data['OpenPorchSF'],data['SalePrice'],alpha=0.2)

plt.xlabel('OpenPorchSF')

plt.ylabel('SalePrice')
plt.scatter(np.sqrt(data['OpenPorchSF']+1),np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('Square root of OpenPorchSF')

plt.ylabel('log_SalePrice')
plt.scatter(np.log(data['OpenPorchSF']+1),np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('log of OpenPorchSF')

plt.ylabel('log_SalePrice')
plt.scatter(data['CentralAir'],data['SalePrice'],alpha=0.2)

plt.xlabel('CentralAir')

plt.ylabel('SalePrice')
plt.scatter(data['KitchenAbvGr'],data['SalePrice'],alpha=0.2)

plt.xlabel('KitchenAbvGr')

plt.ylabel('SalePrice')
#Feature for age of House

a=(data['YrSold']/100).astype(int)-(data['YearBuilt']/100).astype(int)

b=(data['YearBuilt']%100)-(data['YrSold']%100)

data['age']=a*100-b





sns.lineplot(x=data['age'],y=data['SalePrice'])

plt.xlabel('age')

plt.ylabel('SalePrice')
plt.scatter(x=data['age'],y=data['SalePrice'],alpha=0.2)

plt.xlabel('age')

plt.ylabel('SalePrice')
plt.scatter(x=np.sqrt(data['age']),y=np.log(data['SalePrice']),alpha=0.2)

plt.xlabel('Square root of age')

plt.ylabel('log_SalePrice')
#Total sum of basement first floor and 2nd floor area

data['TotalSF']=data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
#Total Number of bathroooms

data['Total_Bathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +

                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
#Total Porch area

data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +

                              data['EnclosedPorch'] + data['ScreenPorch'] +

                              data['WoodDeckSF'])
#Total no. of baths in basement and on floors

data['BsmntBath']=0.5*data['BsmtHalfBath']+data['BsmtFullBath']

data['bath']=0.5*data['HalfBath']+data['FullBath']
#Age of house since it has been remodeled

a=(data['YrSold']/100).astype(int)-(data['YearRemodAdd']/100).astype(int)

b=(data['YearRemodAdd']%100)-(data['YrSold']%100)

data['remod_age']=a*100-b
sns.lineplot(x=data['remod_age'],y=data['SalePrice'])
#variables which has correlation with SalePrice(Target variable) less than 0.2

corr=data.corr()

a=[]

for i in range(corr.shape[1]):

    if np.abs(corr.iloc[i,corr.columns.get_loc("SalePrice")])<0.2:

        a.append(corr.columns.values[i])



a
data.columns
#variables highly correlated (correlation >0.8)  

a=corr[corr>0.8]

b={}

for i in corr.columns:

    index = corr[i].index[corr[i].apply(lambda x:True if (x>0.8)&(x<1) else False)]

    x=list(index.values)

    if len(x)!=0:

        b[i]=x

    

                                      

b    
#no. of missing values in different columns

data.isna().sum()
data[['LotFrontage','GarageType','GarageFinish','GarageQual','GarageCond','GarageYrBlt']].describe(include='all')
#Imputed missing values with their mean and median values based on their distribution

data['LotFrontage'].fillna(data['LotFrontage'].mean(),inplace=True)

data['MasVnrArea'].fillna(data['MasVnrArea'].median(),inplace=True)

#Filled missing garageyear built with the yearbuilt

data['GarageYrBlt'].fillna(data['YearBuilt'],inplace=True)
data11=data.copy()

data=data11.copy()
#Imputation of categorical variables with their mode value

a=data.isnull().sum()

a=a[a>0]

for i in list(a.index):

    if data[i].dtypes==object:

        data[i].fillna(data[i].mode()[0],inplace=True)

        

    
#No. of rows with negative age

print(data[data['age']<0].shape)

print(data[data['remod_age']<0].shape)
#Dropping rows with negative age

data=data[data['age']>=0]

data=data[data['remod_age']>=0]
#Removed Outliers

data=data[data['GrLivArea']<6000]

data=data[data['LotFrontage']<210]

data=data[data['LotArea']<100000]
data['Pool']=data['PoolArea'].apply(lambda x:1 if x>0 else 0)
#Target encoding of categorical variables

data['GarageCond']=data['GarageCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})

data['BsmtQual']=data['BsmtQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2})

data['ExterQual']=data['ExterQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2})

data['ExterCond']=data['ExterCond'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})

data['BsmtCond']=data['BsmtCond'].replace({'Po':1,'Gd':4,'TA':3,'Fa':2})

data['BsmtExposure']=data['BsmtExposure'].replace({'No':1,'Mn':2,'Av':3,'Gd':4})

data['KitchenQual']=data['KitchenQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2})

data['GarageQual']=data['GarageQual'].replace({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
#Interaction feature

data['Qual']=data['BsmtQual']*data['ExterQual']*data['ExterCond']*data['BsmtCond']*data['BsmtExposure']*data['KitchenQual']*data['GarageQual']
#Interaction feature

data['baseQ']=data['BsmtQual']*data['BsmtCond']*data['BsmtExposure']

data['ExQ']=data['ExterQual']*data['ExterCond']

data['GaraQ']=data['GarageCond']*data['GarageQual']
#Dropping irrelavent columns

#Delete columns with lots of misssing entries

a=['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

data.drop(a,axis=1,inplace=True)



#remove categorical variable

#PoolQC,Utilities,Condition2,Street,MiscFeature,RoofMatl,Heating,Alley,GarageCond

data.drop(['BsmtFinType2','Utilities','Condition2','Street','RoofMatl','Heating','GarageCond','RoofMatl','GarageQual','Functional','Exterior1st','Exterior2nd','Neighborhood','Condition2'],axis=1,inplace=True)



#Remove one of highly correlated variables and variables less correlated with target variable



data.drop(['BsmtFullBath','HalfBath','FullBath','BsmtHalfBath','RoofStyle','3SsnPorch','EnclosedPorch','ScreenPorch','PoolArea','BedroomAbvGr','2ndFlrSF','OpenPorchSF','1stFlrSF','GarageCars'],axis=1,inplace=True)



x=['Id', 'BsmtFinSF2', 'LowQualFinSF', 'MiscVal']

data.drop(x,axis=1,inplace=True)
#Dropping columns with very less contribution

data.drop(['Pool','Total_Bathrooms','YrSold','GarageYrBlt','YearBuilt','YearRemodAdd','MasVnrArea','MasVnrType','MSSubClass','Condition1','HouseStyle','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtFinType1','CentralAir','Electrical','KitchenAbvGr','Fireplaces','GarageType','GarageFinish','SaleType'],axis=1,inplace=True)
#One-hot encoding of remaining categorical variables

num_feat=set(data._get_numeric_data().columns)

feat=set(data.columns)

cat_feat=list(feat-num_feat)

a=set(['BsmtQual','ExterQual','ExterCond','BsmtCond','BsmtExposure','KitchenQual','GarageQual'])

cat_feat=set(cat_feat)-a

len(cat_feat)

b=[]

for i in cat_feat:

    a=pd.get_dummies(data[i],prefix=i)

    b=b+list(a.columns.values)

    data=pd.concat([data,a],axis=1)

    data.drop(i,axis=1,inplace=True)
data['LotArea']=np.log(data['LotArea']+1)
c=[]

for i in b:

    x=data[i].value_counts()

    if x.index[0]==0:

        if x.values[0]>1440:

           c.append(str(i))

data.drop(c,axis=1,inplace=True)        
data3=data.copy()

data3=data.dropna()

Y=data3['SalePrice']

data3.drop(['SalePrice'],inplace=True,axis=1)

X=data3.copy()



#Splitting of data into train and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Fitting the model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression(normalize=True)

regressor.fit(X_train, np.log(y_train))



# Predicting the Test set results

y_pred_test = np.exp(regressor.predict(X_test))

y_pred_train=np.exp(regressor.predict(X_train))
def feature_importance(model,data):

   ser = pd.Series(model.coef_,data.columns).sort_values()

   plt.figure(figsize=(12,12))

   ser.plot(kind='bar')

   return ser
#Feature importance

a=feature_importance(regressor,X)

print(a)
#R Square value of the fit

from sklearn.metrics import r2_score

test=r2_score(y_test,y_pred_test)

print("Test Rsquare"+str(r2_score(y_test,y_pred_test)))

print("Train RSquare"+str(r2_score(y_train,y_pred_train)))
# RMSE(between the logarithm of the predicted value and the logarithm of the observed sales price) so that errors in predicting expensive houses and cheap houses will contribute the result equally.

from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(np.log(y_test),np.log(y_pred_test)))

print("test data rms error : "+str(rms))
from sklearn.metrics import mean_squared_error

rms = np.sqrt(mean_squared_error(np.log(y_train),np.log(y_pred_train)))

print("train_data rms_error: " +str(rms))
adjusted_r_squared = 1 - (1-test)*(len(y_test)-1)/(len(y_test)-X.shape[1]-1)

print("adjusted_r_squared value for test data :"+str(adjusted_r_squared))
plt.figure()

plt.scatter(y_train,y_pred_train,alpha=0.2)

plt.plot(y_train,y_train,alpha=0.3,c='r')

plt.title("Training Data Fit")

plt.xlabel("Sales")

plt.ylabel("Predicted Sales")
plt.figure()

plt.scatter(y_test,y_pred_test,alpha=0.2)

plt.plot(y_test,y_test,alpha=0.3,c='r')

plt.title("Test Data Fit")

plt.xlabel("Sales")

plt.ylabel("Predicted Sales")
a=(y_test-y_pred_test)

sns.distplot(a)

plt.title("Skew:"+str(a.skew()))

plt.title("Residual Plot for Training set ....Skew:"+str(a.skew()))
a=(y_train-y_pred_train)

sns.distplot(a)

plt.title("Skew:"+str(a.skew()))

plt.title("Residual Plot for Training set ....Skew:"+str(a.skew()))
plt.scatter(y_train-y_pred_train,y_train,alpha=0.2)

plt.title("Residual Vs  Actual SalePrice for train data")
plt.scatter(y_test-y_pred_test,y_test,alpha=0.2)

plt.title("Residual Vs  Actual SalePrice for test data")