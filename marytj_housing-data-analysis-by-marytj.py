# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, r2_score





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



house_data_train=pd.read_csv('../input/train.csv')

house_data_test=pd.read_csv('../input/test.csv')

#function to fill missing values

def fillmissingval(df,coltofill,valtofill):

        df[coltofill].fillna(valtofill,inplace=True)

        

    

#print(house_data_train.columns.values)



print(house_data_train.describe(include='all'))

print(house_data_train.info())

#house_data_train[['YearBuilt', 'LotArea','SalePrice']].sort_values(by=['YearBuilt', 'LotArea'],ascending=[1,0])

#grid = sns.FacetGrid(house_data_train, row='BedroomAbvGr', size=2.2, aspect=1.6)

#grid.map(plt.scatter, 'YearBuilt','SalePrice')#,palette='deep')

#grid.add_legend()





##########################################################

################Correlation Heatmap for numerical values###################

print((house_data_train.isnull().sum()/(house_data_train.shape[0])*100).sort_values(axis=0,ascending=False))

#house_data_train.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)

#house_data_train.corr().style.format("{:.2}").background_gradient(cmap='coolwarm',axis=1)



#house_data_train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()#.style.format("{:.2}").background_gradient(cmap='coolwarm',axis=1)

f, ax = plt.subplots(figsize=(20, 20))

corr = house_data_train.corr()

sns.heatmap(corr, square=True,fmt='.2f')

fig=plt.figure()

print(corr['SalePrice'].sort_values(ascending=False))

################################################################

print((house_data_train.isnull().sum()/(house_data_train.shape[0])*100).sort_values(axis=0,ascending=False))

#print((house_data_test.isnull().sum()/(house_data_test.shape[0])*100).sort_values(axis=0,ascending=False))

house_data_train.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)

house_data_test.drop(['PoolQC','MiscFeature','Alley','Fence'], axis=1, inplace=True)

#print(house_data_train.columns.values)

###FirePlaceQu###

##The description notebook states FirePlaceQu=NA  for No Fireplace so we can eqaute it to 0

fillmissingval(house_data_train,'FireplaceQu',0)

fillmissingval(house_data_test,'FireplaceQu',0)

#For the LotFrontage we can replace NA with the mean value 

###house_data_train['LotFrontage'].fillna(house_data_train['LotFrontage'].mean,inplace=True)

fillmissingval(house_data_train,'LotFrontage',house_data_train['LotFrontage'].mean)

fillmissingval(house_data_test,'LotFrontage',house_data_test['LotFrontage'].mean)

#GarageFinish,GarageType,GarageCond,GarageQual,GarageYrBlt have same number of missing values 

#which can be due to absence of garage. This NA can be changed  t0 0

fillmissingval(house_data_train,'GarageFinish',0)

fillmissingval(house_data_train,'GarageType',0)

fillmissingval(house_data_train,'GarageCond',0)

fillmissingval(house_data_train,'GarageQual',0)

fillmissingval(house_data_train,'GarageYrBlt',0)

fillmissingval(house_data_test,'GarageFinish',0)

fillmissingval(house_data_test,'GarageType',0)

fillmissingval(house_data_test,'GarageCond',0)

fillmissingval(house_data_test,'GarageQual',0)

fillmissingval(house_data_test,'GarageYrBlt',0)



#MasVnrArea & MasVnrType have same number of missing values which can be due to MasVnrType=None. I will

#update MasVnrArea to 0 for missing values and change the type later

#house_data_train['MasVnrArea'].fillna(0,inplace=True)

fillmissingval(house_data_train,'MasVnrArea',0)

fillmissingval(house_data_test,'MasVnrArea',0)

fillmissingval(house_data_train,'MasVnrType','None')

fillmissingval(house_data_test,'MasVnrType','None')

#BsmtExposure and BsmtFinType2 have same values whereas BsmtFinType1,BsmtCond&BsmtQual have 

#same no of missing values so let's dive deeper into Bsmtexposure & BsmtFinType2



Bsmtdf=house_data_train[house_data_train['BsmtFinType2'].isnull()|house_data_train['BsmtExposure'].isnull()|house_data_train['BsmtFinType1'].isnull()|house_data_train['BsmtCond'].isnull()|house_data_train['BsmtQual'].isnull()]

#print(Bsmtdf[['TotalBsmtSF','BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond','BsmtQual']])





house_data_train.set_value(index=332, col='BsmtFinType2', value=house_data_train['BsmtFinType2'].mode()[0])

house_data_train.set_value(index=332, col='BsmtExposure', value=house_data_train['BsmtExposure'].mode()[0])



house_data_train.set_value(index=948, col='BsmtFinType2', value=house_data_train['BsmtFinType2'].mode()[0])

house_data_train.set_value(index=948, col='BsmtExposure', value=house_data_train['BsmtExposure'].mode()[0])



#Changing the Null values to 0

fillmissingval(house_data_train,'BsmtFinType2',0)

fillmissingval(house_data_train,'BsmtExposure','No')

fillmissingval(house_data_train,'BsmtFinType1',0)

fillmissingval(house_data_train,'BsmtCond',0)

fillmissingval(house_data_train,'BsmtQual','None')

fillmissingval(house_data_test,'BsmtFinType2',0)

fillmissingval(house_data_test,'BsmtExposure','No')

fillmissingval(house_data_test,'BsmtFinType1',0)

fillmissingval(house_data_test,'BsmtCond',0)

fillmissingval(house_data_test,'BsmtQual','None')



####Changing the Electrical null value to the most recurring value

fillmissingval(house_data_train,'Electrical',house_data_train['Electrical'].mode()[0])

fillmissingval(house_data_test,'Electrical',house_data_train['Electrical'].mode()[0])

#################################################################################

######Filling in null values for test data for prediction########################

#################################################################################



fillmissingval(house_data_test,'MSZoning',house_data_test['MSZoning'].mode()[0])

fillmissingval(house_data_test,'Functional',house_data_test['Functional'].mode()[0])

#print((house_data_test.isnull().sum().sort_values(axis=0,ascending=False)))



a=house_data_test[house_data_test['BsmtHalfBath'].isnull()|house_data_test['BsmtFullBath'].isnull()|\

      house_data_test['TotalBsmtSF'].isnull() |house_data_test['BsmtFinSF2'].isnull()|\

      house_data_test['BsmtUnfSF'].isnull()|house_data_test['BsmtFinSF1'].isnull()]

#print(a[['BsmtQual','MSZoning','Functional','Utilities','BsmtFullBath','BsmtFinSF1','GarageCars',\

#                      'BsmtUnfSF','TotalBsmtSF','SaleType','Exterior1st','Exterior2nd','KitchenQual',\

#                      'GarageArea']])

######the values are 0 due to no basement########################

fillmissingval(house_data_test,'BsmtHalfBath',0)

fillmissingval(house_data_test,'BsmtFullBath',0)

fillmissingval(house_data_test,'TotalBsmtSF',0)

fillmissingval(house_data_test,'BsmtFinSF2',0)

fillmissingval(house_data_test,'BsmtUnfSF',0)

fillmissingval(house_data_test,'BsmtFinSF1',0)





######################garage cars & area might be null because of no garage##################

a=house_data_test[house_data_test['GarageCars'].isnull()|house_data_test['GarageArea'].isnull()]

print(a[['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea']])

fillmissingval(house_data_test,'GarageCars',0) # since we dont have sufficient data updating it to 0

fillmissingval(house_data_test,'GarageArea',0)





a=house_data_test[house_data_test['Utilities'].isnull()|house_data_test['SaleType'].isnull()|\

                 house_data_test['KitchenQual'].isnull()|house_data_test['Exterior1st'].isnull()|\

                 house_data_test['Exterior2nd'].isnull()]

###################uodating rest of the columns with most frequent value########################

fillmissingval(house_data_test,'Utilities',house_data_test['Utilities'].mode()[0])

fillmissingval(house_data_test,'SaleType',house_data_test['SaleType'].mode()[0])

fillmissingval(house_data_test,'KitchenQual',house_data_test['KitchenQual'].mode()[0])

fillmissingval(house_data_test,'Exterior1st',house_data_test['Exterior1st'].mode()[0])

fillmissingval(house_data_test,'Exterior2nd',house_data_test['Exterior2nd'].mode()[0])

print((house_data_test.isnull().sum().sort_values(axis=0,ascending=False)))

print((house_data_train.isnull().sum().sort_values(axis=0,ascending=False)))

##########################################################

### Relationship between categorical data & SalesPrice######

##########################################################

g = sns.PairGrid(house_data_train,\

                 y_vars=['ExterQual','Neighborhood','HeatingQC','RoofMatl','Condition2',\

                         'PavedDrive','BsmtQual','RoofStyle','LotShape','KitchenQual',\

                         'HouseStyle','LotConfig','Exterior1st','Functional','BldgType',\

                         'Utilities','ExterCond','Exterior2nd','CentralAir','Street',\

                         'SaleType','SaleCondition','MSZoning','LandSlope','LandContour',\

                         'MasVnrType','Condition1','Foundation','Electrical','Heating'],\

                 x_vars='SalePrice',aspect=.75, size=10)

g.map(sns.violinplot)

fig=plt.figure()



### So after looking at the violenplots it seems that the following variables 

### might have an effect on SalePrice: ExterQual,KitchenQual,heatingQC,BsmntQual

##########################################################

### Convert four categorical variables to numerical values by using map function and not pd.getdummies######

##########################################################

house_data_train['BsmtQual'] = house_data_train['BsmtQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

house_data_train['HeatingQC'] = house_data_train['HeatingQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

house_data_train['KitchenQual'] = house_data_train['KitchenQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

house_data_train['ExterQual'] =house_data_train['ExterQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



house_data_test['BsmtQual'] = house_data_test['BsmtQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

house_data_test['HeatingQC'] = house_data_test['HeatingQC'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

house_data_test['KitchenQual'] = house_data_test['KitchenQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})

house_data_test['ExterQual'] =house_data_test['ExterQual'].map({'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



##########################################################

################Correlation Heatmap for all data###################



#f, ax = plt.subplots(figsize=(20, 20))

corr = house_data_train.corr()

sns.heatmap(corr, square=False,fmt='.5f')

fig=plt.figure()

################Keeping columns that are strongly correlated to SalePrice###################

#print(corr['SalePrice'].sort_values(ascending=False))



#############Distribution of salePrice######################################

#f, (ax1,ax2) = plt.subplots(2, sharey=True)

sns.distplot(house_data_train['SalePrice'])

fig=plt.figure()

#SalePrice is not normally distributed and is rightly skewed. We can apply square root or log transformation.

#choosing log since squareroot is weaker type of transformation

house_data_train['SalePrice']=np.log(house_data_train['SalePrice'])

#rechecking saleprice

sns.distplot(house_data_train['SalePrice'])

fig=plt.figure()

#sns.distplot(house_data_train['SalePrice'],ax=ax2)

##########################################################################

col_to_keep=(corr.columns.values)

house_data_train=house_data_train[col_to_keep]



#preparing X & Y of training data

Y_train=house_data_train['SalePrice']



#dropping saleprice column

house_data_train.drop(['SalePrice'], axis=1, inplace=True)

col_to_keep=(house_data_train.columns.values)

house_data_test=house_data_test[col_to_keep]

house_data_train.drop(['BsmtFinSF2','BsmtHalfBath','Id'], axis=1, inplace=True)

X_train=house_data_train

house_data_test.drop(['BsmtFinSF2','BsmtHalfBath','Id'], axis=1, inplace=True)



#print(house_data_train.shape)

#print(house_data_test.shape)

regr = linear_model.LinearRegression()

regr.fit(X_train, Y_train)

print(regr.score(X_train,Y_train))

predict_train=regr.predict(X_train)

predict_test=regr.predict(house_data_test)







#mean squared error of training data

print("Mean squared error:%.2f" % mean_squared_error(Y_train, predict_train))





# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(Y_train, predict_train))






