import warnings, os, math

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline



import numpy as np 

import pandas as pd 

import seaborn as sns

from scipy.stats import norm
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head(5)
recordId='Id'

target='SalePrice'

trainId=train[recordId]

testId=test[recordId]



# Dropping "Id" column from train and test set

train.drop(recordId,axis=1,inplace=True)

test.drop(recordId,axis=1,inplace=True)



# Checking Dataset shape

print('Train Set\t %d X %d'%(train.shape[0],train.shape[1]))

print('Test Set\t %d X %d'%(test.shape[0],test.shape[1]))
numericalFeatures=train.select_dtypes(include=[np.number]).columns

nrows=6

ncols=int(len(numericalFeatures)/nrows)

fig,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(20,20),sharey=True)

fig.subplots_adjust(wspace=0.2,hspace=0.4)

for row in range(nrows):

    for col in range(ncols):

        sns.scatterplot(train[numericalFeatures[row*ncols+col]],train[target],ax=ax[row,col])
# Scatter plot of 'LotFrontage' VS 'SalePrice' has two outliers to the right. Therefore we can delete them.

train.drop(index=train['LotFrontage'].sort_values(ascending=False)[:2].index,inplace=True)

print('Train Set\t %d X %d'%(train.shape[0],train.shape[1]))
nTrain=train.shape[0]

nTest=test.shape[0]

trainY=train[target]

allData=pd.concat((train,test)).reset_index(drop=True)

allData.drop(target,axis=1,inplace=True)

print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))
count=allData.isnull().sum().sort_values(ascending=False)

percentage=(allData.isnull().sum()/allData.isnull().count()).sort_values(ascending=False)*100

dtypes=allData[count.index].dtypes

missingData=pd.DataFrame({'Count':count,'Percentage':percentage,'Type':dtypes})

missingData.drop(missingData[missingData['Count']==0].index,inplace=True)



# Plotting

fig,ax=plt.subplots(figsize=(10,4))

plt.xticks(rotation='90')

sns.barplot(x=missingData.index,y=missingData['Percentage'])

ax.set_xlabel('Features')

ax.set_title('Percentage of missing values');

missingData.head(10)
indices=allData[(allData['PoolQC'].isnull())&(allData['PoolArea']>0)].index

if len(indices)>0:

    print(allData.loc[indices,['PoolQC','PoolArea','OverallQual']])

mapper={0:'None',1:'Po',2:'Fa',3:'TA',4:'Gd',5:'Ex'}

for index in indices:

    allData.loc[index,'PoolQC']=mapper[math.ceil(allData.loc[index,'OverallQual']/2)]

allData['PoolQC'].fillna('None',inplace=True)    
# Check if all 159 NAs are the same observations among all 4 garage variables.

cols=['GarageFinish','GarageQual','GarageCond','GarageYrBlt']

len(allData[allData['GarageFinish'].isnull() & allData['GarageQual'].isnull() & allData['GarageCond'].isnull() & allData['GarageYrBlt'].isnull()])==159

# Check if all 157 NAs are same observations among 159 observations in GarageQual

len(allData[allData['GarageType'].isnull() & allData['GarageQual'].isnull()].index)==157

# Getting observations where GarageType is not null but GarageQual is.

indices=allData[(allData['GarageQual'].isnull())&allData['GarageType'].notnull()].index

allData.loc[indices,['GarageType','GarageArea','GarageCars']+cols]
for index in indices:

    # If GarageArea is not null then replace other garage variables with mode.

    if pd.isnull(allData.loc[index,'GarageArea'])==False:

        for feature in cols:

            allData.loc[index,feature]=allData[feature].mode()[0]

    else:

        allData.loc[index,'GarageType']='None'

# Imputing rest of the 158 observations with None or 0 values

for feature in cols+['GarageCars','GarageArea']:

    allData[feature].fillna(0,inplace=True)

allData['GarageType'].fillna('None',inplace=True)
# check if all 79 NAs are the same observations among the variables with 80+ NAs

cols=['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

len(allData[allData['BsmtFinType1'].isnull() & allData['BsmtFinType2'].isnull() & allData['BsmtQual'].isnull() & allData['BsmtExposure'].isnull()& allData['BsmtCond'].isnull()])==79

indices=allData[allData['BsmtFinType1'].notnull() & (allData['BsmtCond'].isnull() | allData['BsmtQual'].isnull() | allData['BsmtExposure'].isnull() | allData['BsmtFinType2'].isnull())].index

allData.loc[indices,cols]
for feature in cols:

    # Imputing mode in basement variables to fix these 9 houses

    allData.loc[indices,feature]=allData.loc[indices,feature].fillna(allData[feature].mode()[0])

    # Imputing 'None' for the rest of the 79 observations

    allData[feature].fillna('None',inplace=True)

for feature in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):

    allData[feature].fillna(0,inplace=True)
# Check for houses with Masonry but 'NA' values in MasVnrType variable.

indices=allData[(allData['MasVnrType'].isnull())&(allData['MasVnrArea']>0)].index

if len(indices)>0:

    print(allData.loc[indices,['MasVnrType','MasVnrArea']])

# Fixing MasVnrType by imputing mode

allData.loc[indices,'MasVnrType']=allData.loc[indices,'MasVnrType'].fillna(allData['MasVnrType'].value_counts().index[1])

# Imputing rest of the observations with 'None' and 0 values

allData['MasVnrType'].fillna('None',inplace=True)

allData['MasVnrArea'].fillna(0,inplace=True)
# Imputing missing values 'None' or Mode values

allData['MiscFeature'].fillna('None',inplace=True)

allData['Alley'].fillna('None',inplace=True)

allData['Fence'].fillna('None',inplace=True)

allData['FireplaceQu'].fillna('None',inplace=True)

# Imputing missing values with the median of houses in that Neighborhood

allData['LotFrontage']=allData.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))

allData['MSZoning'].fillna(allData['MSZoning'].mode()[0],inplace=True)

allData['Functional'].fillna('Typ',inplace=True)

allData.drop('Utilities',inplace=True,axis=1)

allData['Exterior1st'].fillna(allData['Exterior1st'].mode()[0],inplace=True)

allData['Exterior2nd'].fillna(allData['Exterior2nd'].mode()[0],inplace=True)

allData['KitchenQual'].fillna(allData['KitchenQual'].mode()[0],inplace=True)

allData['Electrical'].fillna(allData['Electrical'].mode()[0],inplace=True)

allData['SaleType'].fillna(allData['SaleType'].mode()[0],inplace=True)

print('Total Missing Count\t%d'%(allData.isnull().sum().max()))
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(9,3))

fig.subplots_adjust(wspace=0.2,hspace=0.4)



sns.distplot(trainY,ax=ax[0],fit=norm)

ax[0].set_ylabel('Frequency')

ax[0].set_xlabel('Skew : %.3f'%(trainY.skew()))

ax[0].set_title('Original Distribution (Skewed)')



# Log Transformation

trainY=np.log(trainY)



sns.distplot(trainY,ax=ax[1],fit=norm)

ax[1].set_ylabel('Frequency')

ax[1].set_xlabel('Skew : %.3f'%(trainY.skew()))

ax[1].set_title('Normal Distribution')

print('Log Transformation : ')

print("(The target variable 'SalePrice' is right skewed. Log tansfromation is done to normalize the distribution.)")
fig,ax=plt.subplots(figsize=(13,10))

corrMat=train.corr()

sns.heatmap(corrMat)
# Reducing Mutlicollinearity

allData.drop(columns=['GarageYrBlt','1stFlrSF','TotRmsAbvGrd','GarageArea'],inplace=True)
cols=['YrSold','MoSold']

for feature in cols:

    allData[feature]=allData[feature].astype('str')
# Converting all the ordinal variables in the dataset

mapper={

    'Alley':{'None':0,'Grvl':1,'Pave':2},

    'BsmtCond':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'BsmtExposure':{'None':0,'No':1,'Mn':2,'Av':3,'Gd':4},

    'BsmtFinType1':{'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},

    'BsmtFinType2':{'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6},

    'BsmtQual':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'ExterCond':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'ExterQual':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'FireplaceQu':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'Functional':{'Sal':0,'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7},

    'GarageCond':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'GarageFinish':{'None':0,'Unf':1,'RFn':2,'Fin':3},

    'GarageQual':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'HeatingQC':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'KitchenQual':{'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},

    'LandSlope':{'None':0,'Sev':1,'Mod':2,'Gtl':3},

    'LotShape':{'None':0,'IR3':1,'IR2':2,'IR1':3,'Reg':4},

    'PavedDrive':{'None':0,'N':1,'P':2,'Y':3},

    'PoolQC':{'None':0,'Fa':1,'TA':2,'Gd':3,'Ex':4},

    'Street':{'None':0,'Grvl':1,'Pave':2},

    'Utilities':{'None':0,'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub': 4}

}

allData=allData.replace(mapper)

print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))

allData.sample(5)
skewness=allData[allData.select_dtypes(include=[np.number]).columns].skew().sort_values(ascending=False)

print('Skewness : ')

print(skewness.head(10))

skewness=skewness[abs(skewness)>0.5]

print('\nPerforming Log transformation on %d features...'%(skewness.shape[0]))

for feature in skewness.index:

    allData[feature]=np.log1p(allData[feature])

numericalFeatures=allData.select_dtypes(include=[np.number]).columns

categoricalFeatures=allData.select_dtypes(include=[np.object]).columns
allData=pd.get_dummies(allData)

print('Train + Test Set\t %d X %d'%(allData.shape[0],allData.shape[1]))

allData.sample(5)
trainX=allData[:nTrain]

testX=allData[nTrain:]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import StandardScaler



# Splitting training set further into training and validation set

subTrainX,valX,subTrainY,valY=train_test_split(trainX,trainY,test_size=0.3,random_state=42)
# Training

lr=LinearRegression()

lr.fit(subTrainX,subTrainY)

log_prediction=lr.predict(subTrainX)

error=np.sqrt(mean_squared_error(subTrainY,log_prediction))

score=r2_score(subTrainY,log_prediction)

print('Training Accuracy : %.2f%%\t Error : %.4f'%(score*100,error))
# Cross Validation

log_prediction=lr.predict(valX)

error=np.sqrt(mean_squared_error(valY,log_prediction))

score=r2_score(valY,log_prediction)

print('Validation Accuracy : %.2f%%\t Error : %.4f'%(score*100,error))
# Training

alphas=np.arange(0.01,10, 0.1)

errors=[]

scores=[]

rlr={}

log_prediction={}



for alpha in alphas:

    rlr[alpha]=Ridge(alpha)

    rlr[alpha].fit(subTrainX,subTrainY)

    log_prediction[alpha]=rlr[alpha].predict(subTrainX)

    error=np.sqrt(mean_squared_error(subTrainY,log_prediction[alpha]))

    score=r2_score(subTrainY,log_prediction[alpha])

    errors.append(error)

    scores.append(score)

alpha=alphas[errors.index(min(errors))]

error=np.sqrt(mean_squared_error(subTrainY,log_prediction[alpha]))

score=r2_score(subTrainY,log_prediction[alpha])

print('Alpha Chosen : %.4f'%alpha)

print('Training Accuracy : %.2f%%\t Error : %.4f'%(score*100,error))

sns.scatterplot(alphas,errors)
# Cross Validation

log_prediction=rlr[alpha].predict(valX)

error=np.sqrt(mean_squared_error(valY,log_prediction))

score=r2_score(valY,log_prediction)

print('Validation Accuracy : %.2f%%\t Error : %.4f'%(score*100,error))
# Training over the entire trainset with optimal alpha

rlr=Ridge(alpha)

rlr.fit(trainX,trainY)

log_prediction=rlr.predict(trainX)

error=np.sqrt(mean_squared_error(trainY,log_prediction))

score=r2_score(trainY,log_prediction)

print('Training Accuracy : %.2f%%\t Error : %.4f'%(score*100,error))
prediction=np.expm1(rlr.predict(testX))

submission=pd.DataFrame()

submission[recordId]=testId

submission[target]=prediction

submission.head()
submission.to_csv('submission.csv',index=False)