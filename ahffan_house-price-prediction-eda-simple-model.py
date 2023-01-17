#import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk

%matplotlib inline

import os as os
os.chdir("/kaggle/input/house-prices-advanced-regression-techniques")

print(os.listdir("../house-prices-advanced-regression-techniques/"))
#load the data into a Pandas dataframe

train_features_df=pd.read_csv("train.csv")

test_features_df=pd.read_csv("test.csv") 
train_features_df.dtypes
#Preview of dataframes

train_features_df.head(7)
test_features_df.head(7)
train_features_df.shape
test_features_df.shape
#Use info to see length and datatypes

train_features_df.info()
test_features_df.info()
train_df=train_features_df
#look for duplicate data

train_df.duplicated().sum()
test_features_df.duplicated().sum()
#Check for entries with SalePrice<=0

(train_df.SalePrice<=0).sum() 


# Copy of the input Data frame



train_data=train_df

traindata_df=train_df.copy()

test_data=test_features_df 
len(traindata_df) 
#Identify numerical and categorical varibales

train_data.columns

#len(train_data.columns)
test_features_df.columns
# Split into numerical and Categorical features



numeric_cols = [ c for c in train_data.columns if train_data.dtypes[c] != 'object' ]

categorical_cols = [ c for c in train_data.columns if train_data.dtypes[c] == 'object' ]
categorical_cols
#Summarize numericaland categorical variables separately

train_data.describe(include=[np.number])


train_data.describe(include=[np.object])
train_data.describe(include=['O'])
#Visualize target variable distribution and boxplot

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

sns.boxplot(train_data.SalePrice)

plt.subplot(1,2,2)

sns.distplot(train_data.SalePrice,bins=20)

plt.show()
print('The skew is '+ str(train_data.SalePrice.skew()))

print('The Kurtosis is '+ str(train_data.SalePrice.kurt()))


stat = train_data.SalePrice.describe()

stat
#Use 1.5 IQR rule to find outliers

stat = train_data.SalePrice.describe()

print(stat)

IQR = stat['75%'] - stat['25%']

upper = stat['75%'] + 1.5*IQR

lower = stat['25%'] - 1.5*IQR

print('The upper and lower bounds of suspected outliers are {} and {}.'.format(upper,lower))
print(" The upper limit is " + str(upper))

print(" The lower limit is " + str(lower))
#Check potential outlier below lower bound

print(train_data[train_data.SalePrice < 3937.5])

print(train_data[train_data.SalePrice > 340037.5 ])
 

#train_data.head()
#Define a function to plot the relation between features and the target

def plot_feature(df,col):

    plt.figure(figsize=[25,5])

    plt.subplot(1,3,2)

    if df[col].dtype=='int64':

        #mean=df.groupby([col])['SalePrice'].mean().plot()

        print(col)

        #df.plot.scatter(x=col, y='SalePrice')#, ylim=(0,800000))

        plt.scatter(x=col, y='SalePrice',data=train_data)

        plt.xlabel(col)

        plt.ylabel("Mean SalePrice")

        plt.subplot(1,3,1)

        #df[col].value_counts().sort_index().plot()

        sns.distplot(df[col],bins=20)

        plt.xlabel(col)

        plt.ylabel("Counts")

    else:

        mean=df.groupby(by=[col])['SalePrice'].mean().sort_values(ascending=True).index

        sns.boxplot(x=col,y="SalePrice",data=df,order=mean)

        plt.xticks(rotation=20)

        plt.subplot(1,3,1)

        df[col].value_counts().sort_index().plot()

        plt.xlabel(col)

        plt.ylabel("Counts")

    
train_data['TotalBsmtSF'].hist()


plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

train_data['TotalBsmtSF'].value_counts().sort_index().plot()

 



plt.subplot(1,2,2)

plt.scatter(x='TotalBsmtSF', y='SalePrice',data=train_data)# ylim=(0,800000))

numeric_cols
#numeric_cols.remove( ("Id","MSSubClass","LotArea","OverallQual","OverallCond","YearBuilt","YearRemodAdd","BsmtFinSF1","BsmtFinSF2"))

# for col in numeric_cols:

#     plot_feature(train_data,col)
#plot_feature(train_data,"CentralAir")

for col in categorical_cols:

    plot_feature(train_data,col)

    


plt.figure(figsize=[35,15])

sns.boxplot(x="Neighborhood",y="SalePrice",data=train_data)
numeric_cols
#Find relations between features

sns.set()

cols = ["GrLivArea","SalePrice","WoodDeckSF","OverallQual"]

sns.pairplot(train_data[cols],size = 2.5)
plt.scatter(x='GrLivArea',y="SalePrice",data=train_data)
# len(train_data)

# np.log1p(train_data['SalePrice'])
#plt.scatter(y=np.log1p(train_data['SalePrice']),x="GrLivArea",data=train_data)
train_data=train_data.drop(train_data[ (train_data['SalePrice'] < 300000) & (train_data['GrLivArea'] > 4000) ].index )
#combine data for handling missing values



nrows_train = train_data.shape[0]

nrows_test = test_data.shape[0]

all_data=pd.concat([train_data,test_data]).reset_index(drop=True)



all_data.drop(['SalePrice'], axis=1, inplace=True)



print("all_data size is : {}".format(all_data.shape)) 
# needed in future

nrows_train
#find missing values



Total=all_data.isnull().count()

counts=all_data.isnull().sum()



summary = pd.concat([counts,counts/Total],axis =1,keys=["Count","Percentage"])

summary.sort_values(by="Percentage",ascending=False)
all_data['MiscFeature']=all_data['MiscFeature'].fillna("None")

all_data['Alley']=all_data['Alley'].fillna("None")

all_data['Fence']=all_data['Fence'].fillna("None")

all_data['FireplaceQu']=all_data['FireplaceQu'].fillna("None")

all_data['GarageFinish']=all_data['GarageFinish'].fillna("None")

all_data['GarageQual']=all_data['GarageFinish'].fillna("None")

all_data['GarageType']=all_data['GarageType'].fillna("None")

all_data['BsmtCond']=all_data['BsmtCond'].fillna("None")

all_data['BsmtExposure']=all_data['BsmtExposure'].fillna("Nobase")

all_data['BsmtQual']=all_data['BsmtQual'].fillna("None")

all_data['BsmtFinType2']=all_data['BsmtFinType2'].fillna("None")

all_data['BsmtFinType1']=all_data['BsmtFinType1'].fillna("None")

all_data['GarageCond']=all_data['GarageCond'].fillna("None")

all_data['PoolQC']=all_data['PoolQC'].fillna("None")

all_data['MasVnrType']=all_data['MasVnrType'].fillna("None")

all_data['MasVnrArea']=all_data['MasVnrArea'].fillna(0)

all_data['Functional']=all_data['Functional'].fillna("Typ")

#all_data['Utilities']=all_data['Utilities'].fillna("AllPub")

all_data=all_data.drop(['Utilities'],axis=1)

all_data['MSZoning']=all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical']=all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ['GarageYrBlt','GarageArea','GarageCars']:

    all_data[col]=all_data[col].fillna(0)

    

for col in ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']:

    all_data[col]=all_data[col].fillna(0)

    

for col in ['SaleType','Exterior1st','Exterior2nd','KitchenQual']:

    all_data[col]=all_data[col].fillna(all_data[col].mode()[0])
# Before treaing missing values

sns.boxplot(x="MiscFeature",y="SalePrice",data=train_data)

# After treaing missing values

sns.boxplot(x="MiscFeature",y=train_data['SalePrice'],data=all_data[:1458])
#Data frame length after dropping 2 outliers

len(all_data.dropna())
#check if any more missing values are there



Total=all_data.isnull().count()

counts=all_data.isnull().sum()



summary = pd.concat([counts,counts/Total],axis =1,keys=["Count","Percentage"])

summary.sort_values(by="Percentage",ascending=False)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LassoCV,LinearRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error 

 
linreg = LinearRegression()
all_data.head(7)
all_data.shape
#new_data created for converting categorical to numerical values

new_data=pd.get_dummies(all_data)
new_data.head(7)
#train input

train_model_input = new_data[:1458]



#train output

train_model_output=train_data['SalePrice']



#log train output

train_model_log_output=np.log1p(train_data['SalePrice']).values



#test data

test_model_data = new_data[1458:]
#defin

n_folds = 5

def rmsle_cv(model,data_input,data_output):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(data_input.values)

    rmse= np.sqrt(-cross_val_score(model, data_input.values, data_output, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
linreg.fit(train_model_input, train_model_output) 

rmsle_cv(linreg,train_model_input,train_model_output).mean()
train_model_prediction= linreg.predict(new_data[:1458])
print('Mean Squared Error:', mean_squared_error(train_model_output, train_model_prediction)) 

print('Mean Squared Error in log terms :', mean_squared_error(np.log(train_model_output) , np.log(train_model_prediction) )) 
linreg_log = LinearRegression()

train_model_log_output=np.log(train_data['SalePrice']) 

linreg_log.fit(train_model_input, train_model_log_output)

train_model_log_prediction= linreg_log.predict(new_data[:1458])

print('Mean Squared Error:', mean_squared_error(np.exp(train_model_log_output) , np.exp(train_model_log_prediction) )) 

print('Mean Squared Error in log terms :', mean_squared_error(train_model_log_output , train_model_log_prediction )) 

train_model_log_prediction
n_folds = 5

#rmsle_cv(linreg).mean()

rmsle_cv(linreg_log,train_model_input,train_model_log_output).mean()
#X_train, X_test, y_train, y_test = train_test_split(train_model_input, train_model_log_output, test_size=0.33, random_state=42)
Final_prediction=linreg_log.predict(test_model_data)
Final_prediction
test_data
results=pd.concat([test_data['Id'],pd.Series(np.expm1(Final_prediction))],axis=1,keys=['Id','SalePrice'])
results.head(5)
!pwd
print(os.listdir("/kaggle/working"))

results.to_csv('/kaggle/working/submissions_linreg_3.csv',index=False)