#Core packages

import pandas as pd

import numpy as np



#Data Profiling

import pandas_profiling as pp



#Visualisation

import matplotlib.pyplot as plt

import seaborn as sns



#sklearn

import category_encoders as ce

from sklearn.preprocessing import PowerTransformer, RobustScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression, ElasticNet,ElasticNetCV



#XGBoost

from xgboost import XGBRegressor
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



print(test.shape)

print(train.shape)
featuresDF = train.drop(['Id','SalePrice'],axis=1)   

featuresDF.profile_report(style={'full_width':True})
# Filling 'None' for missing values

for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure',

            'BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType',

            'GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'):

    train.loc[:,col] = train.loc[:,col].fillna('None')

    test.loc[:,col] = test.loc[:,col].fillna('None')



# Filling the mode value for the features actual missing values

for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):

    train.loc[:,col] = train.loc[:,col].fillna(train.loc[:,col].mode()[0])

    test.loc[:,col] = test.loc[:,col].fillna(train.loc[:,col].mode()[0])
# Filling '0' for missing values

for col in ('2ndFlrSF','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

            'BsmtHalfBath','GarageCars','TotalBsmtSF', 'BsmtFullBath'):

    train.loc[:,col] = train.loc[:,col].fillna(0)

    test.loc[:,col] = test.loc[:,col].fillna(0)



# Filling the mean value for the missing values

train.loc[:,'LotFrontage'] = train.loc[:,'LotFrontage'].fillna(train.loc[:,'LotFrontage'].mean())

test.loc[:,'LotFrontage'] = test.loc[:,'LotFrontage'].fillna(train.loc[:,'LotFrontage'].mean())
train = train.loc[train.loc[:,'GrLivArea']<4000,:]

idx_train = train.shape[0] #keep track of the training inde

idx_train
corrMatrix = train.drop('Id',axis=1).corr()

plt.figure(figsize=[30,15])

heatmap = sns.heatmap(corrMatrix, annot=True, cmap='Blues')
df = pd.concat([train,test], sort=False)

df.drop(['TotRmsAbvGrd','GarageArea', 'GarageYrBlt', '1stFlrSF'], axis=1, inplace=True)

df.shape
# transforming MSSubClass to category type

df.loc[:,'MSSubClass'] = df.loc[:,'MSSubClass'].astype('category')

print(df.loc[:,'MSSubClass'].dtype)



#splitting the dataset in test id, the training set and the testing set

test_id = df.iloc[idx_train:,0]

df.drop(['Id'], axis=1, inplace=True)



train = df.iloc[:idx_train,:]

test = df.iloc[idx_train:,:]



print(test.shape)

print(train.shape)
#saving predictor columns and label column

predictors = list(df.columns.drop(['SalePrice']))

label = 'SalePrice'



#splitting the dataset

X_train = df.iloc[:idx_train, :].loc[:, predictors]

y_train = df.iloc[:idx_train, :].loc[:, label]

X_test = df.iloc[idx_train:, :].loc[:, predictors]



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
#obtaining numerical features with absolute skewness > .5

skewed_cols_bool = np.absolute(df.select_dtypes(include=['int','float']).skew(axis = 0)) > .5

skewed_cols = skewed_cols_bool.loc[skewed_cols_bool].index.drop('SalePrice').tolist()



#applying a power transformer to skweded cols

pt = PowerTransformer()

X_train.loc[:,skewed_cols] = pt.fit_transform(X_train.loc[:,skewed_cols])

X_test.loc[:,skewed_cols] = pt.transform(X_test.loc[:,skewed_cols])



# print(X_test.shape)

# print(X_train.shape)

# print(X_train.head())

# print(skewed_cols)
#one-hot encoding

df = pd.concat([X_train,X_test], sort=False)

df = pd.get_dummies(df)



X_train = df.iloc[:idx_train, :]

X_test = df.iloc[idx_train:, :]



print(df.shape)

print(X_train.shape)

print(X_test.shape)
scaler=RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



print(X_train.shape)

print(X_test.shape)
#log transforming the label

temp = np.log1p(y_train) # creating a temp series to avoid chained indexing

y_train = temp
elasticNet = ElasticNetCV(l1_ratio = [.1, .3, .5, .7, .8, .85, .9, .95, .99, 1], cv=10, n_jobs=-1)

elasticNet.fit(X_train, y_train)

predictions = elasticNet.predict(X_test)

predictionsExp = np.exp(predictions)



predictionsExp.shape
submission = pd.DataFrame({

    "Id": test_id,

    "SalePrice": predictionsExp

})

submission.to_csv('submission_ElasticNetCVFinal.csv', index=False)



# Prepare CSV

print(submission.head())