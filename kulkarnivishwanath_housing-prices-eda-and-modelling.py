# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 500)



from sklearn.preprocessing import LabelEncoder,StandardScaler,RobustScaler

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV,train_test_split,KFold

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

import statsmodels.regression.linear_model as sm

import xgboost as xgb

from sklearn.pipeline import make_pipeline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index,inplace=True)
train.dtypes.value_counts()
test.dtypes.value_counts()
train.head()
test.head()
train.shape,test.shape
train.drop('Id',axis=1,inplace=True)

Submission = test[['Id']]

test.drop('Id',axis=1,inplace=True)
train_missing = pd.DataFrame(train.isna().sum()[train.isna().sum() !=0].sort_values())

train_missing.columns = ['#Missing']

train_missing['Percent_Missing'] = train.isna().sum()[train.isna().sum() !=0]/len(train)

train_missing
test_missing = pd.DataFrame(test.isna().sum()[test.isna().sum() !=0].sort_values())

test_missing.columns = ['#Missing']

test_missing['Percent_Missing'] = test.isna().sum()[test.isna().sum() !=0]/len(test)

test_missing
train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)

test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
for col in train.columns:

    if train[col].dtype == "object":

        print ("Cardinality of %s variable in Train Data:"%col,train[col].nunique())

        print ("Cardinality of %s variable in Test Data:"%col,test[col].nunique())

        print ("\n")
train['SalePrice'].describe()
sns.distplot(train['SalePrice'])

plt.title("Distribution of Sale Price variable")

plt.xlabel("Price")
sns.boxplot(train['SalePrice'],orient='vert')
train.drop(train.index[[691,1182]],inplace=True)

train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'])
train['MSSubClass'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['MSSubClass'].value_counts(dropna=False).sort_values().plot(kind='bar')
plt.figure(figsize=(8,8))

sns.boxplot(x=train['MSSubClass'],y=train['SalePrice'])
train.groupby('MSSubClass')['SalePrice'].mean().sort_values().plot(kind='bar')
train['MSSubClass'] = train['MSSubClass'].astype(str)

test['MSSubClass'] = test['MSSubClass'].astype(str)
train['MSZoning'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(x=train['MSZoning'],y=train['SalePrice'])
train.groupby('MSZoning')['SalePrice'].mean().sort_values().plot(kind='bar')
test['MSZoning'].fillna(test['MSZoning'].value_counts(dropna=False).index[0],inplace=True)
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
train['LotArea'].describe()
sns.scatterplot(x=train['LotArea'],y=train['SalePrice'])
train['Street'].value_counts(dropna=False)
test['Street'].value_counts(dropna=False)
train.drop('Street',axis=1,inplace=True)

test.drop('Street',axis=1,inplace=True)
train['LotShape'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['LotShape'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(x=train['LotShape'],y=train['SalePrice'])
train['LandContour'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['LandContour'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['LandContour'],train['SalePrice'])
train['Utilities'].value_counts(dropna=False)
test['Utilities'].value_counts(dropna=False)
train.drop('Utilities',axis=1,inplace=True)

test.drop('Utilities',axis=1,inplace=True)
train['LotConfig'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(x=train['LotConfig'],y=train['SalePrice'])
train.drop('LandSlope',axis=1,inplace=True)

test.drop('LandSlope',axis=1,inplace=True)
train['Neighborhood'].value_counts(dropna=False).sort_values().plot(kind='bar')
plt.figure(figsize=(12,12))

sns.boxplot(x=train['Neighborhood'],y=train['SalePrice'])
train.groupby('Neighborhood')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['Condition1'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['Condition1'].value_counts(dropna=False).sort_values().plot(kind='bar')
train.groupby('Condition1')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train.drop('Condition2',axis=1,inplace=True)

test.drop('Condition2',axis=1,inplace=True)
train['BldgType'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['BldgType'].value_counts(dropna=False)
sns.boxplot(x=train['BldgType'],y=train['SalePrice'])
train['HouseStyle'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['HouseStyle'].value_counts(dropna=False).sort_values().plot(kind='bar')
train['HouseStyle'].value_counts(dropna=False)
test['HouseStyle'].value_counts(dropna=False)
train['OverallQual'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['OverallQual'],train['SalePrice'])
sns.scatterplot(train['OverallQual'],train['SalePrice'])
train.groupby('OverallQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
#train['OverallQual'] = train['OverallQual'].astype(str)

#test['OverallQual'] = test['OverallQual'].astype(str)
train['OverallCond'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['OverallCond'],train['SalePrice'])
train.groupby('OverallCond')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
plt.figure(figsize=(14,14))

sns.boxplot(train['YearBuilt'],train['SalePrice'])
train['YearBuilt'].value_counts(dropna=False).sort_values(ascending=False).head()
plt.figure(figsize=(14,14))

sns.boxplot(train['YearRemodAdd'],train['SalePrice'])
train['YearRemodAdd'].value_counts(dropna=False).sort_values(ascending=False).head()
train['RoofStyle'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['RoofStyle'],train['SalePrice'])
train['RoofMatl'].value_counts(dropna=False)
test['RoofMatl'].value_counts(dropna=False)
train['Exterior1st'].value_counts(dropna=False)
test['Exterior1st'].value_counts(dropna=False)
test['Exterior1st'].fillna(test['Exterior1st'].value_counts(dropna=False).index[0],inplace=True)
train['Exterior1st'].nunique(),test['Exterior1st'].nunique()
train['Exterior2nd'].value_counts(dropna=False)
test['Exterior2nd'].value_counts(dropna=False)
test['Exterior2nd'].fillna(test['Exterior2nd'].value_counts(dropna=False).index[0],inplace=True)
train['Exterior2nd'].nunique(),test['Exterior2nd'].nunique()
train['MasVnrType'].value_counts(dropna=False).sort_values().plot(kind='bar')
train['MasVnrType'].fillna("None",inplace=True)

test['MasVnrType'].fillna("None",inplace=True)
train['MasVnrArea'].fillna(0,inplace=True)

test['MasVnrArea'].fillna(0,inplace=True)
sns.scatterplot(train['MasVnrArea'],train['SalePrice'])
train['ExterQual'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['ExterQual'],train['SalePrice'])
train.groupby('ExterQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['ExterCond'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['ExterCond'],train['SalePrice'])
train.groupby('ExterCond')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['Foundation'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['Foundation'],train['SalePrice'])
train.groupby('Foundation')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['BsmtQual'].value_counts(dropna=False).sort_values().plot(kind='bar')
train['BsmtQual'].fillna("No_Basement",inplace=True)

test['BsmtQual'].fillna("No_Basement",inplace=True)
train.groupby('BsmtQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['BsmtCond'].value_counts(dropna=False)
train['BsmtCond'].fillna("No_Basement",inplace=True)

test['BsmtCond'].fillna("No_Basement",inplace=True)
train['BsmtExposure'].value_counts(dropna=False).sort_values().plot(kind='bar')
train['BsmtExposure'].fillna("No_Basement",inplace=True)

test['BsmtExposure'].fillna("No_Basement",inplace=True)
train['BsmtFinType1'].value_counts(dropna=False).sort_values().plot(kind='bar')
train['BsmtFinType1'].fillna("No_Basement",inplace=True)

test['BsmtFinType1'].fillna("No_Basement",inplace=True)
test['BsmtFinSF1'].fillna(0,inplace=True)
sns.scatterplot(train['BsmtFinSF1'],train['SalePrice'])
train['BsmtFinType2'].fillna("No_Basement",inplace=True)

test['BsmtFinType2'].fillna("No_Basement",inplace=True)
sns.scatterplot(train['BsmtFinSF2'],train['SalePrice'])
test['BsmtFinSF2'].fillna(0,inplace=True)
test['BsmtUnfSF'].fillna(0,inplace=True)
sns.scatterplot(train['BsmtUnfSF'],train['SalePrice'])
train['TotalBsmtSF'].isna().sum(),test['TotalBsmtSF'].isna().sum()
test['TotalBsmtSF'].fillna(0,inplace=True)
sns.scatterplot(train['TotalBsmtSF'],train['SalePrice'])
train['Heating'].value_counts(dropna=False)
test['Heating'].value_counts(dropna=False)
train.groupby('Heating')['SalePrice'].agg(['count','min','max','mean'])
train['HeatingQC'].isna().sum(),test['HeatingQC'].isna().sum()
train['HeatingQC'].value_counts().sort_values().plot(kind='bar')
sns.boxplot(train['HeatingQC'],train['SalePrice'])
train['CentralAir'].value_counts(dropna=False).sort_values().plot(kind='bar')
train.groupby('CentralAir')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['Electrical'].value_counts(dropna=False)
test['Electrical'].value_counts(dropna=False)
train['Electrical'].fillna(train['Electrical'].value_counts(dropna=False).index[0],inplace=True)
train['1stFlrSF'].describe()
sns.scatterplot(train['1stFlrSF'],train['SalePrice'])
train.drop(train.index[1300],inplace=True)
train['2ndFlrSF'].describe()
len(train[train['2ndFlrSF'] == 0])/len(train)
sns.scatterplot(train['2ndFlrSF'],train['SalePrice'])
sns.scatterplot(train['LowQualFinSF'],train['SalePrice'])
train.drop('LowQualFinSF',axis=1,inplace=True)

test.drop('LowQualFinSF',axis=1,inplace=True)
sns.scatterplot(train['GrLivArea'],train['SalePrice'])
train['GrLivArea'].isna().sum(),test['GrLivArea'].isna().sum()
train['BsmtFullBath'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['BsmtFullBath'],train['SalePrice'])
train.groupby('BsmtFullBath')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
test['BsmtFullBath'].fillna(0,inplace=True)
train['BsmtHalfBath'].value_counts(dropna=False).sort_values().plot(kind='bar')
test['BsmtHalfBath'].fillna(0,inplace=True)
train['FullBath'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['FullBath'],train['SalePrice'])
train['HalfBath'].value_counts(dropna=False)
test['HalfBath'].value_counts(dropna=False)
train['BedroomAbvGr'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['BedroomAbvGr'],train['SalePrice'])
train['KitchenAbvGr'].value_counts(dropna=False).sort_values().plot(kind='bar')
train['KitchenQual'].value_counts(dropna=False).sort_values().plot(kind='bar')
train.groupby('KitchenQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
test['KitchenQual'].fillna(test['KitchenQual'].value_counts(dropna=False).index[0],inplace=True)
train['TotRmsAbvGrd'].value_counts(dropna=False).sort_index()
sns.boxplot(train['TotRmsAbvGrd'],train['SalePrice'])
train['Functional'].value_counts(dropna=False).sort_values()
test['Functional'].fillna(test['Functional'].value_counts(dropna=False).index[0],inplace=True)
sns.boxplot(train['Functional'],train['SalePrice'])
train['Fireplaces'].value_counts(dropna=False).sort_index()
sns.scatterplot(train['Fireplaces'],train['SalePrice'])
train.groupby('Fireplaces')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['GarageType'].isna().sum(),test['GarageType'].isna().sum()
train['GarageType'].value_counts(dropna=False).sort_values().plot(kind='bar')
sns.boxplot(train['GarageType'],train['SalePrice'])
train['GarageType'].fillna("No_Garage",inplace=True)

test['GarageType'].fillna("No_Garage",inplace=True)
train.groupby('GarageType')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['GarageYrBlt'].isna().sum(),test['GarageYrBlt'].isna().sum()
train['GarageYrBlt'].fillna(0,inplace=True)

test['GarageYrBlt'].fillna(0,inplace=True)
train['GarageYrBlt'].value_counts().sort_values(ascending=False).head()
plt.figure(figsize=(14,14))

sns.boxplot(train['GarageYrBlt'],train['SalePrice'])
test.loc[1132,'GarageYrBlt'] = 2007
train['GarageFinish'].isna().sum(),test['GarageFinish'].isna().sum()
train['GarageFinish'].fillna("No_Garage",inplace=True)

test['GarageFinish'].fillna("No_Garage",inplace=True)
train.groupby('GarageFinish')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
sns.boxplot(train['GarageFinish'],train['SalePrice'])
train['GarageCars'].isna().sum(),test['GarageCars'].isna().sum()
test['GarageCars'].fillna(0,inplace=True)
train['GarageCars'].value_counts().sort_values().plot(kind='bar')
sns.boxplot(train['GarageCars'],train['SalePrice'])
train['GarageArea'].isna().sum(),test['GarageArea'].isna().sum()
train['GarageArea'].describe()
sns.scatterplot(train['GarageArea'],train['SalePrice'])
test['GarageArea'].fillna(0,inplace=True)
train['GarageQual'].isna().sum(),test['GarageQual'].isna().sum()
train['GarageQual'].fillna("No_Garage",inplace=True)

test['GarageQual'].fillna("No_Garage",inplace=True)
train['GarageQual'].value_counts().sort_values(ascending=False)
test['GarageQual'].value_counts().sort_values(ascending=False)
train.groupby('GarageQual')['SalePrice'].agg(['count','min','max','mean']).sort_values(by='count')
train['GarageCond'].isna().sum(),test['GarageCond'].isna().sum()
train['GarageCond'].fillna("No_Garage",inplace=True)

test['GarageCond'].fillna("No_Garage",inplace=True)
train['GarageCond'].value_counts().sort_values().plot(kind='bar')
sns.boxplot(train['GarageCond'],train['SalePrice'])
train['PavedDrive'].isna().sum(),test['PavedDrive'].isna().sum()
train['PavedDrive'].value_counts()
test['PavedDrive'].value_counts()
train['WoodDeckSF'].isna().sum(),test['WoodDeckSF'].isna().sum()
sns.scatterplot(train['WoodDeckSF'],train['SalePrice'])
train['OpenPorchSF'].isna().sum(),test['OpenPorchSF'].isna().sum()
sns.scatterplot(train['OpenPorchSF'],train['SalePrice'])
train['EnclosedPorch'].isna().sum(),test['EnclosedPorch'].isna().sum()
sns.scatterplot(train['EnclosedPorch'],train['SalePrice'])
train['3SsnPorch'].isna().sum(),test['3SsnPorch'].isna().sum()
sns.scatterplot(train['3SsnPorch'],train['SalePrice'])
train['ScreenPorch'].isna().sum(),test['ScreenPorch'].isna().sum()
sns.scatterplot(train['ScreenPorch'],train['SalePrice'])
train['PoolArea'].isna().sum(),test['PoolArea'].isna().sum()
sns.scatterplot(train['PoolArea'],train['SalePrice'])
sns.scatterplot(train['MiscVal'],train['SalePrice'])
train['MoSold'].isna().sum(),test['MoSold'].isna().sum()
train['MoSold'].value_counts().sort_values().plot(kind='bar')
sns.boxplot(train['MoSold'],train['SalePrice'])
train.groupby('MoSold')['SalePrice'].agg(['count','min','max','mean']).sort_index()
train['MoSold'] = train['MoSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)
train['YrSold'].isna().sum(),test['YrSold'].isna().sum()
train['YrSold'].value_counts().sort_values(ascending=False).head(10)
sns.boxplot(train['YrSold'],train['SalePrice'])
train.groupby(['MoSold','YrSold'])['SalePrice'].count().sort_values(ascending=False)
train['YrSold'] = train['YrSold'].astype(str)

test['YrSold'] = test['YrSold'].astype(str)
train['SaleType'].isna().sum(),test['SaleType'].isna().sum()
test['SaleType'].fillna(test['SaleType'].value_counts().index[0],inplace=True)
train['SaleType'].value_counts().sort_values().plot(kind='bar')
test['SaleType'].value_counts().sort_values().plot(kind='bar')
train['SaleCondition'].isna().sum(),test['SaleCondition'].isna().sum()
train['SaleCondition'].value_counts().sort_values().plot(kind='bar')
test['SaleCondition'].value_counts().sort_values().plot(kind='bar')
#train['Total_sqr_footage'] = (train['BsmtFinSF1'] + train['BsmtFinSF2'] + train['1stFlrSF'] + train['2ndFlrSF'])

train['Total_Bathrooms'] = (train['FullBath'] + (0.5*train['HalfBath']) +  train['BsmtFullBath'] + (0.5*train['BsmtHalfBath']))

train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] + train['EnclosedPorch'] + train['ScreenPorch'] + train['WoodDeckSF'])



#test['Total_sqr_footage'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] + test['1stFlrSF'] + test['2ndFlrSF'])

test['Total_Bathrooms'] = (test['FullBath'] + (0.5*test['HalfBath']) +  test['BsmtFullBath'] + (0.5*test['BsmtHalfBath']))

test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] + test['EnclosedPorch'] + test['ScreenPorch'] + test['WoodDeckSF'])
train.isna().sum()[train.isna().sum() !=0]
test.isna().sum()[test.isna().sum() !=0]
for col in train.columns:

    if train[col].dtype == "object":

        print ("Cardinality of %s variable in Training Data:"%col,train[col].nunique())

        print ("Cardinality of %s variable in Testing Data:"%col,test[col].nunique())

        print ("\n")
## Plot sizing. 

plt.subplots(figsize = (35,20))

## plotting heatmap.  

sns.heatmap(train.corr(), cmap="BrBG", annot=True, center = 0, );

## Set the title. 

plt.title("Heatmap of all the Features", fontsize = 30);
train.corr()['SalePrice'].sort_values(ascending=False)
train.drop(['GarageCars','TotRmsAbvGrd','1stFlrSF'],axis=1,inplace=True)

test.drop(['GarageCars','TotRmsAbvGrd','1stFlrSF'],axis=1,inplace=True)
train.corr()['SalePrice'].sort_values(ascending=False)
# saleprice correlation matrix

corr_num = 20 #number of variables for heatmap

cols_corr = train.corr().nlargest(corr_num, 'SalePrice')['SalePrice'].index

corr_mat_sales = np.corrcoef(train[cols_corr].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(15,15))

hm = sns.heatmap(corr_mat_sales, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",annot_kws = {'size':12}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)

plt.show()
train.shape, test.shape
trn = train.copy()

tst = test.copy()
y = train['SalePrice']

train.drop('SalePrice',inplace=True,axis=1)

len_train = len(train)

full_data = pd.concat([train,test],axis=0).reset_index(drop=True)

full_data = pd.get_dummies(full_data)

train = full_data[:len_train]

test = full_data[len_train:]
train.shape, test.shape
set(train.columns)-set(test.columns)
overfit = []

for col in train.columns:

    counts = train[col].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(train) * 100 >99:

        overfit.append(col)

print (len(overfit))
train.drop(overfit,axis=1,inplace=True)

test.drop(overfit,axis=1,inplace=True)
X = train
X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=0.25)

print ("X_Train Shape:",X_Train.shape)

print ("X_Test Shape:",X_Test.shape)

print ("y_Train Shape:",y_Train.shape)

print ("y_Test Shape:",y_Test.shape)
kfolds = KFold(n_splits=5, shuffle=True)

def rmse(true,pred):

    return np.sqrt(mean_squared_error(true,pred))

def feature_importance(model,data):

    ser = pd.Series(model.coef_,data.columns).sort_values()

    plt.figure(figsize=(14,14))

    ser.plot(kind='bar')

def cv_score(model):

    return np.mean(np.sqrt(-(cross_val_score(model,X,y,cv=kfolds,scoring='neg_mean_squared_error'))))

def plot_importance(model,indep):

    Ser = pd.Series(model.coef_,indep.columns).sort_values()

    plt.figure(figsize=(30,20))

    Ser.plot(kind='bar')

def calc_r2(model,true,data):

    return r2_score(true,model.predict(data))
lr = LinearRegression()

lr.fit(X_Train,y_Train)

print ("Linear Regression, Training Set RMSE:",rmse(y_Train,lr.predict(X_Train)))

print ("Linear Regression,Training R Squared:",calc_r2(lr,y_Train,X_Train))

print ("\nLinear Regression,Testing Set RMSE:",rmse(y_Test,lr.predict(X_Test)))

print ("Linear Regression,Testing R Squared:",calc_r2(lr,y_Test,X_Test))

print ("\nLinear Regression,Cross Validation Score:",cv_score(lr))
lasso = Lasso(alpha=0.001,max_iter=5000)

lasso.fit(X_Train,y_Train)

print ("Lasso Regression, Training Set RMSE:",rmse(y_Train,lasso.predict(X_Train)))

print ("Lasso Regression,Training R Squared:",calc_r2(lasso,y_Train,X_Train))

print ("\nLasso Regression,Testing Set RMSE:",rmse(y_Test,lasso.predict(X_Test)))

print ("Lasso Regression,Testing R Squared:",calc_r2(lasso,y_Test,X_Test))

print ("\nLasso Regression,Cross Validation Score:",cv_score(lasso))
#Submission['SalePrice'] = np.expm1(lasso.predict(test))

#Submission.to_csv("Lasso_Latest.csv",index=None)
coeffs = pd.DataFrame(list(zip(X.columns, lasso.coef_)), columns=['Predictors', 'Coefficients'])

coeffs.sort_values(by='Coefficients',ascending=False)
scores = []

alpha = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10]

for i in alpha:

    las = Lasso(alpha=i,max_iter=10000)

    las.fit(X_Train,y_Train)

    scores.append(rmse(y_Test,las.predict(X_Test)))

print ("Lasso Scores with Different Alpha Values \n",scores)
#Submission['SalePrice'] = np.expm1(lasso.predict(test))

#Submission.to_csv("Sub.csv",index=None)
ridge = Ridge()

ridge.fit(X_Train,y_Train)

print ("Ridge Regression,Training Set RMSE:",rmse(y_Train,ridge.predict(X_Train)))

print ("Ridge Regression,Training R Squared:",calc_r2(ridge,y_Train,X_Train))

print ("\nRidge Regression,Testing Set RMSE:",rmse(y_Test,ridge.predict(X_Test)))

print ("Ridge Regression,Testing R Squared:",calc_r2(ridge,y_Test,X_Test))

print ("\nRidge Regression,Cross Validation Score:",cv_score(ridge))
scores = []

alpha = [0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10]

for i in alpha:

    ridge = Ridge(alpha=i)

    ridge.fit(X_Train,y_Train)

    scores.append(rmse(y_Test,ridge.predict(X_Test)))

print ("Ridge Scores with Different Alpha Values \n",scores)
en = ElasticNet(max_iter=5000)

params = {"alpha":[0.0001,0.0002,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10],

          "l1_ratio":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}

grid = GridSearchCV(estimator=en,param_grid=params,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')

grid.fit(X,y)
grid.best_estimator_,grid.best_params_,grid.best_score_
en = grid.best_estimator_

en.fit(X_Train,y_Train)

print ("Elastic Net Regression,Training Set RMSE:",rmse(y_Train,en.predict(X_Train)))

print ("Elastic Net Regression,Training R Squared:",calc_r2(en,y_Train,X_Train))

print ("\nElastic Net Regression,Testing Set RMSE:",rmse(y_Test,en.predict(X_Test)))

print ("Elastic Net Regression,Testing R Squared:",calc_r2(en,y_Test,X_Test))

print ("\nElastic Net Regression,Cross Validation Score:",cv_score(en))
en_submission = np.expm1(en.predict(test))

lasso_submission = np.expm1(lasso.predict(test))

ridge_submission = np.expm1(ridge.predict(test))

average = (en_submission+lasso_submission+ridge_submission)/3

Submission['SalePrice'] = average

Submission.to_csv("Stacked7.csv",index=None)