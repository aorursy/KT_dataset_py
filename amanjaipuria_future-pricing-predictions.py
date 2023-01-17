# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print(train.shape)

train.head()
print(test.shape)

test.head()
train.info()
import matplotlib.pyplot as plt

import seaborn as sns



# Suppress Warnings

import warnings

warnings.filterwarnings('ignore')



# Setting no of columns to 500 to view each and every column whenever printed

pd.set_option('display.max_columns', 500)
train['Id'].nunique()
train.describe()
train.describe(include=['O'])
sns.distplot(train['SalePrice'])
sns.distplot(train['LotFrontage'])
# sns.distplot(train['MasVnrArea'])
ID_train = train['Id']

ID_test = test['Id']
train.drop('Id',axis=1,inplace=True)

test.drop("Id",axis=1,inplace=True)



print(train.shape)

print(test.shape)
y_train = train['SalePrice']

X_train = train.drop('SalePrice',axis=1)



merge_df = pd.concat([X_train,test]).reset_index(drop=True)



print(merge_df.shape)
100*merge_df.isnull().sum()/len(merge_df)
merge_df.columns[round(100*merge_df.isnull().sum()/len(merge_df),2)>60.0]
merge_df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'],axis=1,inplace=True)
merge_df[merge_df.columns[round(100*merge_df.isnull().sum(axis=0)/len(merge_df),2)!=0]].info()
merge_df['MSZoning'].value_counts()
merge_df.shape
merge_df['MSZoning'] = merge_df['MSZoning'].fillna(merge_df['MSZoning'].mode()[0])
print(merge_df['Utilities'].value_counts())

print(merge_df['Exterior1st'].value_counts())

print(merge_df['Exterior2nd'].value_counts())

print(merge_df['MasVnrType'].value_counts())
merge_df.drop('Utilities',axis=1,inplace=True)

merge_df['Exterior1st'] = merge_df['Exterior1st'].fillna(merge_df['Exterior1st'].mode()[0])

merge_df['Exterior2nd'] = merge_df['Exterior2nd'].fillna(merge_df['Exterior2nd'].mode()[0])

merge_df['MasVnrType'] = merge_df['MasVnrType'].fillna("None")
print(merge_df['BsmtQual'].value_counts())

print(merge_df['BsmtCond'].value_counts())

print(merge_df['BsmtExposure'].value_counts())

print(merge_df['BsmtFinType1'].value_counts())

print(merge_df['BsmtFinType2'].value_counts())
merge_df['BsmtQual'] = merge_df['BsmtQual'].fillna("NA")



# 'BsmtQual' being an ordered categorical column, let us replace values with ordered numbers

merge_df['BsmtQual'] = merge_df['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'NA':0})
merge_df['BsmtCond'] = merge_df['BsmtCond'].fillna("NA")



# 'BsmtCond' being an ordered categorical column, let us replace values with ordered numbers

merge_df['BsmtCond'] = merge_df['BsmtCond'].map({'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0})
merge_df['BsmtExposure'] = merge_df['BsmtExposure'].fillna("NA")



# 'BsmtExposure' being an ordered categorical column, let us replace values with ordered numbers

merge_df['BsmtExposure'] = merge_df['BsmtExposure'].map({'Gd':3,'Av':2,'Mn':1,'No':0,'NA':0})
merge_df['BsmtFinType1'] = merge_df['BsmtFinType1'].fillna("NA")



# 'BsmtFinType1' being an ordered categorical column, let us replace values with ordered numbers

merge_df['BsmtFinType1'] = merge_df['BsmtFinType1'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0})
merge_df['BsmtFinType2'] = merge_df['BsmtFinType2'].fillna("NA")



# 'BsmtFinType2' being an ordered categorical column, let us replace values with ordered numbers

merge_df['BsmtFinType2'] = merge_df['BsmtFinType2'].map({'GLQ':6,'ALQ':5,'BLQ':4,'Rec':3,'LwQ':2,'Unf':1,'NA':0})
print(merge_df['KitchenQual'].value_counts())

print(merge_df['Functional'].value_counts())

print(merge_df['FireplaceQu'].value_counts())
merge_df['KitchenQual'] = merge_df['KitchenQual'].fillna(merge_df['KitchenQual'].mode()[0])



# Converting ordered categorical values to ordered numbers

merge_df['KitchenQual'] = merge_df['KitchenQual'].map({'TA':3,'Gd':4,'Ex':5,'Fa':2})
merge_df.drop('Functional',axis=1,inplace=True)
merge_df['FireplaceQu'] = merge_df['FireplaceQu'].fillna('NA')



# Converting ordered categorical values to ordered numbers

merge_df['FireplaceQu'] = merge_df['FireplaceQu'].map({'TA':3,'Gd':4,'Ex':5,'Fa':2,'Po':1,'NA':0})
print(merge_df['GarageType'].value_counts())

print(merge_df['GarageFinish'].value_counts())

print(merge_df['GarageQual'].value_counts())

print(merge_df['GarageCond'].value_counts())

print(merge_df['SaleType'].value_counts())
merge_df['GarageType'] = merge_df['GarageType'].fillna('NA')
merge_df['GarageFinish'] = merge_df['GarageFinish'].fillna('NA')



# 'GarageFinish' being an ordered categorical column, let us replace values with ordered numbers

merge_df['GarageFinish'] = merge_df['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'NA':0})
# 'GarageQual','GarageCond' & 'SaleType' has same values occuring around 90% times in dataset

# Its good to drop all three of these columns



merge_df = merge_df.drop(['GarageQual','GarageCond','SaleType'],axis=1)

merge_df.shape
merge_df['Electrical'].value_counts()
merge_df.drop('Electrical',axis=1,inplace=True)
float_col = merge_df[merge_df.columns[round(100*merge_df.isnull().sum(axis=0)/len(merge_df),2)!=0]].columns

float_col
float_col = float_col.drop('GarageYrBlt')

float_col
merge_df[float_col].describe()
for col in float_col:

    merge_df[col] = merge_df[col].fillna(merge_df[col].median())
# 'GarageYrBlt'



print(merge_df['GarageYrBlt'].max())

print(merge_df['GarageYrBlt'].min())
merge_df.loc[merge_df['GarageYrBlt']==2207,:]
print(merge_df['YearBuilt'].max())

print(merge_df['YearBuilt'].min())
sns.boxplot(merge_df['GarageYrBlt'])
merge_df.loc[merge_df['GarageYrBlt']==2207,'GarageYrBlt'] = 2007
sns.boxplot(merge_df['GarageYrBlt'])
print(merge_df['GarageYrBlt'].max())

print(merge_df['GarageYrBlt'].min())
merge_df['GarageYrBlt'] = merge_df['GarageYrBlt'].apply(lambda x:2010.0-x)
merge_df['GarageYrBlt'] = merge_df['GarageYrBlt'].fillna(merge_df['GarageYrBlt'].median())
merge_df.isnull().any().sum()
merge_df[['YearBuilt', 'YearRemodAdd', 'YrSold']].max()
merge_df[['YearBuilt', 'YearRemodAdd', 'YrSold']] = merge_df[['YearBuilt', 'YearRemodAdd', 'YrSold']].apply(lambda x:2010.0-x)
CategoryCol = merge_df.select_dtypes(include='object').columns

CategoryCol
for i in CategoryCol:

    print(100*merge_df[i].value_counts()/len(merge_df))
merge_df.drop(['Street','LandSlope','Condition2','RoofMatl','Heating',],axis=1,inplace=True)
merge_df['ExterQual'] = merge_df['ExterQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2})

merge_df['ExterCond'] = merge_df['ExterCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})

merge_df['HeatingQC'] = merge_df['HeatingQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})

merge_df['CentralAir'] = merge_df['CentralAir'].map({'Y':1,'N':0})
sns.distplot(y_train)
y_train = np.log1p(y_train)
sns.distplot(y_train)
merge_df.info()
sns.boxplot(x=merge_df['MSZoning'],y=y_train)
sns.boxplot(x=merge_df['LotShape'],y=y_train)
sns.boxplot(x=merge_df['BldgType'],y=y_train)
sns.boxplot(x=merge_df['HouseStyle'],y=y_train)
plt.figure(figsize=(12,6))

sns.boxplot(x=merge_df['Neighborhood'],y=y_train)

plt.xticks(rotation=90)
plt.figure(figsize=(20,12))

sns.heatmap(merge_df.corr(),annot=True,cmap='coolwarm')

plt.show()
correlation = merge_df.corr().abs().unstack().sort_values(ascending=False)

correlation = correlation[correlation<1]

correlation.drop_duplicates().head(10)
cat_cols = merge_df.select_dtypes(include='object').columns

cat_cols
dummy_df = pd.get_dummies(merge_df[cat_cols],drop_first=True)

dummy_df.head()
dummy_df.shape
housing = pd.concat([merge_df,dummy_df],axis=1)

housing = housing.drop(cat_cols,axis=1)           # Dropping original categorical columns

print(housing.shape)

housing.head()
housing.info()
train = housing[:train.shape[0]]

test = housing[train.shape[0]:]



print(train.shape)
print(test.shape)
y_train.shape
# Importing library for test-train split



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn import metrics
X_train,X_test,y_train,y_test = train_test_split(train, y_train, test_size = 0.2, random_state = 45)
scaler = MinMaxScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Building a Ridge regression model taking alpha = 0.01



# Initializing Ridge regression

lm_ridge = Ridge(alpha=0.01)



# Fitting the model on train dataset

lm_ridge.fit(X_train,y_train)



# Prediction on train data

y_train_pred = lm_ridge.predict(X_train)

print(metrics.r2_score(y_train,y_train_pred))



# Prediction on test data

y_test_pred = lm_ridge.predict(X_test)

print(metrics.r2_score(y_test,y_test_pred))
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# Cross Validation 

folds = KFold(n_splits=6, shuffle=True, random_state=10)



# List of alpha to tune the model

params = {'alpha':[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20, 50, 100]}



model = Ridge()

model_cv = GridSearchCV(estimator=model, param_grid=params, scoring='neg_root_mean_squared_error', cv=folds, return_train_score=True, verbose = 1)



model_cv.fit(X_train, y_train)
cv_ridge_res = pd.DataFrame(model_cv.cv_results_)



# Plot of RMSE at various alpha values for train and test dataset

plt.figure(figsize=(12,6))

cv_ridge_res['param_alpha'] = cv_ridge_res['param_alpha'].astype('float32')

plt.plot(cv_ridge_res['param_alpha'], cv_ridge_res['mean_train_score'])

plt.plot(cv_ridge_res['param_alpha'], cv_ridge_res['mean_test_score'])

plt.xlabel('alpha', fontsize=12)

plt.ylabel('Neg_RMSE', fontsize=12)

plt.legend(loc='upper right')

plt.grid()

plt.show()
model_cv.best_params_
model_cv.best_score_
# Building final ridge regression model with optimum alpha = 5.0



lm_ridge = Ridge(alpha=5.0)



lm_ridge.fit(X_train,y_train)



y_train_pred = lm_ridge.predict(X_train)

print(metrics.r2_score(y_train,y_train_pred))

y_test_pred = lm_ridge.predict(X_test)

print(metrics.r2_score(y_test,y_test_pred))
# Building 1st Lasso regression model taking alpha = 0.01



lm_lasso = Lasso(alpha=0.01)



lm_lasso.fit(X_train,y_train)



y_train_pred = lm_lasso.predict(X_train)

print(metrics.r2_score(y_train,y_train_pred))

y_test_pred = lm_lasso.predict(X_test)

print(metrics.r2_score(y_test,y_test_pred))
# List of parameters to tune lasso model

params2 = {'alpha':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0007, 0.0009, 0.001, 0.005]}



model2 = Lasso()

model2_cv = GridSearchCV(estimator=model2, param_grid=params2, scoring='neg_root_mean_squared_error', cv=folds, return_train_score=True, verbose = 1)



model2_cv.fit(X_train, y_train)
cv_lasso_res = pd.DataFrame(model2_cv.cv_results_)



# Plot of RMSE at various alpha for both train and test dataset

plt.figure(figsize=(12,6))

cv_lasso_res['param_alpha'] = cv_lasso_res['param_alpha'].astype('float32')

plt.plot(cv_lasso_res['param_alpha'], cv_lasso_res['mean_train_score'])

plt.plot(cv_lasso_res['param_alpha'], cv_lasso_res['mean_test_score'])

plt.xlabel('alpha', fontsize=12)

plt.ylabel('Neg_RMSE', fontsize=12)

plt.legend(loc='upper right')

plt.grid()

plt.show()
model2_cv.best_params_
model2_cv.best_score_
# Building lasso regression model with optimum alpha = 0.0007



lm_lasso = Lasso(alpha=0.0007)



lm_lasso.fit(X_train,y_train)



y_train_pred = lm_lasso.predict(X_train)

print(metrics.r2_score(y_train,y_train_pred))



y_test_pred = lm_lasso.predict(X_test)

print(metrics.r2_score(y_test,y_test_pred))
# Making a dataframe of features and corresponding coefficients



coef_lasso = list(lm_lasso.coef_)

coef_lasso.insert(0,lm_lasso.intercept_)

col_lasso = train.columns

col_lasso.insert(0,'constant')

lasso_coef = pd.DataFrame(list(zip(col_lasso,coef_lasso)))

lasso_coef.columns = ['Feature','lasso_Coefficient']
# Getting top 10 features



lasso_coef.sort_values(by='lasso_Coefficient',ascending=False).head(10)
# Top features with negative impact on saleprice



lasso_coef.sort_values(by='lasso_Coefficient',ascending=True).head(10)
res_train = y_train - y_train_pred

res_test = y_test - y_test_pred



sns.distplot(res_train,hist=False)

sns.distplot(res_test,hist=False)

plt.title('Residuals Analysis')

plt.xlabel('Residuals')
test = scaler.transform(test)
# Prediction on test



prediction = np.expm1(lm_lasso.predict(test))
# Final Submission



final = pd.DataFrame()

final['Id'] = ID_test

final['SalePrice'] = prediction

final.to_csv('submission.csv',index=False)