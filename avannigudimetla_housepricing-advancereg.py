#importing libraries

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

import os

from sklearn import metrics

import seaborn as sns

import numpy as np



# hide warnings

import warnings

warnings.filterwarnings('ignore')
import os

os.getcwd()
os.chdir('/kaggle')
os.chdir('input')
os.listdir()
df=pd.read_csv("train (1).csv")
df.head()
df.columns
df.shape
df.info()
df.dtypes
round(df.isnull().sum()*100/len(df.index),2)
df=df.drop(['Id','MiscFeature','GarageYrBlt','YearBuilt','MiscVal','PoolArea','OverallQual','YrSold'],axis=1)
#Imputing the values for the columns with nulls

df['MasVnrType']=df['MasVnrType'].replace(['NA','None'],'others')

#Imputing with the median value

df['LotFrontage']=df['LotFrontage'].replace(['NA'],'69')

df['LotFrontage']=df['LotFrontage'].replace(np.nan,'69')

df['GarageType']=df['GarageType'].replace(['NA'],'Others')

df['GarageFinish']=df['GarageFinish'].replace(['NA'],'Others')

df['GarageQual']=df['GarageQual'].replace(['NA'],'Others')

df['GarageCond']=df['GarageCond'].replace(['NA'],'Others')

df['PoolQC']=df['PoolQC'].replace(['NA'],'Others')

df['Fence']=df['Fence'].replace(['NA'],'Others')

df['Alley']=df['Alley'].replace(['NA'],'Others')

df['FireplaceQu']=df['FireplaceQu'].replace(['NA'],'Others')

df['BsmtFinType2']=df['BsmtFinType2'].replace(['NA'],'Others')

df['Electrical']=df['Electrical'].replace(['NA'],'SBrkr')

df['MasVnrArea']=df['MasVnrArea'].replace(np.nan,'0')
round(df.isnull().sum()*100/len(df.index),2)
ds=df.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1])

ds
num=ds.columns

num=num.drop(['YearRemodAdd'])

print(num)

print(len(num))

plt.figure(figsize=(20,15))

for i in range(1,29):

    plt.subplot(5,6,i)

    sns.boxplot(y=df[num[i-1]])
# simple density plot

df['SalePrice']=np.log(df['SalePrice'])

sns.distplot(df['SalePrice'])

plt.show()
sns.distplot(df['SalePrice'])

plt.show()
df=df.loc[df['MSSubClass']<150]

df=df.loc[df['LotArea']<50000]

df=df.loc[df['BsmtFinSF1']<2000]

df=df.loc[df['WoodDeckSF']<500]

df=df.loc[df['TotalBsmtSF']<3000]

df.loc[df['3SsnPorch']>0,['3SsnPorch']]=1

df.loc[df['ScreenPorch']>0,['ScreenPorch']]=1

df.loc[df['BsmtFinSF2']>0,['BsmtFinSF2']]=1

df.loc[df['EnclosedPorch']>0,['EnclosedPorch']]=1

df.loc[df['LowQualFinSF']>0,['LowQualFinSF']]=1
plt.figure(figsize=(20,15))

for i in range(1,29):

    plt.subplot(5,6,i)

    sns.boxplot(y=df[num[i-1]])
#Combining columns which have similar data

df['TotalFullBath']=df['BsmtFullBath']+df['FullBath']

df['TotalHalfBath']=df['BsmtHalfBath']+df['HalfBath']

# Post the above step, we drop those cols which now become redundant

df=df.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1)
#Creating a new column to compute the age of the house

df['HouseAgeYrs']=2019-df['YearRemodAdd']

#Dropping the original column

df=df.drop(['YearRemodAdd'],axis=1)
plt.figure(figsize = (30, 25))

sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")

plt.show()
dummies = pd.get_dummies(df[['Alley','FireplaceQu','Fence','PoolQC','MSZoning', 'Street', 'LotShape','LandContour','Utilities','LotConfig',

                                'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',

                                'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

                                'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',

                                'Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',

                                'SaleType','SaleCondition']])

df = pd.concat([df,dummies],axis=1)

df = df.drop(['Alley','FireplaceQu','Fence','PoolQC','MSZoning', 'Street', 'LotShape','LandContour','Utilities','LotConfig',

                                'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',

                                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',

                                'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

                                'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',

                                'Functional','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',

                                'SaleType','SaleCondition'],axis=1)
df.head()
#test-train split and min-max scaling

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF','GrLivArea',

           'TotalFullBath', 'TotalHalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd','Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

           'MoSold', 'SalePrice']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
Y_train = df_train.pop('SalePrice')

X_train = df_train
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

# Running RFE 

lm = LinearRegression()

lm.fit(X_train,Y_train)

rfe = RFE(lm,80)             # running RFE

rfe = rfe.fit(X_train,Y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
cols = X_train.columns[rfe.support_]

cols
X_train.columns[~rfe.support_]
# Create the X_test dataframe 

X_train_rfe = X_train[cols]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
# Run the linear model

lm = sm.OLS(Y_train,X_train_rfe).fit()  
print(lm.summary())
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}



ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train_rfe, Y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha = 3

ridge = Ridge(alpha=alpha)



ridge.fit(X_train_rfe, Y_train)

ridge.coef_
Features_v1=pd.DataFrame(X_train_rfe.columns,columns=['Variables'])

Features_v1['Feature_Coeff']=ridge.coef_

top_Features=Features_v1[Features_v1['Feature_Coeff']>0]

Top=top_Features.sort_values(by='Feature_Coeff',ascending=False).head(9)

Top
params = {'alpha': [0.0001,0.0002, 0.0003, 0.0004, 0.0005,0.01]}

lasso = Lasso()

# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train_rfe,Y_train)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha =0.01



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train_rfe, Y_train)
lasso.coef_
Features=pd.DataFrame(X_train_rfe.columns,columns=['Variables'])

Features['Feature_Coeff']=lasso.coef_

top_Features=Features[Features['Feature_Coeff']>0]

top_Features
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

df=df[['GrLivArea','TotalBsmtSF','OverallCond','LotArea','GarageCars','MSZoning_FV','OpenPorchSF','GarageArea','CentralAir_Y','SalePrice']]

df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)





num_vars = ['GrLivArea','TotalBsmtSF','OverallCond','LotArea','GarageCars','MSZoning_FV','OpenPorchSF','GarageArea','CentralAir_Y','SalePrice']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()
y_train = df_train.pop('SalePrice')

X_train = df_train
import statsmodels.api as sm  

X_train = sm.add_constant(X_train)
lm = sm.OLS(Y_train,X_train).fit()  
#Let's see the summary of our linear model

print(lm.summary())
y_train_price = lm.predict(X_train)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((Y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18) 
df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('SalePrice')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[['GrLivArea','TotalBsmtSF','OverallCond','LotArea','GarageCars','MSZoning_FV','OpenPorchSF','GarageArea','CentralAir_Y']]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)  
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)