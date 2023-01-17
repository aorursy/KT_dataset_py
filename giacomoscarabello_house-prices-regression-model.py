# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor

import scipy.stats as stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Importing and EDA



train_data=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_data.info()

test_data.info()





#label and identify numerical and categorical, useful for further EDA

labels_num = list(train_data.select_dtypes([np.number]).columns)

labels_num_test=list(labels_num)

labels_num_test.remove("SalePrice")

labels_cat=list(train_data.select_dtypes([object]).columns)

labels_cat_test=list(test_data.select_dtypes([object]).columns)

train_data_num=train_data[labels_num]

train_data_cat=train_data[labels_cat]

#should include target variable

train_data_cat["SalePrice"]=train_data["SalePrice"]

#remove ID

del labels_num[0]
#check correlations and focus on highly correlated variables to check for outliers



num_labels=(train_data_num.corr().loc[:,"SalePrice"].sort_values(ascending=False

                                                                ))

print(num_labels.head(10))

print(num_labels.tail(10))



#first one is the target variable.



for i in num_labels.index[1:10]:

    fig=plt.figure()

    sns.regplot(x=i,y="SalePrice",data=train_data)

    plt.show
index=train_data[train_data["GrLivArea"]>4000].index

train_data.drop(index,axis=0)



index=train_data[train_data["SalePrice"]>700000].index

train_data.drop(index,axis=0)

y=train_data.SalePrice

sns.distplot(y, fit=stats.norm, hist_kws=dict(edgecolor='w',linewidth=2))

fig=plt.figure()

res=stats.probplot(y, plot=plt)
y=np.log1p(y)

sns.distplot(y, fit=stats.norm, hist_kws=dict(edgecolor='w',linewidth=2))

fig=plt.figure()

res=stats.probplot(y, plot=plt)
num_labels_na = (train_data_num.isnull().sum() / len(train_data_num)) * 100

num_labels_na = num_labels_na.drop(num_labels_na[num_labels_na == 0].index).sort_values(ascending=False)



print("Percentage of NaNs:\n"+str(num_labels_na))
train_data["LotFrontage"].fillna(train_data["LotFrontage"].median(),inplace=True)

train_data["GarageYrBlt"].fillna(value=0,inplace=True)

train_data["MasVnrArea"].fillna(train_data["MasVnrArea"],inplace=True)



test_data["LotFrontage"].fillna(test_data["LotFrontage"].median(),inplace=True)

test_data["GarageYrBlt"].fillna(value=0,inplace=True)

test_data["MasVnrArea"].fillna(test_data["MasVnrArea"],inplace=True)



#check for remaining NaNs



print(train_data[labels_num].isnull().sum().nlargest(10))

print(test_data[labels_num_test].isnull().sum().nlargest(10))
#Fill the rest of the test data NAs with median

train_data=train_data.fillna(train_data[labels_num].median())

print(train_data[labels_num].isnull().sum().nlargest(10))



test_data=test_data.fillna(test_data[labels_num_test].median())

print(test_data[labels_num_test].isnull().sum().nlargest(10))

train_data['TotalSF']=train_data['TotalBsmtSF']+train_data['GrLivArea']+train_data['1stFlrSF']+train_data['2ndFlrSF']

train_data['TotalBath']=train_data['BsmtHalfBath']+train_data['BsmtFullBath']+train_data['HalfBath']+train_data['FullBath']

test_data['TotalSF']=test_data['TotalBsmtSF']+test_data['GrLivArea']+test_data['1stFlrSF']+test_data['2ndFlrSF']

test_data['TotalBath']=test_data['BsmtHalfBath']+test_data['BsmtFullBath']+test_data['HalfBath']+test_data['FullBath']





plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

plt.scatter(train_data.TotalSF,train_data.SalePrice)

plt.xlabel('TotalSF')

plt.ylabel('SalePrice')

plt.subplot(1,2,2)

plt.scatter(train_data.TotalBath,train_data.SalePrice)

plt.xlabel('TotalBath')

plt.ylabel('SalePrice')

# categoricals EDA

#NaN have a meaning in Categoticals, i replace them with "NNN"

cat_na = (train_data_cat.isnull().sum() / len(train_data_cat)) * 100

cat_na = cat_na.drop(cat_na[cat_na == 0].index).sort_values(ascending=False)



print(cat_na)





#fill NA values

train_data=train_data.fillna(value="None")

test_data=test_data.fillna(value="None")



print(train_data.isnull().sum())

print(test_data.isnull().sum())



for i in range(len(labels_cat)):

    fig=plt.figure(figsize=[15,4])

    g=sns.boxplot(x=labels_cat[i],y="SalePrice", data=train_data)

    g.set_xticklabels(g.get_xticklabels(),rotation=30)

    plt.show()

encoded = pd.get_dummies(train_data)

test_encoded = pd.get_dummies(test_data)



a=list(encoded.columns)

b=list(test_encoded.columns)

            

list1 = list(set(a)-set(b))

list1.remove("SalePrice")

print(list1)



encoded=encoded.drop(list1,axis=1)



list2=list(set(b)-set(a))

print(list2)



test_encoded=test_encoded.drop(list2,axis=1)



print(encoded.shape)

print(test_encoded.shape)

            
#select 50% best features. Use RFE, because Both categorical and numerical data



X=encoded.drop(["SalePrice","Id"],axis=1).values



rf = RandomForestRegressor(n_estimators=300,random_state=2)



rf.fit(X,y)



# Create a pd.Series of features importances

importances_rf = pd.Series(rf.feature_importances_, index = encoded.drop(["SalePrice","Id"],axis=1).columns)

# Sort importances_rf: 50% most informative features

sorted_importances_rf = importances_rf.sort_values().nlargest(round(len(importances_rf)/2))

# Make a horizontal bar plot

fig=plt.figure(figsize=[10,20])

sorted_importances_rf.plot(kind='barh', color='lightgreen'); 

plt.show()
from sklearn.svm import SVR

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import Pipeline



X=encoded.loc[:,list(sorted_importances_rf.index)].values



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1)



#regr=Pipeline(steps=[("scaler",RobustScaler()),("SVR",SVR())])

#regr=Pipeline(steps=[("scaler",RobustScaler()),("net",ElasticNet(max_iter=100000))])

regr=Pipeline(steps=[("scaler",StandardScaler()),("lasso",Lasso(random_state=1,max_iter=100000))])



#parameters = {"net__alpha":np.arange(0.0001,0.001, 0.0001),"net__l1_ratio":np.arange(0.25,1,0.05)}

parameters = {"lasso__alpha":np.arange(0.0001,0.001, 0.0001)}

#parameters = {"SVR__C":np.arange(0.5,5,0.5),"SVR__gamma":np.arange(0.001,0.01,0.001)}





cv = GridSearchCV(regr, param_grid=parameters)

cv.fit(X_train, y_train)



best_hyperparams = cv.best_params_

print('Best hyerparameters:\n', best_hyperparams)



# Extract best model from 'grid_rf'

lasso_best_model = cv.best_estimator_



# Predict the test set labels

y_pred = lasso_best_model.predict(X_test)



# Evaluate the test set RMSE

rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE

print('Test set RLMSE of Lasso: {:.3f}'.format(rmse_test))



#cv = GridSearchCV(pipeline, param_grid=parameters)

#cv.fit(X_train, y_train)

#y_pred = cv.predict(X_test)



#run random forest with best features



X=encoded.loc[:,list(sorted_importances_rf.index)].values



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1)



param_grid = { 

    'max_features': ["auto"],

    'max_depth' : [8],

    "n_estimators":[500]

    }



rf = RandomForestRegressor(random_state=2)



CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)

CV_rf.fit(X_train, y_train)



best_hyperparams = CV_rf.best_params_

print('Best hyerparameters:\n', best_hyperparams)



# Extract best model from 'grid_rf'

rf_best_model = CV_rf.best_estimator_

# Predict the test set labels

y_pred = rf_best_model.predict(X_test)



# Evaluate the test set RMSE

rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE

print('Test set RLMSE of rf: {:.3f}'.format(rmse_test))

# let s try another algorithm, GradientBoosting



X=encoded.loc[:,list(sorted_importances_rf.index)].values



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=1)



param_grid = { 

    'max_features': ['sqrt'],

    'max_depth' : [2],

    "n_estimators":[1000]

    }



rf = GradientBoostingRegressor(random_state=2)



CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)

CV_rf.fit(X_train, y_train)



best_hyperparams = CV_rf.best_params_

print('Best hyerparameters:\n', best_hyperparams)



# Extract best model from 'grid_rf'

GBbest_model = CV_rf.best_estimator_

# Predict the test set labels

y_pred = GBbest_model.predict(X_test)



# Evaluate the test set RMSE

rmse_test = MSE(y_test, y_pred)**(1/2)

# Print the test set RMSE

print('Test set RLMSE of GradientBoost: {:.3f}'.format(rmse_test))



fig,ax=plt.subplots(1,2,figsize=[15,5])



sns.regplot(x=y_test,y=y_pred,ax=ax[0])

sns.residplot(x=y_test,y=y_pred,ax=ax[1])
#submission



test_encoded=test_encoded[list(sorted_importances_rf.index)]

X=test_encoded.values



y_pred = GBbest_model.predict(X)

y_pred=np.expm1(y_pred)



#output

output = pd.DataFrame({'Id': test_data.Id,'SalePrice': y_pred})

output.to_csv('submission.csv', index=False)



print(y_pred)