import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn as sk

import scipy as sp

import seaborn as sns

import scipy.stats as stats

import pylab

from sklearn.model_selection import RandomizedSearchCV





import os

for dirname, _, filenames in os.walk('/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





train_df1=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_df1=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")







print(train_df1.shape)
print(test_df1.shape)
print(train_df1.head())
print(train_df1.describe())
#Merging the train and test data

df1=pd.concat([train_df1,test_df1],axis=0)
Missing_df=pd.DataFrame(df1.isnull().sum().sort_values(ascending=False)/len(df1))

Missing_df.head(40)
#Dropping the columns having missing values greater than 50 percent

df1.drop(['PoolQC','MiscFeature','Alley','Fence'],axis=1,inplace=True)
#Other categorical and numerial features has nearly Zero percent missing values Imputing with mode and median.

cat_features_list=df1.select_dtypes(exclude=np.number).columns.tolist()

num_features_list=df1.select_dtypes(include=np.number).columns.tolist()

        

        
#Handling Categorical Features

for feature in cat_features_list:

    df1[feature].fillna(df1[feature].mode()[0],inplace=True)

#Handling Numerical Features

num_features_list.remove('SalePrice')

for feature in num_features_list:

    df1[feature].fillna(df1[feature].median(),inplace=True)
#Ensuring all missing value are handled properly

df1.isnull().sum().sort_values(ascending=False).head(50)


#Derving new Features

df1['YearsOld']=df1['YrSold']-df1['YearBuilt']
#Checking for the top correlated features

corr_matrix=df1.corr()

most_corr_features=corr_matrix.index[abs(corr_matrix['SalePrice']>0.5)]

plt.figure(figsize=(10,10))

sns.heatmap(df1[most_corr_features].corr(),annot=True,cmap="RdYlGn")


df1.skew().sort_values(ascending=False)
#Treating the Variables  with skewness greater than 1 

skewed_fearures=df1.skew().sort_values(ascending=False).head(20).index.tolist()

skewed_fearures.remove('SalePrice')

df1[skewed_fearures]=np.log1p(df1[skewed_fearures])
# Checking the Target Variable



sns.distplot(train_df1['SalePrice'])

fig=plt.figure()

plot_1=stats.probplot(train_df1['SalePrice'],dist='norm', plot=pylab)
#The Target variable is  rightly skewed .Transforming it.

train_df1['SalePrice']=np.log(train_df1['SalePrice']+1)

sns.distplot(train_df1['SalePrice'])

fig=plt.figure()

plot_2 = stats.probplot(train_df1['SalePrice'],dist='norm',plot=pylab)





#Encoding Categorical Features

#One Hot Encoding

df2=pd.get_dummies(df1[cat_features_list],drop_first=True)

df1.drop(cat_features_list,axis=1,inplace=True)

#Merging the transformed dataframes

df3=pd.concat([df1,df2],axis=1)

#Removing the duplicated columns



d3 = df3.loc[:,~df3.columns.duplicated()]
#Dropping the Id Feature

df3.drop(['Id'],axis=1,inplace=True)

df3.shape


#Splitting the data into train and test data

X=df3.drop('SalePrice',axis=1)

Y=df3['SalePrice']

X_train=X.iloc[:1460]

X_test=X.iloc[1460:,]

Y_train=Y.iloc[:1460]

Y_test=Y.iloc[1460:]



Y_train=np.log1p(Y_train)
#Training the Base  Model

from sklearn.linear_model import LinearRegression

lr_model=LinearRegression()

lr_model.fit(X_train,Y_train)

Y_pred_lr=lr_model.predict(X_test)

Y_lr_train=lr_model.predict(X_train)
#Root Mean Sqaured Error

#Evaludating the model

from sklearn.metrics import mean_squared_error

MSE=np.sqrt(mean_squared_error(Y_train,Y_lr_train))

print("RMSE of the Linear Regression Model is :",np.sqrt(MSE))

r2_score=lr_model.score(X_train,Y_train)

print("R2_score of Liner Regression",r2_score)


#Lets check Adjusted R_Squared 

adj_r2 = (1 - (1 - r2_score) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R_Sqaured of Linear Regression Model is :",adj_r2 )
#Training Xgboost Model 

import xgboost

classifier=xgboost.XGBRegressor()

booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]






n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
# Set up the random search with 5-fold cross validation

random_cv = RandomizedSearchCV(estimator=classifier,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_squared_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,Y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.75, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

             importance_type='gain', interaction_constraints='',

             learning_rate=0.15, max_delta_step=0, max_depth=2,

             min_child_weight=3, monotone_constraints='()',

             n_estimators=500, n_jobs=0, num_parallel_tree=1, random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,

             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(X_train,Y_train)
Y_pred_xgb=regressor.predict(X_test)

Y_xgb_train=regressor.predict(X_train)
from sklearn.metrics import mean_squared_error

print("RMSE of XgBosst Model is",np.sqrt(mean_squared_error(Y_train,Y_xgb_train)))
r2_score=lr_model.score(X_train,Y_train)

print("R2_score of XgBosst Model is ",r2_score)
#Submission

sub_df=pd.DataFrame(data={'Id':test_df1['Id'].values,'SalePrice':np.exp(Y_pred_xgb)})

sub_df.to_csv('submission.csv',index=False)