import os

import warnings  

warnings.filterwarnings('ignore')

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew 

from sklearn import metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error     

from sklearn.metrics import r2_score







dataset = pd.read_csv("../input/diamonds.csv")

dataset.describe()
dataset = dataset.drop('Unnamed: 0', axis=1)
dataset[['x','y','z']] = dataset[['x','y','z']].replace(0,np.NaN)

dataset.isnull().sum()

dataset.dropna(inplace=True)







dataset['volume'] = dataset['x']*dataset['y']*dataset['z']   

dataset.drop(['x','y','z'], axis=1, inplace= True)

sns.distplot(dataset["price"] , fit = norm)
sns.boxplot(x = dataset['price'])
sns.distplot(dataset["carat"] , fit = norm)
sns.scatterplot( x = dataset['carat'] , y = dataset['price'])
sns.boxplot( x = dataset['carat'])

dataset = dataset.drop(dataset[(dataset['carat']>1.99)].index)



sns.boxplot( x = dataset['carat'])

sns.distplot(dataset["carat"] , fit = norm)
sns.distplot(dataset["volume"] , fit = norm)
sns.scatterplot(x = dataset['volume'] , y = dataset['price'])

sns.boxplot(dataset['volume'])
dataset = dataset.drop(dataset[(dataset['volume'] > 299)].index)



sns.boxplot(dataset['volume'])

sns.countplot( x = dataset['cut'])

sns.countplot( x = dataset['color'])
sns.countplot( x = dataset['clarity'])

sns.boxplot(x='clarity', y='price', data=dataset ) 

dataset = pd.get_dummies(dataset , drop_first = True)
y = dataset['price'].values



X = dataset.drop(['price'], axis=1)     







from sklearn.preprocessing import RobustScaler 

rb = RobustScaler()

X_scaled = rb.fit_transform(X)



X_scaled = pd.DataFrame(X_scaled, columns = X.columns)  #--> rename columns after scaling

X = X_scaled



plt.figure(figsize=(15,15))

plt.title('Correlation Map')

ax=sns.heatmap(dataset.corr(),

               linewidth=2.1,

               annot=True,

               center=1)
for v in X.columns:

    variance = X.var()

variance = variance.sort_values(ascending = False)

   

plt.figure(figsize=(12,5))

plt.plot(variance)  

variance
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR





regressors = [['Linear Regression :' , LinearRegression()],

       ['Decision Tree Regression :' , DecisionTreeRegressor()],

       ['Random Forest Regression :' , RandomForestRegressor()],

       [' XGB :' , XGBRegressor()] ,

       ['K-Neighbors Regression :', KNeighborsRegressor()],

       ['Support Vector Regression :', SVR()]   

       ]



for name,model in regressors:

        

    model = model      

    

    model.fit(X_train,y_train)

    

    y_pred_train = model.predict(X_train)  

    

    y_pred_valid = model.predict(X_valid)

        



    print('-----------------------------------')

    print(name)

    

    print(' --TRAINING SET --')

    print('MAE:', mean_absolute_error(y_train , y_pred_train))

    print('R2 :', r2_score(y_train , y_pred_train))



    print('-----------------------------------')    

    print(' --VALIDATION SET --')

    print('MAE:', mean_absolute_error(y_valid, y_pred_valid))

    print('R2 :', r2_score(y_valid , y_pred_valid))

    print('---------------------------------')

model = XGBRegressor()

model.fit( X_train , y_train)





importances = model.feature_importances_

index = np.argsort(importances)[::-1][0:15]

feature_names = X.columns.values



plt.figure(figsize=(10,5))

sns.barplot(x = feature_names[index], y = importances[index])

plt.title(" XGB - Top important features ")





model = RandomForestRegressor()

model.fit( X_train , y_train)





importances = model.feature_importances_

index = np.argsort(importances)[::-1][0:15]

feature_names = X.columns.values



plt.figure(figsize=(10,5))

sns.barplot(x = feature_names[index], y = importances[index])

plt.title(" Random Forest - Top important features ")



from sklearn.model_selection import RandomizedSearchCV





colsample_bylevel = [1 , 0.5]

colsample_bytree = [1 , 0.5]

gamma = [0 , 1 , 5]

learning_rate = [  0.01 , 0.0125 , 0.001] 

max_depth = [ 1 , 5 , 10 ]

min_child_weight = [1]

n_estimators = [ 250 , 500 , 750 , 1000]   

random_state = [42]     

reg_alpha = [0, 1]

reg_lambda = [0 , 1]

scale_pos_weight = [1]

subsample = [0.5, 0.8 ,  1 ]





param_distributions = dict(

                           colsample_bylevel = colsample_bylevel,

                           colsample_bytree = colsample_bytree,

                           gamma = gamma, 

                           learning_rate = learning_rate,

                           max_depth = max_depth,

                           min_child_weight = min_child_weight,

                           n_estimators = n_estimators,

                           random_state = random_state,

                           reg_alpha = reg_alpha,

                           reg_lambda = reg_lambda,

                           scale_pos_weight = scale_pos_weight,

                           subsample = subsample , 

                           

                           ) 







estimator = XGBRegressor()     





RandomCV = RandomizedSearchCV(

                            estimator = estimator,         

                            param_distributions = param_distributions,

                            n_iter = 10,

                            cv = 5,

                            scoring = "neg_mean_absolute_error" ,  #'r2', 

                            random_state = 42, 

                            verbose = 1, 

                            n_jobs = -1,

                            )







hyper_model = RandomCV.fit(X_train, y_train)      
           

print('Best Score: ', hyper_model.best_score_)    



print('Best Params: ', hyper_model.best_params_)





hyper_model.best_estimator_.fit(X_train , y_train)




y_pred_train_hyper = hyper_model.predict(X_train)  



y_pred_valid_hyper = hyper_model.predict(X_valid)  







print(' -- HYPER TRAIN --')

print('MAE:', mean_absolute_error ( y_train , y_pred_train_hyper))

print('R2 :', r2_score ( y_train , y_pred_train_hyper))





print('-----------------------------------')    

print(' -- HYPER VALIDATION  --')

print('MAE:', mean_absolute_error(y_valid, y_pred_valid_hyper))

print('R2 :', r2_score(y_valid , y_pred_valid_hyper))

print('---------------------------------')

y_pred_hyper = hyper_model.predict(X_test)  





print('-----------------------------------')    

print(' -- HYPER TEST --')

print('MAE:', mean_absolute_error(y_test, y_pred_hyper))

print('R2 :', r2_score(y_test , y_pred_hyper))

print('---------------------------------')
