import numpy as np
import os
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
from pandas import read_csv
#Read the data
filename = '../input/all_stocks_5yr.csv'
stock = read_csv(filename)
print("***Structure of data with all its features***")
stock.head()
ticker_name = 'AAPL'
stock_a = stock[stock['Name'] == ticker_name]
stock_a.shape
stock.info()
stock_a.describe()
stock_a['changeduringday'] = ((stock['high'] - stock['low'] )/ stock['low'])*100

stock_a['changefrompreviousday'] = (abs(stock_a['close'].shift() - stock_a['close'] )/ stock['close'])*100

print("**The new features 'changeduring day & change from previous day are added to the dataset. Note: The first row for change from previous day for each stock is NA or blank always")
stock_a.head()
stock_a.hist(bins=50, figsize=(20,15))
plt.show()
stock_a.plot(kind="line", x="date", y="close", figsize=(15, 10))
corr_matrix = stock_a.corr()
corr_matrix["close"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["high", "low", "open", "changefrompreviousday", "changeduringday", "volume"]

scatter_matrix(stock_a[attributes], figsize=(20, 15))
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
corr = stock_a[["high", "low", "open", "changefrompreviousday", "changeduringday", "volume"]].corr()

# generate a mask for the lower triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 12))

# generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, 
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax);
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer
X_stock_a = stock_a.drop(['date', 'Name','close'], axis=1)
y_stock_a = stock_a['close']

X_stock_train, X_stock_test, y_stock_train, y_stock_test = train_test_split(X_stock_a, y_stock_a, test_size=0.2, 
                                                                            random_state=42)

#Data prep pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer,StandardScaler
data_pipeline = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler())
#        ('normalizer', Normalizer()),
    ])
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,Normalizer

from sklearn.pipeline import Pipeline

Lr_pipeline_nor = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('normalizer',Normalizer()),
        ('lr', LinearRegression())
        
    ])

Lr_pipeline_nor.fit(X_stock_train, y_stock_train)
from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

svr_pipeline_nor = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('normalizer',Normalizer()),
        ('svr', SVR(kernel="linear"))
        
    ])

svr_pipeline_nor.fit(X_stock_train, y_stock_train)

from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

svrrbf_pipeline_nor = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('normalizer',Normalizer()),
        ('svr', SVR(kernel="rbf"))
        
    ])

svrrbf_pipeline_nor.fit(X_stock_train, y_stock_train)
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

tree_reg = DecisionTreeRegressor(random_state=42)
dt_pipeline_nor = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('normalizer',Normalizer()),
        ('dt', DecisionTreeRegressor(random_state=42))
        
    ])

dt_pipeline_nor.fit(X_stock_train, y_stock_train)
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Data prep pipeline

Lr_pipeline_std = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler()),
        ('lr', LinearRegression())
        
    ])

Lr_pipeline_std.fit(X_stock_train, y_stock_train)


from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

svr_pipeline_std = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler()),
        ('svr', SVR(kernel="linear"))
        
    ])

svr_pipeline_std.fit(X_stock_train, y_stock_train)

from sklearn.svm import SVR
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

svrrbf_pipeline_std = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler()),
        ('svrrbf', SVR(kernel="rbf"))
        
    ])

svrrbf_pipeline_std.fit(X_stock_train, y_stock_train)
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

tree_reg = DecisionTreeRegressor(random_state=42)
dt_pipeline_std = Pipeline([
        ('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler()),
        ('dt', DecisionTreeRegressor(random_state=42))
        
    ])

dt_pipeline_std.fit(X_stock_train, y_stock_train)
#Doing Regularization Ridge 

#1. Fine tune Ridge Regression using Random search
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation

pipeline = Pipeline([('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler()),
    ('RidgeReg', Ridge()),
])

param_distribs = {#listed in the form of "pipelineStep__parameter", e.g, "RidgeLogReg__alpha":(100., 10., 1.,0.1)
    "RidgeReg__alpha":(1e2, 1e1, 1., 1e-1, 1e-2, 1e-3, 1e-4),
    "RidgeReg__fit_intercept":(True,False)
}


rnd_search_ridge_cv=  RandomizedSearchCV(pipeline, param_distribs,
                                 cv=10, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=-1)

rnd_search_ridge_cv.fit(X_stock_train, y_stock_train)


#Mean CV scores for Ridge
ridgescores = rnd_search_ridge_cv.cv_results_
for mean_score, params in zip(ridgescores["mean_test_score"], ridgescores["params"]):
    print(np.sqrt(-mean_score), params)
print ('Best CV score for Ridge: ', -rnd_search_ridge_cv.best_score_)
from sklearn.metrics import mean_squared_error

rnd_search_stock_predictions = rnd_search_ridge_cv.best_estimator_.predict(X_stock_test)
rnd_search_mse = mean_squared_error(y_stock_test, rnd_search_stock_predictions)
rnd_search_rmse = np.sqrt(rnd_search_mse)
print('Ridge regression best estimator RMSE with Standardization', rnd_search_rmse)
#Lasso
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation

pipeline = Pipeline([('imputer', Imputer(missing_values="NaN",strategy="median")), #Use the "median" to impute missing vlaues
        ('scaler',StandardScaler()),
    ('LassoReg', Lasso()),
])

param_distribs = {
    "LassoReg__alpha":(1e2, 1e1, 1., 1e-1, 1e-2, 1e-3, 1e-4),
    "LassoReg__fit_intercept":(True,False)
}


rnd_search_lasso_cv = RandomizedSearchCV(pipeline, param_distribs,
                                 cv=10, n_iter=10,scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=-1,random_state=42)

rnd_search_lasso_cv.fit(X_stock_train, y_stock_train)


#Mean CV scores for Lasso
Lasscores = rnd_search_lasso_cv.cv_results_
for mean_score, params in zip(Lasscores["mean_test_score"], Lasscores["params"]):
    print(np.sqrt(-mean_score), params)
print ('Best CV score for Lasso: ', -rnd_search_lasso_cv.best_score_)
rnd_search_lasso_stock_predictions = rnd_search_lasso_cv.best_estimator_.predict(X_stock_test)
rnd_search_lasso_mse = mean_squared_error(y_stock_test, rnd_search_lasso_stock_predictions)
rnd_search_lasso_rmse = np.sqrt(rnd_search_lasso_mse)
print('Lasso regression best estimator RMSE with Standardization', rnd_search_lasso_rmse)


#1. Fine tune Linear Regression using Random search
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation

lin_reg = LinearRegression()



X_stock_train = data_pipeline.fit_transform(X_stock_train)

scores = cross_validation.cross_val_score(lin_reg, X_stock_train, y_stock_train, scoring='neg_mean_squared_error', cv=10,)


#Metrics - Mean CV scores for Linear Regression
print (-scores)
print ('Mean score for Linear Regression: ', np.mean(-scores))
#2. Fine tune Decision Tree Regressor using Random search
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
        'max_depth': [1,2,3,4,5,6,7,8,9,10],
    }

tree_reg = DecisionTreeRegressor(random_state=42)
rnd_search_tree = RandomizedSearchCV(tree_reg, param_distributions=param_distribs,
                                n_iter=10, cv=10, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=-1, random_state=42)

X_stock_fe_prep = data_pipeline.transform(X_stock_train)
rnd_search_tree.fit(X_stock_fe_prep, y_stock_train)

#Metrics - Mean CV scores for Decistion Tree
cvres2 = rnd_search_tree.cv_results_
for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
    print(np.sqrt(-mean_score), params)
print ('Best CV score for Decision Tree: ', -rnd_search_tree.best_score_)
#3. Fine tune SVM regression using Random search


from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal


# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 1000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search_svr = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=10, cv=2, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=-1, random_state=42)
X_stock_fe_prep = data_pipeline.fit_transform(X_stock_train[0:500])
rnd_search_svr.fit(X_stock_fe_prep, y_stock_train[0:500])


#Mean CV scores for SVR
cvres3 = rnd_search_svr.cv_results_
for mean_score, params in zip(cvres3["mean_test_score"], cvres3["params"]):
    print(np.sqrt(-mean_score), params)
print ('Best CV score for SVR: ', -rnd_search_svr.best_score_)
#Lasso Regression
lasso_stock_predictions_std = rnd_search_lasso_cv.best_estimator_.predict(X_stock_test)
lasso_mse_std = mean_squared_error(y_stock_test, lasso_stock_predictions_std)
lasso_rmse_std = np.sqrt(lasso_mse_std)
print('Lasso Regression RMSE with Standardization', lasso_rmse_std)

#Ridge Regression
ridge_stock_predictions_std = rnd_search_ridge_cv.best_estimator_.predict(X_stock_test)
ridge_mse_std = mean_squared_error(y_stock_test, ridge_stock_predictions_std)
ridge_rmse_std = np.sqrt(ridge_mse_std)
print('Ridge Regression RMSE with Standardization', ridge_rmse_std)


from sklearn.metrics import mean_absolute_error

#Linear Regression with normalisation and standardisation
lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
lr_mae_nor = mean_absolute_error(y_stock_test, lr_stock_predictions_nor)
print('Lr MAE with Normalization', lr_mae_nor)

lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
lr_mae_std = mean_absolute_error(y_stock_test, lr_stock_predictions_std)
print('Lr MAE with standardization', lr_mae_std)

#SVM with normalisation and standardisation
svm_stock_predictions_nor = svr_pipeline_nor.predict(X_stock_test)
svm_mae_nor = mean_absolute_error(y_stock_test, svm_stock_predictions_nor)
print('SVM MAE with Normalization', svm_mae_nor)

svm_stock_predictions_std = svr_pipeline_std.predict(X_stock_test)
svm_mae_std = mean_absolute_error(y_stock_test, svm_stock_predictions_std)
print('SVM MAE with standardization', svm_mae_std)


#SVM with RFB Kernel with normalisation and standardisation
svmrbf_stock_predictions_nor = svrrbf_pipeline_nor.predict(X_stock_test)
svmrbf_mae_nor = mean_absolute_error(y_stock_test, svmrbf_stock_predictions_nor)
print('SVM RBF MAE with Normalization', svmrbf_mae_nor)


svmrbf_stock_predictions_std = svrrbf_pipeline_std.predict(X_stock_test)
svmrbf_mae_std = mean_absolute_error(y_stock_test, svmrbf_stock_predictions_std)
print('SVM RBF MAE with standardization', svmrbf_mae_std)

#Decision Tree with normalisation and standardisation
dt_stock_predictions_nor = dt_pipeline_nor.predict(X_stock_test)
dt_mae_nor = mean_absolute_error(y_stock_test, dt_stock_predictions_nor)
print('DecisionTree MAE with Normalization', dt_mae_nor)

dt_stock_predictions_std = dt_pipeline_std.predict(X_stock_test)
dt_mae_std = mean_absolute_error(y_stock_test, dt_stock_predictions_std)
print('DecisionTree MAE with standardization', dt_mae_std)


#Lasso Regression
lasso_stock_predictions_std = rnd_search_lasso_cv.best_estimator_.predict(X_stock_test)
lasso_mae_std = mean_absolute_error(y_stock_test, lasso_stock_predictions_std)
print('Lasso Regression MAE with Standardization', lasso_mae_std)

#Ridge Regression
ridge_stock_predictions_std = rnd_search_ridge_cv.best_estimator_.predict(X_stock_test)
ridge_mae_std = mean_absolute_error(y_stock_test, ridge_stock_predictions_std)
print('Ridge Regression MAE with Standardization', ridge_mae_std)


import pandas as pd
import numpy as np

#Predict and report RMSE
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error

#Linear Regression with normalisation and standardisation
lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
lr_mse_nor = mean_squared_error(y_stock_test, lr_stock_predictions_nor)
lr_rmse_nor = np.sqrt(lr_mse_nor)
print('Lr RMSE with Normalization', lr_rmse_nor)

lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
lr_mse_std = mean_squared_error(y_stock_test, lr_stock_predictions_std)
lr_rmse_std = np.sqrt(lr_mse_std)
print('Lr RMSE with standardization', lr_rmse_std)

#SVM with normalisation and standardisation
svm_stock_predictions_nor = svr_pipeline_nor.predict(X_stock_test)
svm_mse_nor = mean_squared_error(y_stock_test, svm_stock_predictions_nor)
svm_rmse_nor = np.sqrt(svm_mse_nor)
print('SVM RMSE with Normalization', svm_rmse_nor)

svm_stock_predictions_std = svr_pipeline_std.predict(X_stock_test)
svm_mse_std = mean_squared_error(y_stock_test, svm_stock_predictions_std)
svm_rmse_std = np.sqrt(svm_mse_std)
print('SVM RMSE with standardization', svm_rmse_std)


#SVM with RFB Kernel with normalisation and standardisation
svmrbf_stock_predictions_nor = svrrbf_pipeline_nor.predict(X_stock_test)
svmrbf_mse_nor = mean_squared_error(y_stock_test, svmrbf_stock_predictions_nor)
svmrbf_rmse_nor = np.sqrt(svmrbf_mse_nor)
print('SVM RBF RMSE with Normalization', svmrbf_rmse_nor)


svmrbf_stock_predictions_std = svrrbf_pipeline_std.predict(X_stock_test)
svmrbf_mse_std = mean_squared_error(y_stock_test, svmrbf_stock_predictions_std)
svmrbf_rmse_std = np.sqrt(svmrbf_mse_std)
print('SVM RBF RMSE with standardization', svmrbf_rmse_std)

#Decision Tree with normalisation and standardisation
dt_stock_predictions_nor = dt_pipeline_nor.predict(X_stock_test)
dt_mse_nor = mean_squared_error(y_stock_test, dt_stock_predictions_nor)
dt_rmse_nor = np.sqrt(dt_mse_nor)
print('DecisionTree RMSE with Normalization', dt_rmse_nor)

dt_stock_predictions_std = dt_pipeline_std.predict(X_stock_test)
dt_mse_std = mean_squared_error(y_stock_test, dt_stock_predictions_std)
dt_rmse_std = np.sqrt(dt_mse_std)
print('DecisionTree RMSE with standardization', dt_rmse_std)


#Lasso Regression
lasso_stock_predictions_std = rnd_search_lasso_cv.best_estimator_.predict(X_stock_test)
lasso_mse_std = mean_squared_error(y_stock_test, lasso_stock_predictions_std)
lasso_rmse_std = np.sqrt(lasso_mse_std)
print('Lasso Regression RMSE with Standardization', lasso_rmse_std)

#Ridge Regression
ridge_stock_predictions_std = rnd_search_ridge_cv.best_estimator_.predict(X_stock_test)
ridge_mse_std = mean_squared_error(y_stock_test, ridge_stock_predictions_std)
ridge_rmse_std = np.sqrt(ridge_mse_std)
print('Ridge Regression RMSE with Standardization', ridge_rmse_std)

lr_std = ['1',"Linear Regression with standardisation",np.round(lr_rmse_std,3),np.round(lr_mae_std,3)]
lr_nor = ['2',"Linear Regression with normalisation",np.round(lr_rmse_nor,3),np.round(lr_mae_nor,3)]
dt_std = ['3',"Decision Tree with standardisation",np.round(dt_rmse_std,3),np.round(dt_mae_std,3)]
dt_nor = ['4',"Decision Tree with normalisation",np.round(dt_rmse_nor,3),np.round(dt_mae_nor,3)]

svm_std = ['5',"SVM with standardisation",np.round(svm_rmse_std,3),np.round(svm_mae_std,3)]
svm_nor = ['6',"SVM with normalisation",np.round(svm_rmse_nor,3),np.round(svm_mae_nor,3)]

svmrfb_std = ['7',"SVM RFB with standardisation",np.round(svmrbf_rmse_std,3),np.round(svmrbf_mae_std,3)]
svmrfb_nor = ['8',"SVM RFB with normalisation",np.round(svmrbf_rmse_nor,3),np.round(svmrbf_mae_nor,3)]
ridge_std = ['9',"Ridge Regression with standardisation",np.round(ridge_rmse_std,3),np.round(ridge_mae_std,3)]
lasso_std = ['10',"Lasso Regression with standardisation",np.round(lasso_rmse_std,3),np.round(lasso_mae_std,3)]


linear_model_result= pd.DataFrame([lr_std,lr_nor,dt_std,dt_nor,svm_std,svm_nor,svmrfb_std,svmrfb_nor,ridge_std,lasso_std],columns=[ "ExpID", "Model", "RMSE","MAE"])

linear_model_result
#function to return all the models for a given ticker
from sklearn.preprocessing import Imputer
    
def allModelsResultForAllStocks():
    
    best_result_per_ticker = pd.DataFrame(columns=['Ticker','Model','RMSE'])
    ticker_list = np.unique(stock["Name"])
    best_result_per_ticker = list()
    for ticker_name in ticker_list:
        result = pd.DataFrame(columns=['Ticker','Model','RMSE'])
        stock_a = stock[stock['Name'] == ticker_name]
        #Adding new features 
        #1 Price movement during day time 
        stock_a['changeduringday'] = ((stock['high'] - stock['low'] )/ stock['low'])*100

        #2 Price movement 
        stock_a['changefrompreviousday'] = (abs(stock_a['close'].shift() - stock_a['close'] )/ stock['close'])*100

        X_stock_a = stock_a.drop(['date', 'Name','close'], axis=1)
        y_stock_a = stock_a['close']

        
        imputer = Imputer(missing_values='NaN', strategy='median')
        
        imputer.fit_transform(X_stock_a)
       
        X_stock_train, X_stock_test, y_stock_train, y_stock_test = train_test_split(X_stock_a, y_stock_a, test_size=0.2, 
                                                                                random_state=42)


        Lr_pipeline_std.fit(X_stock_train, y_stock_train)
        Lr_pipeline_nor.fit(X_stock_train, y_stock_train)

        svr_pipeline_nor.fit(X_stock_train, y_stock_train)
        svr_pipeline_std.fit(X_stock_train, y_stock_train)

        svrrbf_pipeline_nor.fit(X_stock_train, y_stock_train)
        svrrbf_pipeline_std.fit(X_stock_train, y_stock_train)


        dt_pipeline_nor.fit(X_stock_train, y_stock_train)
        dt_pipeline_std.fit(X_stock_train, y_stock_train)    
   
        # Predict & Calculate RMSE for all the models 

        #Linear Regression with normalisation and standardisation
        lr_stock_predictions_nor = Lr_pipeline_nor.predict(X_stock_test)
        lr_mse_nor = mean_squared_error(y_stock_test, lr_stock_predictions_nor)
        lr_rmse_nor = np.sqrt(lr_mse_nor)
        rmse_row =   [ticker_name,'Lr RMSE with Normalization', lr_rmse_nor]

        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index
     
    
        lr_stock_predictions_std = Lr_pipeline_std.predict(X_stock_test)
        lr_mse_std = mean_squared_error(y_stock_test, lr_stock_predictions_std)
        lr_rmse_std = np.sqrt(lr_mse_std)
        rmse_row =   [ticker_name,'Lr RMSE with standardization', lr_rmse_std]
    
    

        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index

        #SVM with normalisation and standardisation
        svm_stock_predictions_nor = svr_pipeline_nor.predict(X_stock_test)
        svm_mse_nor = mean_squared_error(y_stock_test, svm_stock_predictions_nor)
        svm_rmse_nor = np.sqrt(svm_mse_nor)
        rmse_row =   [ticker_name,'SVM RMSE with Normalization', svm_rmse_nor]
        

        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index

        svm_stock_predictions_std = svr_pipeline_std.predict(X_stock_test)
        svm_mse_std = mean_squared_error(y_stock_test, svm_stock_predictions_std)
        svm_rmse_std = np.sqrt(svm_mse_std)
        rmse_row =   [ticker_name,'SVM RMSE with standardization', svm_rmse_std]
    
        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index


        #SVM with RFB Kernel with normalisation and standardisation
        svmrbf_stock_predictions_nor = svrrbf_pipeline_nor.predict(X_stock_test)
        svmrbf_mse_nor = mean_squared_error(y_stock_test, svmrbf_stock_predictions_nor)
        svmrbf_rmse_nor = np.sqrt(svmrbf_mse_nor)
        rmse_row =   [ticker_name,'SVM RBF RMSE with Normalization', svmrbf_rmse_nor]
   
        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index


        svmrbf_stock_predictions_std = svrrbf_pipeline_std.predict(X_stock_test)
        svmrbf_mse_std = mean_squared_error(y_stock_test, svmrbf_stock_predictions_std)
        svmrbf_rmse_std = np.sqrt(svmrbf_mse_std)
        rmse_row =   [ticker_name,'SVM RBF RMSE with standardization', svmrbf_rmse_std]
    
        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index

        #Decision Tree with normalisation and standardisation
        dt_stock_predictions_nor = dt_pipeline_nor.predict(X_stock_test)
        dt_mse_nor = mean_squared_error(y_stock_test, dt_stock_predictions_nor)
        dt_rmse_nor = np.sqrt(dt_mse_nor)
        rmse_row =   [ticker_name,'DecisionTree RMSE with Normalization', dt_rmse_nor]

        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index

        dt_stock_predictions_std = dt_pipeline_std.predict(X_stock_test)
        dt_mse_std = mean_squared_error(y_stock_test, dt_stock_predictions_std)
        dt_rmse_std = np.sqrt(dt_mse_std)
        rmse_row = [ticker_name,'DecisionTree RMSE with standardization', dt_rmse_std]
 
        result.loc[-1] = rmse_row  # adding a row
        result.index = result.index + 1  # shifting index
        result = result.sort_values(by = ['RMSE'])
        
       
        best_result_per_ticker.append(np.array(result.iloc[0, :]))
       


    best_result_per_ticker_df = pd.DataFrame(data=best_result_per_ticker, columns=['Ticker','Model','RMSE'])
    
    
    return best_result_per_ticker_df

best_result_per_ticker = allModelsResultForAllStocks()
#Statistics Significance test  

from sklearn.model_selection import cross_val_score
from scipy import stats

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# A sampling based bakeoff using K-fold cross-validation: 
# it randomly splits the training set into K distinct subsets (k=30)
# this bakeoff framework can be used for regression or classification
#Control system is a linear regression based pipeline

kFolds=30

lin_scores = cross_val_score(Lr_pipeline_std, X_stock_train, y_stock_train,
                             scoring="neg_mean_squared_error", cv=kFolds)
control = lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

scores = cross_val_score(svr_pipeline_std, X_stock_train, y_stock_train,
                         scoring="neg_mean_squared_error", cv=kFolds)
treatment = tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

#paired t-test; two-tailed p-value (aka two-sided)
(t_score, p_value) = stats.ttest_rel(control, treatment)
print("The p-value is %0.5f for a t-score of %0.5f." %(p_value, t_score))
#"The p-value is 0.00019 for a t-score of -4.28218." 

if p_value > 0.05/2:  #Two sided 
    print('There is no significant difference between the two machine learning pipelines (Accept H0)')
else:
    print('The two machine learning pipelines are different (reject H0) \n(t_score, p_value) = (%.2f, %.5f)'%(t_score, p_value) )
    if t_score < 0.0:
        print('Machine learning pipeline Linear regression is better than linear SVR pipeline')
    else:
        print('Machine learning pipeline linear SVR pipeline is better than Linear regression')
#Classification function homegrown logic based on stock price mean variation  
def classify (meanValue):
    if meanValue <=1.5:
        return 'Low'
    elif meanValue >1.5 and  meanValue <=2.5:
        return 'Medium'
    elif meanValue >2.5:
        return 'High'
#function to get linear model for given ticker 

def linearModel(ticker):
    stock_a = stock[stock['Name'] == ticker]
    #Adding new features 
    #1 Price movement during day time 
    stock_a['changeduringday'] = ((stock['high'] - stock['low'] )/ stock['low'])*100

    #2 Price movement 
    stock_a['changefrompreviousday'] = (abs(stock_a['close'].shift() - stock_a['close'] )/ stock['close'])*100

    X_stock_a = stock_a.drop(['date', 'Name','close'], axis=1)
    y_stock_a = stock_a['close']

    Lr_pipeline_std.fit(X_stock_a, y_stock_a)
    
    model = Lr_pipeline_std.named_steps['lr']
    
    return model,stock_a

#using all the 500 stocks for training 
ticker_list = np.unique(stock['Name'])

df = pd.DataFrame(columns=['TICKER','CLASS','Coef for open','Coef for high','Coef for low','Coef for volume','Coef for change within day','Coef for change from prev day'])
for ticker in ticker_list:
    
    model,stock_a = linearModel(ticker)    
    
    print("Mean value:",stock_a["changeduringday"].mean())
    #adding target class 
    stock_features = np.concatenate((np.asarray([ticker,classify(stock_a["changeduringday"].mean())]),model.coef_))
    
    df.loc[-1] = stock_features  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index() 
   
#print(df)

#saving feature coefficients and target class for 500 stocks 
df.to_csv('coeff1.csv', mode='a',header=['TICKER','CLASS','Coef for open','Coef for high','Coef for low','Coef for volume','Coef for change within day','Coef for change from prev day'])
    

# loading libraries
import numpy as np
from sklearn.cross_validation import train_test_split

X_class = np.array(df.ix[:, 2:8]) 
y_class = np.array(df['CLASS']) 


# split into train and test
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train_class, y_train_class)

# predict the response
pred = knn.predict(X_test_class)

# evaluate accuracy
print ("Accuracy of KNN ", accuracy_score(y_test_class, pred))
from sklearn.cluster import KMeans

X_class = np.array(df.ix[:, 2:8]) 	# end index is exclusive

k_mean = KMeans()

#number of clusters will be decided by K-mean++ , by default 
k_mean_model = k_mean.fit(X_class)

print("Number of clusters",k_mean_model.n_clusters) 

df_cluster = df.drop(['CLASS'], axis=1)

#Selecting features from dataframe , there are 6 features 
X_cluster = np.array(df_cluster.ix[:, 1:7])

y_pred = k_mean_model.predict(X_cluster)

pred_df = pd.DataFrame({'labels': y_pred, 'companies': df_cluster.ix[:, 0]})


#Cluster assignment for the stocks 
pred_df
#Taking Investors input 

stock_customer = input("Enter the stock ticker customer is interested in buying(we will use clustering) ?")
print(stock_customer)
customer_stock_model,stock_modified = linearModel(stock_customer)
customer_stock_class_pred = knn.predict([customer_stock_model.coef_])


print("Class Prediction for Investor's Stock",customer_stock_class_pred)
customer_stock_model,stock_modified = linearModel(stock_customer)

print(customer_stock_model.coef_)

customer_stock_class_pred = k_mean_model.predict([customer_stock_model.coef_])


print("Cluster for customer Stock:",stock_customer, " is :",customer_stock_class_pred)
