import os

import sys

import time 



from copy import deepcopy



import numpy as np

import pandas as pd

import statsmodels.api as sm





import matplotlib.pyplot as plt

import seaborn as sns; sns.set()





import warnings

warnings.filterwarnings("ignore") # warnings.filterwarnings("ignore", category=DeprecationWarning) # 



# utilities 

from sklearn.utils import shuffle



# metrics 

from sklearn.metrics import ( mean_squared_error, r2_score, mean_squared_log_error

                            , roc_curve, confusion_matrix

                            )



# pre-processing 

from sklearn.impute import ( SimpleImputer, MissingIndicator )  # IterativeImputer

from sklearn.preprocessing import (  PolynomialFeatures, LabelEncoder, OneHotEncoder

                                    , StandardScaler , MinMaxScaler, RobustScaler

                                  )

# Dimensionality reduction

from sklearn.decomposition import KernelPCA, PCA 

from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold





# Regression solvers  

from sklearn.linear_model import ( LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor 

                                   , RANSACRegressor, HuberRegressor, BayesianRidge

                                  )



from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC

from sklearn.neighbors    import KNeighborsRegressor

from sklearn.ensemble     import ( RandomForestRegressor, BaggingRegressor, AdaBoostRegressor

                                  , GradientBoostingRegressor, ExtraTreesRegressor 

                                 )

import lightgbm as lgb

from xgboost import XGBRegressor, plot_importance 





# model selection and evaluation 

from sklearn.model_selection import  ( train_test_split, KFold, cross_val_score

                                      , learning_curve, validation_curve

                                      , GridSearchCV , RandomizedSearchCV

                                     )



# ensemble 

from sklearn.ensemble import RandomForestRegressor



# pipelines

from sklearn.pipeline import Pipeline 





# other experimental libraries 

from tpot import TPOTRegressor

import featuretools as ft # importing featuretools for auto feature engineering 

from featuretools import variable_types as vtypes # importing vtypes to classify or categoricals



################################################################################################



# set pyplot parameters to make things pretty



plt.rc('axes', linewidth = 1.5)

plt.rc('xtick', labelsize = 14)

plt.rc('ytick', labelsize = 14)

plt.rc('xtick.major', size = 3, width = 1.5)

plt.rc('ytick.major', size = 3, width = 1.5)

# input files 



import os

print(os.listdir("../input"))



# observation data 

housePriceData = pd.read_csv('../input/train.csv')

housePriceData.head()
# observations data that we will train/fit our model on. 



obs_TrainData = housePriceData[['OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'OpenPorchSF', 'SalePrice']]

obs_TrainData.isnull().any()
obs_final = obs_TrainData



X_train_final = obs_final[['OverallQual','LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'OpenPorchSF']]

y_train = obs_final[['SalePrice']].values.reshape(-1,1)


# Building a model chain and  Model pipeline. 



reg_model_pipeline = []

reg_model_pipeline = [   

                         ('linear_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values= np.nan, strategy = 'median')),

                                                             ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                             ("scaler" , StandardScaler() ),

                                                             ("lin_reg", LinearRegression() )

                                                        ])

                         ),   

                         ('ridge_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                            ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                            ("scaler" , StandardScaler() ),

                                                            ("Ridge_reg",Ridge(alpha = 10) )

                                                        ])

                         ),

                         ('lasso_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                            ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                            ("scaler" , StandardScaler() ),

                                                            ("lasso_reg",Lasso(alpha = 10) )

                                                        ])

                         ),

                         ('ElasticNet_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                                 ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                                 ("scaler" , StandardScaler() ),

                                                                 ("ElasticNet_reg", ElasticNet(alpha = 0.1, l1_ratio = 0.5) )

                                                             ])

                         ),

                         ('SGD_normal_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                                 ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                                 ("scaler" , StandardScaler() ),

                                                                 ("SGD_normal_reg",SGDRegressor(loss='squared_loss', penalty= None, alpha=0.0)   )

                                                            ])

                         ),

                         ('SGD_lasso_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                                ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                                ("scaler" , StandardScaler() ),

                                                                ("SGD_lasso_reg", SGDRegressor(loss='squared_loss', penalty='l1', alpha=10)  )

                                                        ])

                         ),

                         ('SGD_ridge_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                                ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                                ("scaler" , StandardScaler() ),

                                                                ("SGD_ridge_reg", SGDRegressor(loss='squared_loss', penalty='l2', alpha=10))  

                                                           ])

                         ),

                         ('SGD_elasticNet_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                                     ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                                ("scaler" , StandardScaler() ),

                                                                ("SGD_elasticnet_reg",  SGDRegressor(loss='squared_loss', penalty='elasticnet', l1_ratio = 0.5 , alpha=0.1))   

                                                           ])

                         ),

                         ('KNN_regressor' ,  Pipeline([  ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                         ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                         ("scaler" , StandardScaler() ),

                                                         ("KNN_reg", KNeighborsRegressor (n_neighbors = 10 , weights = 'distance', algorithm = 'auto' ) )   

                                                      ])

                         ),

                            # SGD Normal with Cross Validation - Early Stopping

                         ('SGD_CV_regressor' ,  Pipeline([  ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                            ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                         ("scaler" , StandardScaler() ),

                                                         ("SGD_CV_reg", SGDRegressor(loss='squared_loss', penalty=None, alpha = 0.001 , shuffle = True

                                                                                 , eta0=0.00001, learning_rate = 'adaptive'

                                                                                 , early_stopping = True, validation_fraction= 0.2,  n_iter_no_change = 500, max_iter=100000

                                                                                 )

                                                         )   

                                                        ])

                         ),                         

                         ('GB_regressor' ,  Pipeline([   ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                                                         ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                                                         ("scaler" , StandardScaler() ),

                                                         ("GB_reg", GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1) )   

                                                      ])

                         )

                         # LGBM - takes too long to run. 

                         #,('LGBM_regressor' ,  Pipeline([  ('imp'    , SimpleImputer(missing_values=np.nan, strategy = 'median')),

                         #                                  ("PolyFt" , PolynomialFeatures(degree = 4, include_bias = False) ),

                         #                                ("scaler" , StandardScaler() ),

                         #                                ("LGBM_reg", lgb.LGBMRegressor( objective='regression', boosting_type = 'gbdt',num_leaves=60, learning_rate=0.01, n_estimators=10000) )   

                         #                             ])

                         #)   

                     ]



# runnning the mdoel chain without CV ( cross-validation)



regression_scores = pd.DataFrame()  # empty dataframe to hold regression error metrics 



print("\n\n  For multiple models(model chain) with  (transformers + estimator) without CV ... ")



for name, model in reg_model_pipeline :

    model.fit(X_train_final , y_train)

    y_pred_model_pl = model.predict(X_train_final)  # predict on the train set itself. 

    regression_scores.loc[name, 'RMSE'] =  np.sqrt( mean_squared_error(y_train, y_pred_model_pl )  )

    regression_scores.loc[name, 'R2'] = r2_score(y_train, y_pred_model_pl )

    regression_scores 

    

regression_scores    
# Running the model chain using KFold Cross-validation using 'r2' as the scoring method.



splits = 5   # 7, 10 

scoring = 'r2'   # 



regression_scores = pd.DataFrame(columns = [i for i in range(1,splits+1)] )  # empty datafram to hold regression error metrics 



# In the model chain, for each individual model pipeline, run KFold validation and print regression scores.  

for name, model in reg_model_pipeline :

    kfold = KFold(n_splits= splits , shuffle = True, random_state=42)  # what would happen if i place this before the loop ? 

    cv_results = cross_val_score(estimator= model, X = X_train_final , y = y_train, cv = kfold , scoring = 'r2')

    regression_scores.loc[name , : ] = cv_results

    

regression_scores.loc[:, 'mean'] = regression_scores.mean(axis = 1)

regression_scores    





# KNN , GB , ElasticNet, Ridge, Lasso : producing okay/acceptable results ( in that order) except for one split in the data . 

# but one thing in common is that they are  ALL performing badly on a particular split of the data. what does this data contain ? 

# Running the model chain using KFold Cross-validation using 'r2' and 'neg_mean_squared_error' errors-metrics ( seperately ).



splits = 5   # 7, 10 

scoring = 'r2'   # 



regression_scores = pd.DataFrame(columns = [i for i in range(1,splits+1)] )  # empty datafram to hold regression error metrics 



regression_scores_r2 = regression_scores

regression_scores_mse = regression_scores





# In the model chain, for each individual model pipeline, run KFold validation and print regression scores.  

for name, model in reg_model_pipeline :

    kfold = KFold(n_splits= splits , shuffle = True, random_state=42)

    

    cv_results_r2 = cross_val_score(estimator= model, X = X_train_final , y = y_train, cv = kfold , scoring = 'r2')

    regression_scores_r2.loc[name , : ] = cv_results_r2



    cv_results_mse = cross_val_score(estimator= model, X = X_train_final , y = y_train, cv = kfold , scoring = 'neg_mean_squared_error')

    regression_scores_mse.loc[name , : ] = cv_results_mse

    

    

    

regression_scores_r2.loc[:, 'mean'] = regression_scores.mean(axis = 1)

regression_scores_mse.loc[:, 'mean'] = regression_scores.mean(axis = 1)

 
regression_scores_r2
regression_scores_mse
elasticNet_pipeline = Pipeline([   ('imp'    , SimpleImputer(missing_values= np.nan,strategy = 'median')),

                                   ("PolyFt" , PolynomialFeatures (include_bias = False) ),

                                   ("scaler" , StandardScaler() ),

                                   ("ElasticNet_reg", ElasticNet() )

                              ])



parameters = {}   # a dictionary 



parameters['imp__strategy'] = ['mean', 'median', 'most_frequent']

parameters['PolyFt__degree'] = [3,4,5,6]

parameters['ElasticNet_reg__alpha'] = [0.1, 0.5, 1.0 , 10 ]

parameters['ElasticNet_reg__l1_ratio'] = [0.1, 0.5, 1.0 , 10]



splits = 5   # 7, 10 

score = 'neg_mean_squared_error'  #  'r2' 



kfold = KFold(n_splits= splits , shuffle = True, random_state=42)



gridsearch = GridSearchCV(estimator = elasticNet_pipeline , param_grid = parameters 

                          , cv = kfold , scoring = 'r2' , n_jobs= 1

                         )

gridsearch.fit(X = X_train_final , y = y_train) 



print('Best score and parameter combination : ')



print(gridsearch.best_score_)    

print(gridsearch.best_params_) 
pd.DataFrame(gridsearch.cv_results_).sort_values('rank_test_score', ascending = True)



# there is one split that produces a very deviant score. find out what is the data in that split and why the deviant score ?  

# might help to find out the outlier data. 