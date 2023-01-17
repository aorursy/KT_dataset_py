# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from xgboost import plot_importance

import seaborn as sns

import os
#Data pre-processing



from sklearn.preprocessing import StandardScaler as ss
#Dimensionality reduction

from sklearn.decomposition import PCA
#Data splitting and model parameter search

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



#Modeling modules

from xgboost.sklearn import XGBClassifier



# Model pipelining

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline



# Model evaluation metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix
#Bayes optimization

from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization



import eli5

from eli5.sklearn import PermutationImportance



#Misc

import time

import os



import random

from scipy.stats import uniform
pd.set_option('display.max_columns', 100)



# Set working directory

os.chdir("../input")

# Reading data and plotting graphs

data = pd.read_csv("winequalityN.csv")

data.head()



data.info()



data.shape

data.isnull().any()



data.dropna(axis=0,inplace=True)

data.isnull().any()



g = sns.pairplot(data,palette="husl",diag_kind="kde",hue='type')



plt.figure(figsize=(12, 12))

sns.heatmap(data.corr(),annot=True,vmin=-1,cmap='YlGnBu')



sns.barplot(x=data.type, y=data.alcohol,data=data, hue=data.quality,palette='spring')
#setting the predictors

X = data.iloc[ :, 1:14]

X.head(2)
#First column as target

y = data.iloc[ : , 0]

y.head()



y = y.map({'white':1, 'red' : 0})

y.dtype



#Divide dataset into Training data and validation data

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.35,

                                                    shuffle = True

                                                    )



xgb_list = [('ss',ss()),         #Scaling parameters

            ('pca',PCA()),       #Instantiate PCA

            ('xgb',XGBClassifier(silent = False,  #Instantiate XGB classifier with 2 cpu threads 

                                  n_jobs=2))]

            

xgb_pipeline = Pipeline(xgb_list)
#Parameter for grid search

parameter_gs = {'xgb__learning_rate':  [0.03, 0.08],

              'xgb__n_estimators':   [100,  160],

              'xgb__max_depth':      [3,5],

              'pca__n_components' : [7,12]

              }
#Grid Search



grid_search = GridSearchCV(xgb_pipeline,           

                   parameter_gs,         

                   n_jobs = 2,         

                   cv =2 ,             

                   verbose =2,      

                   scoring = ['accuracy', 'roc_auc'],  # Performance metrics

                   refit = 'roc_auc'   

                   )
#Fitting data to Pipeline

start = time.time()

grid_search.fit(X_train, y_train)   

end = time.time()

(end - start)/60



f"Best score: {grid_search.best_score_} "

f"Best parameter set {grid_search.best_params_}"



plt.bar(grid_search.best_params_.keys(), grid_search.best_params_.values(), color='g')

plt.xticks(rotation=10)

y_pred = grid_search.predict(X_test)

y_pred



accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
#Random Search



parameter_rs = {'xgb__learning_rate':  uniform(0, 1),

              'xgb__n_estimators':   range(50,100),

              'xgb__max_depth':      range(3,7),

              'pca__n_components' : range(5,7)}

random_search = RandomizedSearchCV(xgb_pipeline,

                        param_distributions=parameter_rs,

                        scoring= ['roc_auc', 'accuracy'],

                        n_iter=15,          

                        verbose = 3,

                        refit = 'roc_auc',

                        n_jobs = 2,         

                        cv = 2               

                        )
#Fitting data to Pipeline

start = time.time()

random_search.fit(X_train, y_train)

end = time.time()

(end - start)/60
f"Best score: {random_search.best_score_} "

f"Best parameter set: {random_search.best_params_}" 

plt.bar(random_search.best_params_.keys(), random_search.best_params_.values(), color='g')

plt.xticks(rotation=10)



y_pred = random_search.predict(X_test)

y_pred



accuracy = accuracy_score(y_test, y_pred)

f"Accuracy: {accuracy * 100.0}"
#Bayesian Optimization



parameter_bo = {

           'learning_rate':  (0, 1),            

           'n_estimators':   (50,100),         

           'max_depth':      (3,7),            

           'n_components' :  (5,7)

            }



def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    #Make Pipeling for BO

    pipe_xg1 = make_pipeline (ss(),

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )

    #Fitting into pipeline 

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X= X_train,

                                y = y_train,

                                cv = 2,

                                n_jobs = 2,

                                scoring = 'f1'

                                ).mean()             # taking mean of all results



    return cv_result       #Returning final mean of all results of cross val score

bayesian_opt = BayesianOptimization(

                             xg_eval,     

                             parameter_bo   

                             )

start = time.time()

bayesian_opt.maximize(init_points=5,

               n_iter=15,        

               )
f"Best parameter set: {bayesian_opt.max} "

bayesian_opt.max.values()



for features in bayesian_opt.max.values(): 

    print(features)



features

plt.bar(features.keys(), features.values(), color='g')

plt.xticks(rotation=10)
#Fitting parameters and Feature Importance

#Model with parameters of grid search

model_gs = XGBClassifier(

                    learning_rate = grid_search.best_params_['xgb__learning_rate'],

                    max_depth = grid_search.best_params_['xgb__max_depth'],

                    n_estimators=grid_search.best_params_['xgb__n_estimators']

                    )



#Model with parameters of random search

model_rs = XGBClassifier(

                    learning_rate = random_search.best_params_['xgb__learning_rate'],

                    max_depth = random_search.best_params_['xgb__max_depth'],

                    n_estimators=random_search.best_params_['xgb__n_estimators']

                    )



#Model with parameters of bayesian optimization

model_bo = XGBClassifier(

                    learning_rate = int(features['learning_rate']),

                    max_depth = int(features['max_depth']),

                    n_estimators=int(features['n_estimators'])

                    )

start = time.time()

model_gs.fit(X_train, y_train)

model_rs.fit(X_train, y_train)

model_bo.fit(X_train, y_train)



y_pred_gs = model_gs.predict(X_test)

y_pred_rs = model_rs.predict(X_test)

y_pred_bo = model_bo.predict(X_test)

accuracy_gs = accuracy_score(y_test, y_pred_gs)

accuracy_rs = accuracy_score(y_test, y_pred_rs)

accuracy_bo = accuracy_score(y_test, y_pred_gs)

print("Grid search Accuracy: "+str(accuracy_gs))

print("Grid search Accuracy: "+str(accuracy_rs))

print("Bayesian Optimization Accuracy: "+str(accuracy_bo))



model_gs.feature_importances_

model_rs.feature_importances_

model_bo.feature_importances_

plot_importance(model_gs)

plot_importance(model_rs)

plot_importance(model_bo)

plt.show()