# data analysis libraries:

from flask import Flask, jsonify, request, render_template

import pandas as pd

import numpy as np

import json

import pickle



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns



# to ignore warnings:

import sys

if not sys.warnoptions:

    import os, warnings

    warnings.simplefilter("ignore") 

    os.environ["PYTHONWARNINGS"] = "ignore" 



# to display all columns:

pd.set_option('display.max_columns', None)



#timer

import time

from contextlib import contextmanager



# Importing modelling libraries

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold

from sklearn.preprocessing import StandardScaler  

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,BaggingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV

pd.options.display.float_format = "{:,.2f}".format



@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} done in {:.0f}s".format(title, time.time() - t0))
# !pip install xgboost

# !pip install lightgbm>=0.90

# !pip install catboost==0.23.1
!pip install catboost==0.23.1
pwd
# Read train and test data with pd.read_csv():

data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
X = data.iloc[:, 3:-1]

y = data.iloc[:, -1]
X
def data_encode(df):

    df = pd.get_dummies(data = df, columns=["Geography"], drop_first = False)

    for col in df.select_dtypes(include=['category','object']).columns:

        codes,_ = df[col].factorize(sort=True)    

        df[col]=codes

    return df
X = data_encode(X)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)

pickle.dump(X.columns, open("columns.pkl", 'wb'))
# Memory Reduction 
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
for df in [x_train,x_val]:

    reduce_mem_usage(df)    
r=1309

models = [LogisticRegression(random_state=r),GaussianNB(), KNeighborsClassifier(),

          SVC(random_state=r,probability=True),BaggingClassifier(random_state=r),DecisionTreeClassifier(random_state=r),

          RandomForestClassifier(random_state=r), GradientBoostingClassifier(random_state=r),

          XGBClassifier(random_state=r), MLPClassifier(random_state=r),

          CatBoostClassifier(random_state=r,verbose = False)]

names = ["LogisticRegression","GaussianNB","KNN","SVC","Bagging",

             "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]
print('Default model validation accuracies for the train data:', end = "\n\n")

for name, model in zip(names, models):

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val) 

    print(name,':',"%.3f" % accuracy_score(y_pred, y_val))
predictors=pd.concat([x_train,x_val])
results = []

print('10 fold Cross validation accuracy and std of the default models for the train data:', end = "\n\n")

for name, model in zip(names, models):

    kfold = KFold(n_splits=10, random_state=1001)

    cv_results = cross_val_score(model, predictors, y, cv = kfold, scoring = "accuracy")

    results.append(cv_results)

    print("{}: {} ({})".format(name, "%.3f" % cv_results.mean() ,"%.3f" %  cv_results.std()))
# Possible hyper parameters

logreg_params= {"C":np.logspace(-1, 1, 10),

                    "penalty": ["l1","l2"], "solver":['lbfgs', 'liblinear', 'sag', 'saga'], "max_iter":[1000]}



NB_params = {'var_smoothing': np.logspace(0,-9, num=100)}

knn_params= {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

svc_params= {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1, 5, 10 ,50 ,100],

                 "C": [1,10,50,100,200,300,1000]}

bag_params={"n_estimators":[50,120,300]}

dtree_params = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}

rf_params = {"max_features": ["log2","auto","sqrt"],

                "min_samples_split":[2,3,5],

                "min_samples_leaf":[1,3,5],

                "bootstrap":[True,False],

                "n_estimators":[50,100,150],

                "criterion":["gini","entropy"]}



gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],

             "n_estimators": [100,500,100],

             "max_depth": [3,5,10],

             "min_samples_split": [2,5,10]}



xgb_params ={

        'n_estimators': [50, 100, 200],

        'subsample': [ 0.6, 0.8, 1.0],

        'max_depth': [1,2,3,4],

        'learning_rate': [0.1,0.2, 0.3, 0.4, 0.5],

        "min_samples_split": [1,2,4,6]}



mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],

              "hidden_layer_sizes": [(10,10,10),

                                     (100,100,100),

                                     (100,100),

                                     (3,5), 

                                     (5, 3)],

              "solver" : ["lbfgs","adam","sgd"],"max_iter":[1000]}

catb_params =  {'depth':[2, 3, 4],

              'loss_function': ['Logloss', 'CrossEntropy'],

              'l2_leaf_reg':np.arange(2,31)}

classifier_params = [logreg_params,NB_params,knn_params,svc_params,bag_params,dtree_params,rf_params,

                     gbm_params, xgb_params,mlpc_params,catb_params]               

                  
# Tuning by Cross Validation  

cv_result = {}

best_estimators = {}

for name, model,classifier_param in zip(names, models,classifier_params):

    with timer(">Model tuning"):

        clf = GridSearchCV(model, param_grid=classifier_param, cv =10, scoring = "accuracy", n_jobs = -1,verbose = False)

        clf.fit(x_train,y_train)

        cv_result[name]=clf.best_score_

        best_estimators[name]=clf.best_estimator_

        print(name,'cross validation accuracy : %.3f'%cv_result[name])
accuracies={}

print('Validation accuracies of the tuned models for the train data:', end = "\n\n")

for name, model_tuned in zip(best_estimators.keys(),best_estimators.values()):

    y_pred =  model_tuned.fit(x_train,y_train).predict(x_val)

    accuracy=accuracy_score(y_pred, y_val)

    print(name,':', "%.3f" %accuracy)

    accuracies[name]=accuracy
n=3

accu=sorted(accuracies, reverse=True, key= lambda k:accuracies[k])[:n]

firstn=[[k,v] for k,v in best_estimators.items() if k in accu]

print(firstn)
# Ensembling First n Score



votingC = VotingClassifier(estimators = firstn, voting = "soft", n_jobs = -1)

model= votingC.fit(x_train, y_train)

print("\nAccuracy_score is:",accuracy_score(model.predict(x_val),y_val))
# save the model so created above into a picle.

pickle.dump(model, open('model.pkl', 'wb')) 
print("\nAccuracy_score is:",accuracy_score(model.predict(x_val),y_val))