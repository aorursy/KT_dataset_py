import os

import warnings  

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import math 

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score , f1_score , precision_score, recall_score , roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
dataset.info()
dataset.describe()
dataset['sex'] = dataset['sex'].astype('object')

dataset['cp'] = dataset['cp'].astype('object')

dataset['fbs'] = dataset['fbs'].astype('object')

dataset['restecg'] = dataset['restecg'].astype('object')

dataset['exang'] = dataset['exang'].astype('object')

dataset['slope'] = dataset['slope'].astype('object')

dataset['ca'] = dataset['ca'].astype('object')

dataset['thal'] = dataset['thal'].astype('object')
dataset.dtypes
dataset.isnull().sum().sort_values()
sns.set_style('whitegrid')
sns.countplot(dataset["target"])
sns.distplot(dataset['age'].sort_values())
plt.figure(figsize = (12,6))

sns.countplot(dataset['age'].sort_values()  ,  hue = dataset['target'])
sns.countplot('sex' , data  = dataset , hue = 'target')
sns.countplot('cp' , data  = dataset , hue = 'target')
sns.distplot(dataset['trestbps'].sort_values())
sns.boxplot(dataset['trestbps'])
sns.distplot(dataset['chol'].sort_values())
sns.boxplot(dataset['chol'])
sns.countplot('fbs' , data  = dataset , hue = 'target')
sns.countplot('restecg' , data  = dataset , hue = 'target')
sns.distplot(dataset['thalach'].sort_values())

sns.countplot('exang' , data  = dataset , hue = 'target')

sns.distplot(dataset['oldpeak'].sort_values())
sns.boxplot(dataset['oldpeak'])
sns.countplot('slope' , data  = dataset , hue = 'target')
sns.countplot('ca' , data  = dataset , hue = 'target')
sns.countplot('thal' , data  = dataset , hue = 'target')
X = dataset.drop(["target"], axis = 1)



y = dataset["target"].values
X = pd.get_dummies(X)  #drop_first = True?
X.head(10)
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()

X[["age" , "trestbps" , "chol" , "thalach" , "oldpeak"]] = rb.fit_transform(X[["age" , "trestbps" , "chol" , "thalach" , "oldpeak"]])
X.head(10)
for v in X.columns:

    variance = X.var()

  

variance = variance.sort_values(ascending = False)

   

plt.figure(figsize=(12,5))

plt.plot(variance)  
variance
from sklearn.feature_selection import VarianceThreshold



thresholder = VarianceThreshold(threshold=0.01)



X = X.loc[:, thresholder.fit(X).get_support()]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score 

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score





from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

classifiers =  [

       ['Logistic Regression Classifier :', LogisticRegression()] ,

       ['XGB Classifier :', XGBClassifier()] ,

       ['K-Neighbors Classifier :', KNeighborsClassifier()] ,

       ['Support Vector Classifier :', SVC()] ,

       ['Naive Bayes :' , GaussianNB()] ,

       ]



for name,model in classifiers:    



    model = model

    

    model.fit(X_train,y_train)

    

    y_pred_train = model.predict(X_train)



    y_pred = model.predict(X_test)

     

    print('-----------------------------------')

    print(name)

    

    print(" --  TRAIN scores --  ") 

    print('Accuracy: ', accuracy_score( y_train , y_pred_train))

    print("f1: ",f1_score( y_train , y_pred_train))

    print("precision: ", precision_score( y_train , y_pred_train))

    print("recall: ", recall_score( y_train , y_pred_train))

    print("ROC AUC: ", roc_auc_score( y_train , y_pred_train))



    print('---------------------------------')

    

    print(" --  TEST scores --  ")

    print('Accuracy: ', accuracy_score( y_test, y_pred))

    print("f1: ",f1_score( y_test, y_pred))

    print("precision: ", precision_score( y_test, y_pred))

    print("recall: ", recall_score( y_test, y_pred))

    print("ROC AUC: ", roc_auc_score( y_test, y_pred))



    print('---------------------------------')

from xgboost import XGBClassifier



model = XGBClassifier()

model.fit( X_train , y_train )





importances = model.feature_importances_

index = np.argsort(importances)[::-1][0:15]

feature_names = X_train.columns.values



plt.figure(figsize=(10,5))

sns.barplot(x = feature_names[index], y = importances[index]);

plt.title("Top important features ");
from sklearn.feature_selection import SelectFromModel



importances = pd.Series(importances)

importances = importances.sort_values(ascending = False)  



importances.tail(10)
sfm = SelectFromModel(model, threshold=0.001)   



X_train = X_train.loc[ :, sfm.fit(X_train , y_train).get_support()]



X_test = X_test[X_train.columns]
from sklearn.model_selection import RandomizedSearchCV





colsample_bylevel = [1 , 0.5]

colsample_bytree = [1 , 0.5]

gamma = [0 , 1 , 5]

learning_rate = [0.1 , 0.05 , 0.0125 , 0.001]

max_delta_step = [0]

max_depth = [ 1 , 5 , 10 ]

min_child_weight = [1]

n_estimators = [ 50 , 100 , 250 , 500 , 750]

objective = ['binary:logistic']

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

                           objective = objective,

                           random_state = random_state,

                           reg_alpha = reg_alpha,

                           reg_lambda = reg_lambda,

                           scale_pos_weight = scale_pos_weight,

                           subsample = subsample , 

                           

                           ) 







estimator = XGBClassifier()     





RandomCV = RandomizedSearchCV(

                            estimator = estimator,         

                            param_distributions = param_distributions,

                            n_iter = 10,

                            cv = 5,

                            scoring = "roc_auc",   

                            random_state = 42, 

                            verbose = 1, 

                            n_jobs = None,

                            )







hyper_model = RandomCV.fit(X_train, y_train)                   

                                              



print('Best Score: ', hyper_model.best_score_)    



print('Best Params: ', hyper_model.best_params_)

hyper_model.best_estimator_.fit(X_train,y_train)



y_pred_train_hyper = hyper_model.best_estimator_.predict(X_train)  



y_pred_hyper = hyper_model.best_estimator_.predict(X_test)  

print("HYPER   TRAIN")

print('Accuracy Score ', accuracy_score( y_train , y_pred_train_hyper))

print("f1: ",f1_score(y_train , y_pred_train_hyper))

print("precision: ", precision_score(y_train , y_pred_train_hyper))

print("recall_score: ", recall_score( y_train, y_pred_train_hyper))

print("ROC AUC: ", roc_auc_score( y_train, y_pred_train_hyper))

print('---------------------------------')





print(" HYPER  TEST")

print('Accuracy Score ', accuracy_score( y_test, y_pred_hyper))

print("f1: ",f1_score(y_test, y_pred_hyper))

print("precision: ", precision_score(y_test, y_pred_hyper))

print("recall_score: ", recall_score( y_test, y_pred_hyper))

print("ROC AUC: ", roc_auc_score( y_test, y_pred_hyper))
