#import standard libraries

import sys

import matplotlib as mpl

import scipy as sp

import IPython

from IPython import display

import sklearn

import numpy as np

import pandas as pd

import sklearn

import random

import time

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



#import modelling libraries

#algorithms

from sklearn import svm,tree,ensemble,linear_model,neighbors,naive_bayes,discriminant_analysis,gaussian_process

from xgboost import XGBClassifier

import lightgbm as lgb



#helper

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn import model_selection

from sklearn import feature_selection

from sklearn import metrics



#visulization

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
dr = pd.read_csv("../input/train.csv")#Train Data(raw)

dt = pd.read_csv("../input/test.csv")#Test Data

d1 = dr.copy(deep=True)#To play with train data.

dc = pd.concat([d1,dt])#To clean data

print(dr.info())

dr.sample(10)
print("Train data's null values. ",d1.isnull().sum())

print("Test data's null values.",dt.isnull().sum())

d1.describe(include='all')
#replacing and completing

d1=d1.replace({"male":0,"female":1})

dt=dt.replace({"male":0,"female":1})

d1['Age'].fillna(dc['Age'].median(), inplace=True)

dt['Age'].fillna(dc['Age'].median(), inplace=True)

d1["Embarked"].fillna(dc["Embarked"].mode()[0],inplace=True)

dt["Fare"].fillna(dc["Fare"].median(), inplace=True)



#get dummies

e = pd.get_dummies(d1["Embarked"])#to train

d1["S"] = e["S"]

d1["C"] = e["C"]

d1["Q"] = e["Q"]

del(d1["Embarked"])

p = pd.get_dummies(d1["Pclass"])

d1["p1"] = p[1]

d1["p2"] = p[2]

d1["p3"] = p[3]

del(d1["Pclass"])



e_t = pd.get_dummies(dt["Embarked"])#to test

dt["S"] = e_t["S"]

dt["C"] = e_t["C"]

dt["Q"] = e_t["Q"]

del(dt["Embarked"])

p_t = pd.get_dummies(dt["Pclass"])

dt["p1"] = p_t[1]

dt["p2"] = p_t[2]

dt["p3"] = p_t[3]

del(dt["Pclass"])



#delete columns

del([d1["PassengerId"],d1["Cabin"],d1["Ticket"]])

del([dt["Cabin"],dt["Ticket"]])
#fimily imfo

d1["Family size"] = d1["Parch"]+d1["SibSp"]+1

d1["Is Alone"] = 1

d1["Is Alone"].loc[d1["Family size"]>1]=0

dt["Family size"] = dt["Parch"]+dt["SibSp"]+1

dt["Is Alone"] = 1

dt["Is Alone"].loc[dt["Family size"]>1]=0



#name info

d1["Mr"]=0

d1["Mrs"]=0

d1["Miss"]=0

d1["Master"]=0

d1["Misc"]=0

for i in range(len(d1)):

    if 'Mr.' in d1.ix[i,"Name"]:

        d1.ix[i,"Mr"]=1

    elif 'Mrs.' in d1.ix[i,"Name"]:

        d1.ix[i,"Mrs"]=1

    elif 'Miss.' in d1.ix[i,"Name"]:

        d1.ix[i,"Miss"]=1

    elif 'Master.' in d1.ix[i,"Name"]:

        d1.ix[i,"Master"]=1

    elif 'Misc.' in d1.ix[i,"Name"]:

        d1.ix[i,"Misc"]=1

dt["Mr"]=0

dt["Mrs"]=0

dt["Miss"]=0

dt["Master"]=0

dt["Misc"]=0

for i in range(len(dt)):

    if 'Mr.' in dt.ix[i,"Name"]:

        dt.ix[i,"Mr"]=1

    elif 'Mrs.' in dt.ix[i,"Name"]:

        dt.ix[i,"Mrs"]=1

    elif 'Miss.' in dt.ix[i,"Name"]:

        dt.ix[i,"Miss"]=1

    elif 'Master.' in dt.ix[i,"Name"]:

        dt.ix[i,"Master"]=1

    elif 'Misc.' in dt.ix[i,"Name"]:

        dt.ix[i,"Misc"]=1
X = d1[["Age","SibSp","Parch","Sex","Fare","S","C","Q","p1","p2","p3","Mr","Mrs","Miss","Master","Misc","Family size","Is Alone"]]

y = d1["Survived"]
vote_est = [

    #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html

    ('ada', ensemble.AdaBoostClassifier(random_state=0)),

    ('bc', ensemble.BaggingClassifier(random_state=0)),

    ('etc',ensemble.ExtraTreesClassifier(random_state=0)),

    ('gbc', ensemble.GradientBoostingClassifier(random_state=0)),

    ('rfc', ensemble.RandomForestClassifier(random_state=0)),



    #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc

    ('gpc', gaussian_process.GaussianProcessClassifier(random_state=0)),

    

    #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    ('lr', linear_model.LogisticRegressionCV(random_state=0)),

    

    #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html

    ('bnb', naive_bayes.BernoulliNB()),

    ('gnb', naive_bayes.GaussianNB()),

    

    #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html

    ('knn', neighbors.KNeighborsClassifier()),

    

    #SVM: http://scikit-learn.org/stable/modules/svm.html

    ('svc', svm.SVC(probability=True)),

    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

   ('xgb', XGBClassifier(seed=0))



]

cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3,train_size=.6,random_state=0)

vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = model_selection.cross_validate(vote_hard, X, y, cv  = cv_split)

vote_hard.fit(X, y)



print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 

print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))

print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))

print('-'*10)
# ランダムフォレスト

prams_r = {'n_estimators':[10,40,100],'criterion':["gini","entropy"],"max_depth":[2,3,4,5,6]}

clf_r = ensemble.RandomForestClassifier(random_state=0)

model_r = model_selection.GridSearchCV(clf_r,prams_r,n_jobs=-1)

model_r.fit(X,y)

model_r.best_score_
#xgboost

prams_x = {

    'criterion':["gini","entropy"],

    'max_depth':range(2,11,2),

    'n_estimators':[10,50,100,300],

    'learning_rate':[0.01,0.03,0.5,0.1,0.25],

}

clf_x = XGBClassifier(verbose=1)

model_x = model_selection.GridSearchCV(clf_x,prams_x,n_jobs=-1)

model_x.fit(X,y)

model_x.best_score_
Xt = dt[["Age","SibSp","Parch","Sex","Fare","S","C","Q","p1","p2","p3","Mr","Mrs","Miss","Master","Misc","Family size","Is Alone"]]

dt['Survived'] = vote_hard.predict(Xt)



submit = dt[['PassengerId','Survived']]

submit.to_csv("../working/submit.csv", index=False)

#output=model_x.predict(Xt)
#print(len(dt.index), len(output))

#zip_datan = zip(dt.PassengerId.astype(int), output.astype(int))

#predict_datan = list(zip_datan)
#import csv

#with open("predict_result_data.csv", "w") as f:

#    writer = csv.writer(f, lineterminator='\n')

#    writer.writerow(["PassengerId", "Survived"])

#    for pid, survived in zip(dt.PassengerId.astype(int), output.astype(int)):

#        writer.writerow([pid, survived])