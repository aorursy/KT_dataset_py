from flask import Flask

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV

import pickle as pkl

import warnings

warnings.filterwarnings("ignore")
train_data = pd.read_csv('../input/train_u.csv')

test_data = pd.read_csv('../input/test_Y.csv')

train_data.columns
for _ in train_data.columns:

    print("the number of null values in :{} == {}".format(_,train_data[_].isnull().sum()))
train_data
var = ['Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',

       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

train_X,test_X,train_Y,test_Y = train_test_split(train_data[var],train_data['Loan_Status'],test_size = 0.3,random_state=0)
train_Y=train_Y.replace({'Y':1,'N':0}).as_matrix()

test_Y=test_Y.replace({'Y':1,'N':0}).as_matrix()
class pre_processing(BaseEstimator,TransformerMixin):

    def __init__(self):

        pass

    def transform(self,df):

        gender_values = {'Female':0,'Male':1}

        married_values = {'No':0,'Yes':1}

        education_values = {'Graduate':0,'Not Graduate':1}

        employment_values = {'No':0,'Yes':1}

        property_values = {'Urban':1,'Rural':0,'Semiurban':2}

        dependents_values = {'0':0,'1':1,'2':2,'3+':3}

        var = ['Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',

       'Loan_Amount_Term', 'Credit_History', 'Property_Area']

        df = df[var]

        df['Dependents']=df['Dependents'].fillna(0)

        df['Gender']=df['Gender'].fillna('Male')

        df['Married']=df['Married'].fillna('No')

        df['Self_Employed']=df['Self_Employed'].fillna('No')

        df['LoanAmount']=df['LoanAmount'].fillna(self.term_mean_)

        df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(self.amt_mean_)

        df['Credit_History']=df['Credit_History'].fillna(1)

        df.replace({'Gender':gender_values,'Married':married_values,'Education':education_values,'Self_Employed':employment_values,'Property_Area':property_values,'Dependents':dependents_values},inplace = True)

        return df.as_matrix()

    def fit(self,df,y = None,**fit_params):

        self.amt_mean_=df['Loan_Amount_Term'].mean()

        self.term_mean_=df['LoanAmount'].mean()

        return self

pipe = make_pipeline(pre_processing(),RandomForestClassifier())

pipe
parameters = ({'randomforestclassifier__max_depth':[None,2,3],'randomforestclassifier__n_estimators':[10,20,30],'randomforestclassifier__min_impurity_split':[0.1,0.2,0,3],'randomforestclassifier__max_leaf_nodes':[None,5,10,20]})
grid_search = GridSearchCV(estimator = pipe,param_grid = parameters,scoring = 'accuracy',cv=3)
grid_search.fit(train_X,train_Y)
best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_

best_parameters,best_accuracy

grid_search.score(test_X,test_Y)
grid_search.predict(test_data)
test_data.shape
filename = 'model_v1_pk'

with open(filename,'wb') as f:

    pkl.dump(grid_search,f)