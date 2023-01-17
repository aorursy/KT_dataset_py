## On ‘Wuzzuf_Job_Postings_Sample.csv’, the following has been done:
## I used OpenRefine to:
## Remove all the rows with missing values in job_description, job_requirements, 
##         payment_period, currency (21850 rows to 19112 rows)
## Clustered all the similar cities in city, but kept certain data that were dates as is - 
##        “10th of Ramadan”, “6th October”, etc., as it is uninformative on which 
##        city it refers to
## the input data here is the cleaned dataset
## ----------------------------------------------------------------------------------------------
## I have also done a few pre-modeling Vizs with a couple of good insights which you can
## check out on: 
## https://public.tableau.com/profile/pooja.vijaykumar#!/vizhome/WuzzufJobPostingsDataset/Story1
## ----------------------------------------------------------------------------------------------
## A Classifier can be created that classifies each application based on job_category1 so that 
## when a person enters their qualifications, max/min salary expectations, their skills, etc., 
## their application falls into the respective category such as Engineering, IT, Marketing, 
## HR, Banking, Retail, Fashion, Journalism etc., 
## Features: salary min/max, payment_period, currency, num_vacancies, career_level
## Target: job_category1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
%matplotlib inline
print(os.listdir('../input'))
data1 = pd.read_csv('../input/cleaned-wuzzuf-job-posts-sample/Cleaned_Wuzzuf_Job_Posts_Sample-csv.csv')
data2 = pd.read_csv('../input/wuzzuf-job-posts/Wuzzuf_Applications_Sample.csv')
data1.head()
data1.info()
## check for missing values
data1.isnull().sum()
## remove record with the single missing value
data1 = data1.dropna(axis=0)
data2.head()
data2.info()
data11 = data1[['job_category1','job_industry1','salary_minimum','salary_maximum','num_vacancies','career_level','payment_period','currency']]
data11.head(3)
data11 = data11.join(pd.get_dummies(data11['career_level'],prefix='CareerLevel'))
data11 = data11.join(pd.get_dummies(data11['payment_period'],prefix='paymentperiod'))
data11 = data11.join(pd.get_dummies(data11['currency'],prefix='currency'))
data11 = data11.drop(['job_industry1','career_level','payment_period','currency'],axis=1)
data11.head()
x = data11.drop('job_category1',axis=1)
y = data11['job_category1']
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=123)
DTmodel = DecisionTreeClassifier(max_depth=4).fit(xtrain,ytrain)
DTpred = DTmodel.predict(xtest)
DTpred[:5,]
ytest[:5,]
from sklearn.metrics import accuracy_score

accuracy_score(ytest,DTpred)
## Run this if you want to observe the Decision Tree diagram
## useful to observe the feature split, gini index, etc

#from sklearn.tree import export_graphviz
#import graphviz

#dot_data = export_graphviz(DTmodel, filled=True, rounded=True, feature_names=xtrain.columns, out_file=None)
#graphviz.Source(dot_data)
from sklearn.model_selection import GridSearchCV
param = [{"max_depth":[5,6,7,8,9,10,11,12,13,14, None], "max_features":[7,10,11,12,13,14,15,16]}]
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=321),param_grid=param,cv=2)
gs.fit(xtrain,ytrain)
gs.best_params_
pred = gs.predict(xtest)
accuracy_score(ytest,pred)
from sklearn.ensemble import RandomForestClassifier

RFmodel = RandomForestClassifier(n_estimators=500, n_jobs=-1)
RFmodel.fit(xtrain, ytrain)
RFpred=RFmodel.predict(xtest)
accuracy_score(ytest,RFpred)
