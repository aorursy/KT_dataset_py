# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#IN THIS DATA WE HAVE TO PREDICT THE CUSTOMERS WHO WILL SUBSCRIBE TERM DEPOSITS IN THAT BANK...LETS DO IT...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Import dataset
bank=pd.read_csv("../input/bank.csv")
#Lets took a look on our Data
bank.head(10)
bank.describe()
bank.info()
#Thats a big Data set consists of 41188 Rows and 21columns
#COLUMN NAMES
bank.columns
bank.dtypes
#Finding Null Values
bank.isnull().sum()
#Amazing ,we got Zero null vales...its good to start.
bank.describe().plot()
len(bank)
#let's find correlation among our Data set....
bank_cor=bank.corr()
plt.figure(figsize=(12,8))
sns.heatmap(bank_cor,linewidth=0.5)
sns.pairplot(bank)
#Its beautiful.
qualitative=[f for f in  bank.columns if bank[f].dtype == "object"]
quantitative=[f for f in bank.columns if bank[f].dtype!= "object"]
print(len(quantitative))
print(len(qualitative))
plt.figure(figsize=(10,10))
sns.jointplot(x='age',y='y',data=bank,kind='hex')
bank.columns
plt.figure(figsize=(10,10))
sns.jointplot(x='cons_price_idx',y='y',data=bank,kind='reg')
sns.boxplot(x="day_of_week",y="age",data=bank)
plt.figure(figsize=(10,10))
sns.boxplot(x="y",y="age",hue="marital",data=bank,palette="Set3")
#From the above plot we can conclude that Married and Divorced pepole of age b/w 35-55 subscribed for Term deposits.
bank.groupby('job').y.mean().plot(kind='bar')
#THIS SHOWS THAT STUDENTS AND RETIRED EMPLOYEES ARE MORE LIKLEY TO SUBSCRIBE TERM DEPOSITS

bank.columns
bank.groupby('loan').y.mean().plot(kind='bar')

bank.groupby('education').y.mean().plot(kind='bar')

#DATA PREPROCESSING
#CONVERTING CATEGORICAL VARIBALES TO NUMERICAL VARIABLES
from sklearn.preprocessing import LabelEncoder
encoding_list = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact','poutcome','day_of_week','month','duration']
bank[encoding_list] = bank[encoding_list].apply(LabelEncoder().fit_transform)

qualitative=[f for f in  bank.columns if bank[f].dtype == "object"]
quantitative=[f for f in bank.columns if bank[f].dtype!= "object"]
print(len(quantitative))
print(len(qualitative))
#YUPP...NOW OUR DATA HAS NO CATEGORICAL VARIABLE...THATS GREAT.
x=bank.iloc[:,:-1].values
y=bank.iloc[:,-1].values
#SPLITTING DATA
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
#MODEL SELECTION
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,cross_val_score
k_fold=KFold(n_splits=10,shuffle=True,random_state=0)
clf=KNeighborsClassifier()
scoring='accuracy'
score=cross_val_score(clf,x_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=DecisionTreeClassifier()
scoring='accuracy'
score=cross_val_score(clf,x_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=GaussianNB()
scoring='accuracy'
score=cross_val_score(clf,x_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=RandomForestClassifier()
scoring='accuracy'
score=cross_val_score(clf,x_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=SVC()
scoring='accuracy'
score=cross_val_score(clf,x_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
clf=LogisticRegression()
scoring='accuracy'
score=cross_val_score(clf,x_train,y_train,n_jobs=1,cv=k_fold,scoring=scoring)
print(score)
#FITTING DATA TO SELECTED MODEL ..ALL THE ABOVE CLASSIFICATION MODELS HAVE GIVEN BETTER RESULTS...LET WE OPT FOR lOGISTIC REGRESSION MODEL FOR FURTHEL MODEL DEVELOPMENT
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(y_pred)
#METRICS
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm)
cl=classification_report(y_test,y_pred)
print(cl)
#THATS GREAT OUR MODEL GOT 90% ACCURACY.....