# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart-disease-uci/heart.csv")
#examine variables
df.info()
#examine first few rows
df.head()
#examine descriptive statistics
df.describe()
#percent count of null values
df.isnull().sum()/len(df)*100
#create dummy variables for cp
cp = pd.get_dummies(df['cp'],drop_first=True)
#print first few rows of cp
cp.head()
#rename cp columns
cp.rename(columns={1:'cp1',2:'cp2',3:'cp3'}, inplace = True)
#check
cp.head()
#create dummy variables for restecg
restecg = pd.get_dummies(df['restecg'],drop_first=True)
#print first few rows of restecg
restecg.head()
#rename restecg columns
restecg.rename(columns={1:'restecg1',2:'restecg2'}, inplace = True)
#check
restecg.head()
#create dummy variables for slope
slope = pd.get_dummies(df['slope'],drop_first=True)
#print first few rows of slope
slope.head()
#rename slope columns
slope.rename(columns={1:'slope1',2:'slope2'}, inplace = True)
#check
slope.head()
#create dummy variables for ca
ca = pd.get_dummies(df['ca'],drop_first=True)
#print first few rows of ca
ca.head()
#rename ca columns
ca.rename(columns={1:'ca1',2:'ca2',3:'ca3',4:'ca4'}, inplace = True)
#check
ca.head()
#create dummy variables for thal
thal = pd.get_dummies(df['thal'],drop_first=True)
#print first few rows of thal
thal.head()
#rename thal columns
thal.rename(columns={1:'thal1',2:'thal2',3:'thal3'}, inplace = True)
#check
thal.head()
#drop original variables for the dataset
df.drop(['cp','restecg','slope','ca','thal'], inplace = True, axis = 1)
#check for correlative relationships among numeric variables
df.corr()
#concatenate dummies to dataframe
df = pd.concat([df,cp,restecg,slope,ca,thal],axis=1)
#logistic regression model
#import
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#scale data
StSc = StandardScaler()
#variables to be scaled
scaled_num_col = ['age','trestbps','chol','thalach','oldpeak']
#scale selected variables
df[scaled_num_col] = StSc.fit_transform(df[scaled_num_col])
#examine scaled data
df.describe()
#independent variables
X = df.drop('target',axis=1)
#dependent variable
y = df['target']
#train/test split of the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#import
from sklearn.linear_model import LogisticRegression
#logit model
logmodel = LogisticRegression()
#model fit
logmodel.fit(X_train,y_train)
#call predictions
predictions=logmodel.predict(X_test)
#classification report
print (classification_report(y_test,predictions))
#compare distribution in data set - looks good
df['target'].value_counts(normalize=True) * 100
#print confusion matrix - looks good
confusion_matrix (y_test,predictions)
#decision tree model
#import
from sklearn.tree import DecisionTreeClassifier
#decision tree
dtree = DecisionTreeClassifier()
#model fit
dtree.fit(X_train,y_train)
#call predictions
predictions = dtree.predict(X_test)
#print classification report
print(classification_report(y_test,predictions))
#print confusion matrix
print(confusion_matrix(y_test,predictions))
#random forest model
#import
from sklearn.ensemble import RandomForestClassifier
#random forest model
rfc = RandomForestClassifier(n_estimators=300)
#model fit
rfc.fit(X_train,y_train)
#call predictions
predictions = rfc.predict(X_test)
#print classification report
#notice an improvement compared to decesion tree model
print(classification_report(y_test,predictions))
#print confusion matrix
print(confusion_matrix(y_test,predictions))
#conclusion : logit model outperforms both models
#random forest improves decision tree model
#in terms of precision, recall, f1-score, and accuracy