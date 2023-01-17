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
import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
df=pd.read_csv("../input/LoanStats3a.csv")
total_NA_count=df.isnull().sum().sort_values(ascending=False)
perc=(df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
missing_data=pd.concat([total_NA_count,perc],axis=1,keys=['Total','Percent'])
missing_data
#missing data percent is given below:
data_present=[]
for index in range(0,len(missing_data)):
    if(missing_data.iloc[index,1]<20):
        data_present.append(missing_data.index[index])
#dataframe with data present
df1=df[data_present]
missing_value=df1.isnull().sum()*100/len(df1.iloc[:,1])
print(missing_value)
df1=df1.dropna(how='any',axis=0)
missing_value=df1.isnull().sum()*100/len(df1.iloc[:,1])
print(missing_value)
df1["collections_12_mths_ex_med"].unique()
df.dtypes
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df1['emp_length'].unique())
df1["emp_length"]=le.transform(df1["emp_length"]) 
df1["emp_length"].head()
df1['int_rate'] = (df1['int_rate'].str.strip('%').astype(float))
df1["int_rate"].head()
df1=df1.drop(columns=['title','emp_title'])
for col in df1.columns:
    if len(df1[col].unique()) == 1:
        df1.drop(col,inplace=True,axis=1)
import datetime
df1['last_pymnt_d'] = datetime.date.today().year
df1['earliest_cr_line'] = datetime.date.today().year
df1['last_credit_pull_d'] = datetime.date.today().year
df1['issue_d'] = datetime.date.today().year
df1.head()
df1['term'] = (df1['term'].str.strip(' months').astype(float))
df1["term"].head()
df1=df1.drop(columns=['purpose'])
df1['loan_status'].unique()
df1=pd.get_dummies(df1, columns=["verification_status"])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df1['loan_status'].unique())
df1["loan_status"]=le.transform(df1["loan_status"]) 
df1.head()
df1=df1.drop(columns=['home_ownership'])
df1['revol_util'] = (df1['revol_util'].str.strip('%').astype(float))
df1['verification_status_Not Verified']=(df1['verification_status_Not Verified'].astype(float))
df1['verification_status_Source Verified']=(df1['verification_status_Source Verified'].astype(float))
df1['verification_status_Verified']=(df1['verification_status_Verified'].astype(float))
df1['revol_util'].head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df1['debt_settlement_flag'].unique())
df1["debt_settlement_flag"]=le.transform(df1["debt_settlement_flag"])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df1['grade'].unique())
df1["grade"]=le.transform(df1["grade"])
df1.dtypes
df1=df1.drop(columns=['sub_grade','addr_state','zip_code'])
df1.head()
x=df1.drop(columns=['loan_status'])
x.head()
y = df1[['loan_status']].copy()
y.head()
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred
# Making the confusion matrix
from sklearn import metrics
print (metrics.accuracy_score(y_test,classifier.predict(x_test)))
from sklearn.tree import DecisionTreeClassifier
clf_tree=DecisionTreeClassifier()
clf_tree.fit(x_train,y_train)
cm_1=confusion_matrix(y_test,clf_tree.predict(x_test),[1,0])
cm_1
metrics.accuracy_score(y_test,clf_tree.predict(x_test))
#Support Vector Machines
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc
# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree