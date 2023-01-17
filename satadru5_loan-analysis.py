# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Loan payments data.csv")
data.head(4)
data['Principal'].unique()
data['loan_status'].unique()
data1=data.copy()
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

data['loan_status']=le.fit_transform(data['loan_status'])

data['Gender']=le.fit_transform(data['Gender'])

data['education']=le.fit_transform(data['education'])

data['past_due_days']=le.fit_transform(data['past_due_days'])
data.head(5)
df =data1[data['past_due_days'] ==60 ]

df.head(5)
data1.groupby('loan_status')['Gender'].agg(['count'])
sns.barplot(x="Gender", y="loan_status", hue="education", data=data);
data['past_due_days'].unique()
#Men are likly to keep due days

x=data.groupby('Gender')['past_due_days'].agg(['sum'])

x=pd.DataFrame(x)

x
x
sns.barplot(x='Gender',y='loan_status',data=data)
sns.barplot(x='age',y='loan_status',data=data)
sns.factorplot(x='age',y='loan_status',data=data)
sns.barplot(x='education',y='loan_status',data=data)
sns.barplot(x='Gender', y = 'loan_status', hue = 'education', data = data)
sns.countplot(x='Gender',data=data)
data.head(4)
data2=data

data2.drop('Loan_ID', axis=1, inplace=True)

label = data2.pop('loan_status')
data2.drop('effective_date', axis=1, inplace=True)

data2.drop('due_date', axis=1, inplace=True)

data2.drop('paid_off_time', axis=1, inplace=True)
data2.head(5)
####Prediction model########
#Train-Test split

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(data2, label, test_size = 0.2, random_state = 42)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(data_train, label_train)

logis_score_train = logis.score(data_train, label_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(data_test, label_test)

print("Testing score: ",logis_score_test)
coeff_df = pd.DataFrame(data2.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Correlation"] = pd.Series(logis.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
#decision tree

from sklearn.ensemble import RandomForestClassifier

dt = RandomForestClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
#random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(data_train, label_train)

rfc_score_train = rfc.score(data_train, label_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(data_test, label_test)

print("Testing score: ",rfc_score_test)
#Model comparison

models = pd.DataFrame({

        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],

        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)