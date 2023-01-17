# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
print(train.info())

print(train.head())
train.describe()
plt.figure()

sns.heatmap(train.corr(),annot=True)

plt.show()
train.ApplicantIncome.hist(bins=50,edgecolor='black')
train.boxplot(column='ApplicantIncome')
train.boxplot(column='ApplicantIncome',by='Education')
train.LoanAmount.hist(bins=50,edgecolor='black')
train.boxplot(column='LoanAmount')
train.boxplot(column='LoanAmount',by='Education')
print('Credit History')

print(train.Credit_History.value_counts())

print('Loan Status')

print(train.Loan_Status.value_counts())
train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
var1=pd.crosstab(index=train['Credit_History'],columns=train['Loan_Status'])

print(var1)

var1.plot(kind='bar',stacked='true',color=['orange','green'],grid=False)
var2=pd.crosstab(index=[train['Credit_History'],train['Gender']],columns=train['Loan_Status'])

var2.plot(kind='bar',stacked='true',color=['orange','green'],grid=False)
train.apply(lambda x: sum(x.isnull()),axis=0)
train.Gender.fillna(train['Gender'].mode()[0],inplace=True)

train.Married.fillna(train['Married'].mode()[0],inplace=True)

train.Dependents.fillna(train['Dependents'].mode()[0],inplace=True)

train.Self_Employed.fillna(train['Self_Employed'].mode()[0],inplace=True)

train.Credit_History.fillna(train['Credit_History'].mode()[0],inplace=True)
train[train['Loan_Amount_Term']==360].count()
train.Loan_Amount_Term.fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)
la_table=train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

print(la_table)
def la_fill(f):

    return la_table.loc[f['Self_Employed'],f['Education']]



train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(la_fill, axis=1), inplace=True)
train.info()
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

train['LoanAmount'].hist(bins=20,edgecolor='black')

plt.subplot(1,2,2)

train['ApplicantIncome'].hist(bins=20,edgecolor='black')
train['TotalIncome']=train['ApplicantIncome']+train['CoapplicantIncome']

train['TotalIncome_log']=np.log(train['TotalIncome'])

train['LoanAmount_log']=np.log(train['LoanAmount'])

plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

train['LoanAmount_log'].hist(bins=20,edgecolor='black')

plt.subplot(1,2,2)

train['TotalIncome_log'].hist(bins=20,edgecolor='black')
from sklearn.preprocessing import LabelEncoder

# will create a list of column names with non numeric data

col_names=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

le=LabelEncoder()

for name in col_names:

    train[name]=le.fit_transform(train[name])

train.dtypes
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics
def model_fun(model,df,features,target):

    model.fit(df[features],df[target])

    predictions=model.predict(df[features])

    accuracy=metrics.accuracy_score(predictions,df[target])

    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

    cv_score=cross_val_score(model,df[features],df[target],cv=5)

    print ("Cross validation score : %s" % "{0:.3%}".format(np.mean(cv_score)))

    model.fit(df[features],df[target])
#Logistic Regression

target_var='Loan_Status'

model=LogisticRegression()

features_var=['Credit_History']

model_fun(model,train,features_var,target_var)
features_var=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome_log', 'LoanAmount_log']

model_fun(model,train,features_var,target_var)