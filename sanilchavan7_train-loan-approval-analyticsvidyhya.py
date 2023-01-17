# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/loan-approval-1/Train.csv')

train.head()

import seaborn as sns

import matplotlib.pyplot as plt

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']



fig,axes = plt.subplots(4,2,figsize=(12,15))

for idx,cat_col in enumerate(categorical_columns):

    row,col = idx//2,idx%2

    sns.countplot(x=cat_col,data=train,hue='Loan_Status',ax=axes[row,col])





plt.subplots_adjust(hspace=1)
train.isnull().sum().sort_values(ascending=False)



import seaborn as sns

sns.set_style('whitegrid')

#sns.countplot(x='ApplicantIncome',hue='Credit_History',data=train,palette='RdBu_r')

#Replace the credit_history with 1 with loan status y AND credit_history 
def Credit_score(cols):

    Loan_Status = cols[0]

    Credit_History = cols[1]

    if pd.isnull(Credit_History):

        if Loan_Status=='Y':

            return 1.0

        elif Loan_Status=='N':

            return 0.0

    else:

        return Credit_History

        

    
train['Credit_History'] = train[['Loan_Status','Credit_History']].apply(Credit_score,axis=1)
train['Credit_History'].isnull().sum()
train.isnull().sum().sort_values(ascending=False)
sns.set_style('whitegrid')

sns.countplot(x='Self_Employed',hue='Loan_Status',data=train,palette='RdBu_r')

train['Self_Employed'].unique()
def self_employed(cols):

    Loan_Status = cols[0]

    Self_Employed = cols[1]

    if pd.isnull(Self_Employed):

        if Loan_Status=='Y':

            return 'No'

        elif Loan_Status=='N':

            return 'Yes'

    else:

        return Self_Employed
train['Self_Employed'] = train[['Loan_Status','Self_Employed']].apply(self_employed,axis=1)
train['Self_Employed'].isnull().sum()
train.isnull().sum().sort_values(ascending=False)

sns.stripplot(x=train['Loan_Status'], y=train['LoanAmount'], data=train) 

train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mean())
train.isnull().sum().sort_values(ascending=False)

sns.countplot(x='Dependents',hue='Loan_Status',data=train,palette='RdBu_r')

train['Dependents'].unique()

corr = train.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)

train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
train.isnull().sum().sort_values(ascending=False)
sns.countplot(x='Gender',hue='Loan_Status',data=train,palette='RdBu_r')

def gender(cols):

    Loan_Status = cols[0]

    gender = cols[1]

    if pd.isnull(gender):

        if Loan_Status=='Y':

            return 'Male'

        elif Loan_Status=='N':

            return 'Female'

    else:

        return gender
train['Gender'] = train[['Loan_Status','Gender']].apply(self_employed,axis=1)
train.isnull().sum().sort_values(ascending=False)
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean())
train.isnull().sum().sort_values(ascending=False)
train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
train.isnull().sum().sort_values(ascending=False)
train.info()
train.drop('Loan_ID',axis=1,inplace=True)
df = train

cols = df.select_dtypes(include=['object'])

categorical_columns = cols.columns.to_list()

categorical_columns.remove('Loan_Status')
categorical_columns
df1 = pd.get_dummies(df,columns=categorical_columns,drop_first=True)
loan_status  = pd.get_dummies(df['Loan_Status'],drop_first=True)



df1['loan_status'] = loan_status
df1
from sklearn.model_selection import train_test_split

x = df1.iloc[:,:-1]

y = df1['loan_status']

x.drop('Loan_Status',axis=1,inplace=True)

x.drop('Gender_No',axis=1,inplace=True)

x.drop('Gender_Yes',axis=1,inplace=True)

xgboost = x

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train,y_train)
prediction = logreg.predict(x_test)

print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sample = pd.read_csv('/kaggle/input/sample-submission/sample_submission_49d68Cx.csv')

predict1 = pd.read_csv('/kaggle/input/test1234/Test1.csv')

predict1
predict12 = logreg.predict(predict1)

predict12
#Steps to convert dataset to submission file converting 1 to Y AND 0 to N

df123 = pd.DataFrame(data=predict12, columns=["column1"])



hello = pd.concat([sample['Loan_ID'],df123],axis=1)

hello.drop(['column1','Loan_Status'],axis=1,inplace=True)

hello['Loan_Status1'] = hello['Loan_Status'].replace(1,'Y')

hello.rename(columns = {'Loan_Status1':'Loan_Status'}, inplace = True) 

hello
hello.rename(columns = {'Loan_Status1':'Loan_Status'}, inplace = True) 

hello['Loan_Status1'] = hello['Loan_Status'].replace(1,'Y')
hello1
predict12