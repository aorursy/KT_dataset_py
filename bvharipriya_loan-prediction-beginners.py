import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
test_df = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')

df = [train_df, test_df]
train_df
test_df
train_df['Loan_Status'] = train_df.Loan_Status.map({'Y': 1, 'N': 0}).astype(int)
train_df["Gender"].isnull().sum()
train_df[['Gender', 'Loan_Status']].groupby('Gender', as_index=False).mean()
for dataset in df:
    dataset.Gender.fillna('Male', inplace=True)
train_df["Gender"].isnull().sum()
for dataset in df:
    dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0}).astype(int)
train_df.Married.isnull().sum()
train_df[['Married', 'Loan_Status']].groupby('Married', as_index=False).mean()
for dataset in df:
    dataset.Married.fillna('Yes', inplace=True)
train_df.Married.isnull().sum()
for dataset in df:
    dataset['Married'] = dataset['Married'].map({'Yes': 1, 'No': 0}).astype(int)
train_df.Dependents.isnull().sum()
for dataset in df:
    dataset['Dependents'] = dataset['Dependents'].fillna(train_df.Dependents.mode()[0])
    dataset['Dependents'] = dataset['Dependents'].replace('3+', '3')
    dataset['Dependents'] = dataset.Dependents.astype(int)
train_df.head()
for dataset in df:
    dataset['Education'] = dataset['Education'].map({'Graduate': 1, 'Not Graduate': 0}).astype(int)
train_df.Self_Employed.isnull().sum()
train_df[['Self_Employed', 'Loan_Status']].groupby('Self_Employed', as_index=False).mean()
for dataset in df:
    dataset['Self_Employed'] = dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0])
    dataset['Self_Employed'] = dataset['Self_Employed'].map({'No': 0, 'Yes': 1}).astype(int)
train_df.Credit_History.isnull().sum()
for dataset in df:
    dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0]).astype(int)
for dataset in df:
    dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean())

train_df['Loan_Amount_Term'].isnull().sum()
X = train_df[["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome","Loan_Amount_Term"]]
y = train_df['Loan_Status']
X
y
data_corr=pd.concat([X,y],axis=1)
data_corr.corr()
LogReg_classifier = LogisticRegression()
LogReg_classifier.fit(X,y)
SVM_classifier = SVC()
SVM_classifier.fit(X,y)
Knn_classifier = KNeighborsClassifier()
Knn_classifier.fit(X,y)
Tree_classifier = DecisionTreeClassifier()
Tree_classifier.fit(X,y)
Ran_classifier = RandomForestClassifier(n_estimators=100)
Ran_classifier.fit(X, y)
