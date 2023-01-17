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

from xgboost import XGBClassifier



import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

test_df = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')



df = [train_df, test_df]
train_df.info()
train_df.head()
sns.heatmap(train_df.isnull(), cbar=False, yticklabels=False)
# Loan_Status feature --- target variable

train_df['Loan_Status'] = train_df.Loan_Status.map({'Y': 1, 'N': 0}).astype(int)
# Gender feature

train_df.Gender.value_counts()
train_df.Gender.isnull().sum()
test_df.Gender.isnull().sum()
train_df[['Gender', 'Loan_Status']].groupby('Gender', as_index=False).mean()
grid = sns.FacetGrid(train_df, col='Loan_Status')

grid.map(plt.hist, 'Gender')
for dataset in df:

    dataset.Gender.fillna('Male', inplace=True)
train_df.Gender.isnull().sum()
# Changing Gender feature into numeric so that our model works properly, kind of label encoding

for dataset in df:

    dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0}).astype(int)
# Married Feature

train_df.Married.value_counts()
train_df.Married.isnull().sum()
for dataset in df:

    dataset['Married'] = dataset.Married.fillna(dataset.Married.mode()[0])
train_df[['Married', 'Loan_Status']].groupby('Married', as_index=False).mean()
sns.set(style='whitegrid')

grid = sns.FacetGrid(train_df, col='Loan_Status')

grid.map(plt.hist, 'Married') 
grid = sns.FacetGrid(train_df, row='Education', size=2.8, aspect=1.6)

grid.map(sns.barplot, 'Married', 'Loan_Status', 'Gender', ci=None, palette='deep')

grid.add_legend()
for dataset in df:

    dataset['Married'] = dataset['Married'].map({'Yes': 1, 'No': 0}).astype(int)
# Dependents feature

train_df.Dependents.value_counts()
train_df.Dependents.isnull().sum()
grid = sns.FacetGrid(train_df, row='Gender')

grid.map(sns.barplot, 'Dependents', 'Loan_Status', palette='deep', ci=None)

grid.add_legend()
for dataset in df:

    dataset['Dependents'] = dataset['Dependents'].fillna(train_df.Dependents.mode()[0])

    dataset['Dependents'] = dataset['Dependents'].replace('3+', '3')

    dataset['Dependents'] = dataset.Dependents.astype(int)
train_df.head()
# Education, I do this every time to check any error or typos in any categorical feature.

train_df.Education.value_counts()
train_df[['Education', 'Loan_Status']].groupby('Education', as_index=False).mean()
train_df.Education.isnull().sum()
test_df.Education.isnull().sum()
for dataset in df:

    dataset['Education'] = dataset['Education'].map({'Graduate': 1, 'Not Graduate': 0}).astype(int)
# Self_Employed

train_df.Self_Employed.value_counts()
train_df.Self_Employed.isnull().sum()
train_df[['Self_Employed', 'Loan_Status']].groupby('Self_Employed', as_index=False).mean()
for dataset in df:

    dataset['Self_Employed'] = dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0])

    dataset['Self_Employed'] = dataset['Self_Employed'].map({'No': 0, 'Yes': 1}).astype(int)
# Credit_History

train_df.Credit_History.value_counts()
train_df.Credit_History.isnull().sum()
# gender, married, credit history, loan status

# gender, education, credit history, loan status

# gender, self employed, credit history, loan status
grid = sns.FacetGrid(train_df, row='Married', aspect=1.5)

grid.map(sns.barplot, 'Credit_History', 'Loan_Status', 'Gender', palette='deep', ci=None)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Education', aspect=1.5)

grid.map(sns.barplot, 'Credit_History', 'Loan_Status', 'Gender', palette='deep', ci=None)

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Self_Employed', aspect=1.5)

grid.map(sns.barplot, 'Credit_History', 'Loan_Status', 'Gender', palette='deep', ci=None)

grid.add_legend()
for dataset in df:

    dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0]).astype(int)
# Property_Area

train_df.Property_Area.value_counts()
train_df.Property_Area.isnull().sum()
train_df[['Property_Area', 'Loan_Status']].groupby('Property_Area', as_index=False).mean().sort_values(by='Loan_Status', ascending=False)
grid = sns.FacetGrid(train_df, row='Married', aspect=1.5)

grid.map(sns.barplot, 'Property_Area', 'Loan_Status', 'Gender', palette='deep', ci=None)

grid.add_legend()
for dataset in df:

    dataset['Property_Area'] = dataset['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2}).astype(int)
train_df.head()
# ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term.
train_df.describe()
sns.set(style='darkgrid')

sns.boxplot(train_df.ApplicantIncome)
sns.set(style='darkgrid')

sns.boxplot(train_df.CoapplicantIncome)
sns.set(style='darkgrid')

sns.boxplot(train_df.LoanAmount)
sns.set(style='darkgrid')

sns.boxplot(train_df.Loan_Amount_Term)
train_df['ApplicantIncome'] = train_df['ApplicantIncome'].astype(int)
train_df['ApplicantIncomeBand'] = pd.cut(train_df['ApplicantIncome'], 4)

train_df[['ApplicantIncomeBand', 'Loan_Status']].groupby('ApplicantIncomeBand', as_index=False).mean().sort_values(by='ApplicantIncomeBand', ascending=True)
train_df['CoapplicantIncome'] = train_df['CoapplicantIncome'].astype(int)
train_df['CoapplicantIncomeBand'] = pd.cut(train_df['CoapplicantIncome'], 3)

train_df[['CoapplicantIncomeBand', 'Loan_Status']].groupby('CoapplicantIncomeBand', as_index=False).mean().sort_values(by='CoapplicantIncomeBand', ascending=True)
for dataset in df:

    dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
train_df['LoanAmountBand'] = pd.cut(train_df['LoanAmount'], 4)

train_df[['LoanAmountBand', 'Loan_Status']].groupby('LoanAmountBand', as_index=False).mean().sort_values(by='LoanAmountBand', ascending=True)
for dataset in df:

    dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean())
train_df['Loan_Amount_TermBand'] = pd.cut(train_df['Loan_Amount_Term'], 3)

train_df[['Loan_Amount_TermBand', 'Loan_Status']].groupby('Loan_Amount_TermBand', as_index=False).mean().sort_values(by='Loan_Amount_TermBand', ascending=True)
train_df.head()
for dataset in df:

    dataset.loc[dataset['ApplicantIncome'] <= 20362.5, 'ApplicantIncome'] = 0

    dataset.loc[(dataset['ApplicantIncome'] > 20362.5) & (dataset['ApplicantIncome'] <= 40575.0), 'ApplicantIncome'] = 1

    dataset.loc[(dataset['ApplicantIncome'] > 40575.0) & (dataset['ApplicantIncome'] <= 60787.5), 'ApplicantIncome'] = 2

    dataset.loc[(dataset['ApplicantIncome'] > 60787.5), 'ApplicantIncome'] = 3
for dataset in df:

    dataset.loc[dataset['CoapplicantIncome'] <= 13889.0, 'CoapplicantIncome'] = 0

    dataset.loc[(dataset['CoapplicantIncome'] > 13889.0) & (dataset['CoapplicantIncome'] <= 27778.0), 'CoapplicantIncome'] = 1

    dataset.loc[(dataset['CoapplicantIncome'] > 27778.0), 'CoapplicantIncome'] = 2
for dataset in df:

    dataset.loc[dataset['LoanAmount'] <= 181.75, 'LoanAmount'] = 0

    dataset.loc[(dataset['LoanAmount'] > 181.75) & (dataset['LoanAmount'] <= 354.5), 'LoanAmount'] = 1

    dataset.loc[(dataset['LoanAmount'] > 354.5) & (dataset['LoanAmount'] <= 527.25), 'LoanAmount'] = 2

    dataset.loc[(dataset['LoanAmount'] > 527.25), 'LoanAmount'] = 3

    dataset['LoanAmount'] = dataset['LoanAmount'].astype(int)
for dataset in df:

    dataset.loc[dataset['Loan_Amount_Term'] <= 168.0, 'Loan_Amount_Term'] = 0

    dataset.loc[(dataset['Loan_Amount_Term'] > 168.0) & (dataset['Loan_Amount_Term'] <= 324.0), 'Loan_Amount_Term'] = 1

    dataset.loc[(dataset['Loan_Amount_Term'] > 324.0), 'Loan_Amount_Term'] = 2

    dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].astype(int)
train_df.head()
train_df.drop('ApplicantIncomeBand', inplace=True, axis=1)

train_df.drop('CoapplicantIncomeBand', inplace=True, axis=1)

train_df.drop('LoanAmountBand', inplace=True, axis=1)

train_df.drop('Loan_Amount_TermBand', inplace=True, axis=1)
for dataset in df:

    dataset.drop('Loan_ID', axis=1, inplace=True)
X = train_df.drop('Loan_Status', axis=1)

y = train_df['Loan_Status']
data_corr = pd.concat([X, y], axis=1)

corr = data_corr.corr()

plt.figure(figsize=(11,7))

sns.heatmap(corr, annot=True)
LogReg_classifier = LogisticRegression()

LogReg_classifier.fit(X,y)
LogReg_acc = cross_val_score(LogReg_classifier, X, y, cv=10, scoring='accuracy').mean()

LogReg_acc
SVM_classifier = SVC()

SVM_classifier.fit(X,y)
SVM_acc = cross_val_score(SVM_classifier, X, y, cv=10, scoring='accuracy').mean()

SVM_acc
Knn_classifier = KNeighborsClassifier()

Knn_classifier.fit(X,y)
Knn_acc = cross_val_score(Knn_classifier, X, y, cv=10, scoring='accuracy').mean()

Knn_acc
Tree_classifier = DecisionTreeClassifier()

Tree_classifier.fit(X,y)
Tree_acc = cross_val_score(Tree_classifier, X, y, cv=10, scoring='accuracy').mean()

Tree_acc
Ran_classifier = RandomForestClassifier(n_estimators=100)

Ran_classifier.fit(X, y)
Ran_acc = cross_val_score(Ran_classifier, X, y, cv=10, scoring='accuracy').mean()

Ran_acc
XGB_classifier = XGBClassifier()

XGB_classifier.fit(X,y)
XGB_acc = cross_val_score(XGB_classifier, X, y, cv=10, scoring='accuracy').mean()

XGB_acc
acc_dict = {'Logistic Regression': round(LogReg_acc, 2), 

           'Support Vectore Classifier': round(SVM_acc, 2), 

           'K-nearest Neighbor': round(Knn_acc, 2), 

           'Decision Tree': round(Tree_acc, 2), 

           'Random Forest': round(Ran_acc, 2),

            'XGB': round(XGB_acc, 2)

           }

print('Accuracy Scores:-')

acc_dict