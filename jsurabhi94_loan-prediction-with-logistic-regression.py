import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/train_loan.csv')
data.head()
data.shape
data.info()
data.isnull().sum()
data['LoanAmount'].fillna(data.LoanAmount.median(), inplace = True)

data['Loan_Amount_Term'].fillna(data.Loan_Amount_Term.mode()[0], inplace = True)

data['Gender'].fillna(data.Gender.mode()[0], inplace = True)

data['Married'].fillna(data.Married.mode()[0], inplace = True)

data['Dependents'].fillna(data.Dependents.mode()[0], inplace = True)

data['Self_Employed'].fillna(data.Self_Employed.mode()[0], inplace = True)

data['Credit_History'].fillna(data.Credit_History.mode()[0], inplace = True)

data.dtypes
data['ApplicantIncome'] = data['ApplicantIncome'].astype('float64')

data.dtypes
data.Loan_Status.value_counts()
data.Loan_Status.value_counts(normalize = True).plot(kind = 'bar').grid(True, axis = 'y')
plt.figure(figsize = (15,15))



plt.subplot(3,2,1)

data.Gender.value_counts(normalize = True).plot.bar(title = 'Gender').grid(True, axis = 'y')

plt.xticks(rotation = 45)



plt.subplot(3,2,2)

data.Married.value_counts(normalize = True).plot.bar(title = 'Married').grid(True, axis = 'y')

plt.xticks(rotation = 45)



plt.subplot(3,2,3)

data.Education.value_counts(normalize = True).plot.bar(title = 'Education').grid(True, axis = 'y')

plt.xticks(rotation = 30)



plt.subplot(3,2,4)

data.Dependents.value_counts(normalize = True).plot.bar(title= 'Dependants').grid(True, axis = 'y')

plt.xticks(rotation = 45)



plt.subplot(3,2,5)

data.Self_Employed.value_counts(normalize = True).plot.bar(title = 'Self_Employed').grid(True, axis = 'y')

plt.xticks(rotation = 45)



plt.subplot(3,2,6)

data.Property_Area.value_counts(normalize = True).plot.bar(title = 'Property-Area').grid(True, axis = 'y')

plt.xticks(rotation = 45)
plt.figure(figsize = (15,10))

plt.subplot(231)

sns.boxplot(y= data.ApplicantIncome)



plt.subplot(232)

sns.boxplot(y= data.CoapplicantIncome)



plt.subplot(233)

sns.boxplot(y= data.LoanAmount)



plt.subplot(234)

sns.distplot(data.ApplicantIncome)



plt.subplot(235)

sns.distplot(data.CoapplicantIncome)



plt.subplot(236)

sns.distplot(data.LoanAmount)
data.Credit_History.value_counts(normalize = True).plot.bar(title = 'Credit History', figsize = (7,5)).grid(True, axis = 'y')
pd.crosstab(data.Gender, data.Loan_Status)
pd.crosstab(data.Gender, data.Loan_Status).plot.bar(figsize = (5,5))
pd.crosstab(data.Married, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))



pd.crosstab(data.Dependents, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))



pd.crosstab(data.Education, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))



pd.crosstab(data.Self_Employed, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))
sns.boxplot(y= 'ApplicantIncome', x= 'Loan_Status', data = data)
bins = [0,2500,4000,6000,81000]

group = ['low', 'average', 'high', 'very high']

data['ApplicantIncome new'] = pd.cut(data['ApplicantIncome'], bins, labels = group)

pd.crosstab(data['ApplicantIncome new'], data['Loan_Status'], normalize = True).plot.bar(figsize = (5,5), stacked = True)
sns.boxplot(y= 'CoapplicantIncome', x= 'Loan_Status', data = data)
bins = [0,1000,2000,4000,42000]

group = ['low', 'average', 'high', 'very high']

data['CoapplicantIncome new'] = pd.cut(data['CoapplicantIncome'], bins, labels = group)

pd.crosstab(data['CoapplicantIncome new'], data['Loan_Status'], normalize = True).plot.bar(figsize = (5,5), stacked = True)
data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']

sns.boxplot(y= 'Total_Income', x= 'Loan_Status', data = data)
bins = [0,2500,5000,10000,81000]

groups = ['low', 'average', 'high', 'very high']

data['Total_Income_new'] = pd.cut(data['Total_Income'], bins, labels = group)

pd.crosstab(data.Total_Income_new, data.Loan_Status, normalize = True).plot.bar(figsize = (5,5))
pd.crosstab(data.Credit_History, data.Loan_Status).plot.bar(stacked = True, figsize = (5,5))
bin = [0,100,200,700]

group = ['low', 'average', 'high']

data['loanamount'] = pd.cut(data['LoanAmount'], bin, labels = group)

pd.crosstab(data.loanamount, data.Loan_Status, normalize = True).plot.bar(stacked = True, figsize = (5,5))
data['Loan_Status'] = data['Loan_Status'].map({'N': 0, 'Y': 1})

data['Dependents'] = data['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
train = data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']]

sns.heatmap(train.corr(), annot = True, cmap="BuPu")
X = data.drop(['Loan_ID', 'Loan_Status', 'ApplicantIncome new', 'CoapplicantIncome new',

               'Total_Income_new', 'loanamount', 'ApplicantIncome', 'CoapplicantIncome'], axis = 1)

y = data.Loan_Status

X = pd.get_dummies(X)
X.columns
skf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)

for train_index, test_index in skf.split(X,y):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

model = LogisticRegression(random_state = 1)

model.fit(X_train, y_train)

model.predict(X_test)

print(accuracy_score(y_test, model.predict(X_test)))
test = pd.read_csv('../input/test_loan.csv')

test_original = test.copy()

test.head()
test.shape
test['LoanAmount'].fillna(test.LoanAmount.median(), inplace = True)

test['Loan_Amount_Term'].fillna(test.Loan_Amount_Term.mode()[0], inplace = True)

test['Gender'].fillna(test.Gender.mode()[0], inplace = True)

test['Married'].fillna(test.Married.mode()[0], inplace = True)

test['Dependents'].fillna(test.Dependents.mode()[0], inplace = True)

test['Self_Employed'].fillna(test.Self_Employed.mode()[0], inplace = True)

test['Credit_History'].fillna(test.Credit_History.mode()[0], inplace = True)
test['ApplicantIncome'] = test['ApplicantIncome'].astype('float64')

test['Dependents'] = test['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})

test['Total_Income'] =test['ApplicantIncome'] + test['CoapplicantIncome']

test = test.drop(['Loan_ID','ApplicantIncome', 'CoapplicantIncome'], axis = 1)
test = pd.get_dummies(test)
test.columns
test.isna().sum()
prediction = model.predict(test)

submission = pd.DataFrame({'Loan_ID': test_original['Loan_ID'], 'Loan_Status': prediction})

submission['Loan_Status'].replace(0, 'N', inplace = True)

submission['Loan_Status'].replace(1, 'Y', inplace = True)
submission.head()
submission['Loan_Status'].value_counts(normalize = True).plot(kind = 'bar')