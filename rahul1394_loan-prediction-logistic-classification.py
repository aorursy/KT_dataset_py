import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,roc_auc_score

from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.api as sms

from sklearn.preprocessing import StandardScaler,LabelEncoder
loan = pd.read_csv('../input/analytics-vidhya-loan-prediction/train.csv')
loan.head()
loan.info()
pd.DataFrame([loan.isnull().sum(),loan.isnull().sum()/loan.isnull().count() * 100]).T
loan.describe(include=np.object)
loantest= pd.read_csv('../input/analytics-vidhya-loan-prediction/test.csv')

loantest.head()
loantest.info()
objcols = loan.columns[loan.dtypes == np.object]

objcols
for col in objcols:

    if (loan[col].isnull().sum() > 0) :

        loan[col].fillna(loan[col].mode()[0],inplace=True)
intcols = loan.columns[loan.dtypes != np.object]

intcols
for col in intcols:

    if (loan[col].isnull().sum() > 0) :

        loan[col].fillna(loan[col].median(),inplace=True)
objcols = loantest.columns[loantest.dtypes == np.object]

for col in objcols:

    if (loantest[col].isnull().sum() > 0) :

        loantest[col].fillna(loantest[col].mode()[0],inplace=True)
intcols = loantest.columns[loantest.dtypes != np.object]

for col in intcols:

    if (loantest[col].isnull().sum() > 0) :

        loantest[col].fillna(loantest[col].median(),inplace=True)
loan['Loan_Status'] = loan['Loan_Status'].map({'Y':1,'N':0})
sns.distplot(np.log1p(loan.CoapplicantIncome))
dummyTrain = pd.get_dummies(loan.drop(['Loan_Status','Loan_ID','Gender','ApplicantIncome','Loan_Amount_Term'],axis=1))

dummyTrain.head()
sc = StandardScaler()
scaledTrain = pd.DataFrame(sc.fit_transform(dummyTrain),columns=dummyTrain.columns)

scaledTrain['Loan_Status'] = loan['Loan_Status']

scaledTrain.head()
dummyTest = pd.get_dummies(loantest.drop(['Loan_ID','Gender','ApplicantIncome','Loan_Amount_Term'],axis=1))

scaledTest = pd.DataFrame(sc.fit_transform(dummyTest),columns=dummyTest.columns)

scaledTest.head()
x = scaledTrain.drop('Loan_Status',axis=1)

y = scaledTrain['Loan_Status']
lor = LogisticRegression()
lor.fit(x,y)
ypred = lor.predict(scaledTest)

ypred
submission = pd.DataFrame({'Loan_ID':loantest.Loan_ID,'Loan_Status':ypred})

submission['Loan_Status'] = submission['Loan_Status'].map({1:'Y',0:'N'})

submission.head()
plt.figure(figsize=(15,10))

sns.heatmap(scaledTrain.corr(),annot=True)
# result = sms.Logit(y,x).fit()

# result.summary()