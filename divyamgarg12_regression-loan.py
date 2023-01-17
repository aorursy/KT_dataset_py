# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
loan = pd.read_csv("/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv")
# Summary of the data

loan.describe()
# First five rows of data

loan.head()
loan = loan.drop(['Loan_ID'], axis = 1)
# To know the structure of columns

loan.info()
# Convert Loan_Amount_Term into factor

loan['Loan_Amount_Term']= loan['Loan_Amount_Term'].astype(object)
# Convert Credit History into factor

loan['Credit_History']= loan['Credit_History'].astype(object)
# Again Struture of columns

loan.info()
# To show the summary of whole datset including factors as well as characters

loan.describe(include='all')
loan.head()
# Importing data visulaization libraries

import matplotlib.pyplot as plt

import seaborn as sns
# Scatterplot between Applicant income and Loan Amount

sns.scatterplot(loan['ApplicantIncome'], loan['LoanAmount'])
# To know the number of Nas

loan.isna().sum()
# To get all factors into a new dataframe

col = [*loan.select_dtypes('object').columns]

col.remove('Loan_Status')
col
plt.figure(figsize=(20,10))



for i, cols in enumerate(col):

    plt.subplot(3,3,i+1)

    sns.countplot(cols, data= loan, hue = 'Loan_Status')
col_num = [*loan.select_dtypes(['int64', 'float64']).columns]

col_num
plt.figure(figsize=(20,10))



for i, cols in enumerate(col_num):

    plt.subplot(3,3,i+1)

    sns.boxplot(x = 'Loan_Status', y = cols, data = loan)
plt.figure(figsize=(20,10))



for i, cols in enumerate(col_num):

    plt.subplot(3,3,i+1)

    sns.distplot(loan.loc[loan[cols].notna(), cols])



# We can see there is plenty of data where Coapplicant income is 0
# Replacing Y to 0 and N to 1

loan.Loan_Status.replace({'Y': 0, "N" : 1}, inplace = True)
loan['Loan_Status'] = loan['Loan_Status']. astype(int)
# Get dummies of all factors

loan_dummies = pd.get_dummies(loan, drop_first = True)
from sklearn.impute import SimpleImputer
simImp = SimpleImputer()

loan_imp = pd.DataFrame(simImp.fit_transform(loan_dummies), columns = loan_dummies.columns)
loan_imp.head()
loan_imp.info()
# As there are large number of 0 in coapplicant income, we can check if it is important in model or not



Coapplicant_0 = np.where(loan_imp['CoapplicantIncome']==0, 1,0)
sns.countplot(y = Coapplicant_0, hue = loan_imp.Loan_Status)



# Loan_Status = 0 means Yes, 1 means No

# Coapplicant_0 = 0 means not equal to 0 and 1 means 0



# We can see regardless of coapplicant income loan has been rejected
from sklearn.model_selection import train_test_split



X , y = loan_imp.drop('Loan_Status', axis = 1), loan_imp.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 150, stratify = y)
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression
logit = LogisticRegressionCV()

logit.fit(X_train, y_train)
logit_pred = logit.predict(X_test)

logit_pred
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, logit_pred))

confusion_matrix(y_test, logit_pred)
sgd_clf = SGDClassifier()

sgd_clf.fit(X_train, y_train)

sgd_pred = sgd_clf.predict(X_test)

print(accuracy_score(y_test, sgd_pred))

confusion_matrix(y_test, sgd_pred)