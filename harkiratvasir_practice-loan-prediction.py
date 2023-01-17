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
import pandas as pd
data = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
data.head()
data.describe()
data.info()
data.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(data.ApplicantIncome)
sns.boxplot(y = data.ApplicantIncome,x = data.Education)
data['LoanAmount'].hist(bins =30 )
data.boxplot('LoanAmount')
data['Credit_History'].value_counts().plot(kind='bar')
data.pivot_table(index= 'Credit_History',values = 'Loan_Status',aggfunc = lambda x:x.map({'Y':1,'N':0}).mean()).plot(kind = 'bar')

plt.xlabel('Credit_history')

plt.ylabel('Problity of getting the loan')
data['Property_Area'].value_counts().plot(kind = 'bar')
data.pivot_table(index= 'Property_Area',values = 'Loan_Status',aggfunc = lambda x:x.map({'Y':1,'N':0}).mean()).plot(kind = 'bar')

plt.xlabel('Property_Area')

plt.ylabel('Problity of getting the loan')
data['Self_Employed'].value_counts().plot(kind='bar')
data.pivot_table(index= 'Self_Employed',values = 'Loan_Status',aggfunc = lambda x:x.map({'Y':1,'N':0}).mean()).plot(kind = 'bar')

plt.xlabel('Self_Employed')

plt.ylabel('Problity of getting the loan')
pd.crosstab(data.Loan_Status,data.Gender).plot(kind='bar',stacked =True)
data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace = True)
data.isnull().sum()
data['Self_Employed'].value_counts()  

data['Self_Employed'].fillna('No',inplace = True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

data['Married'].fillna(data['Married'].mode()[0], inplace=True)

data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)

data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['TotalIncome'] = np.log(data.ApplicantIncome + data.CoapplicantIncome)
sns.distplot(data['TotalIncome'])
data = data.drop(["Loan_ID","ApplicantIncome","CoapplicantIncome","LoanAmount"],axis=1)
data.head()
X = data.drop('Loan_Status',1)

y = data.Loan_Status
X = pd.get_dummies(X)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter = 10000)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)*100