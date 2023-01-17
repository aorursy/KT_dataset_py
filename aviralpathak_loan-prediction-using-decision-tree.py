# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing the libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df_train.head()



#There are some categorical data
df_train.info()



#there is missing data
#lets see how many get the loan

df_train['Loan_Status'].value_counts().plot.bar()
#percentage of applicant from certain property area



area = df_train['Property_Area'].value_counts()



plt.pie(area, labels = area.index,autopct='%1.1f%%')



plt.show()
#loan status according to the property area



cb = pd.crosstab(df_train['Property_Area'],df_train['Loan_Status'])



cb.plot.bar(stacked=False)
#loan-amount from certain property area, with laon status

 

sns.boxplot(x="Property_Area", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount from the person who are married or not, with laon status



sns.boxplot(x="Married", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount by the person who have number of dependents, with laon status



sns.boxplot(x="Dependents", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount by the person who is educated or not, with laon status





sns.boxplot(x="Education", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount by the person who is self-employed or not, with laon status





sns.boxplot(x="Self_Employed", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount by the person with its credit history, with laon status





sns.boxplot(x="Credit_History", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount by the person with its own income amount, with laon status





sns.scatterplot(x="ApplicantIncome", y="LoanAmount",hue='Loan_Status',data=df_train)
#loan-amount by the person with Co-applicant income amount, with laon status



sns.scatterplot(x="CoapplicantIncome", y="LoanAmount",hue='Loan_Status',data=df_train)
#Now we will split the categorical columns and numerical columns
#data for categorical data



train_cat = df_train.copy()
train_cat = train_cat.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1)
#checking if there any null value



train_cat.isnull().sum().sort_values(ascending=False)
#fill every missing value with their next value in the same column



train_cat.fillna(method='ffill', inplace=True)
#there is no null value



train_cat.isnull().sum().any()
#data for numerical columns



train_num = df_train.copy()
train_num = train_num.drop(['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status'],axis=1)
train_num.isnull().sum().sort_values(ascending=False)
#fill every missing value with its previous value in the same column





train_num.fillna(method='bfill', inplace=True)
train_num.isnull().sum().any()
#now concat both the columns



train_con = pd.concat([train_cat,train_num],axis=1)


X = train_con.drop('Loan_Status', axis=1)

y = train_con.Loan_Status
#get_dummies() function convert categorical variable into dummy variables.



X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
#splitting data into test and train



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=101)
#using decision tree model



algo = DecisionTreeClassifier()

algo.fit(X_train, y_train)
predict_test = algo.predict(X_test)
accuracy_score(y_test,predict_test)