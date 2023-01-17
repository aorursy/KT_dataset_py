# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/assignment1"))

os.chdir("../input/assignment1")
# Any results you write to the current directory are saved as output.
#import os
#os.chdir("C:\\Users\\Prince\\Desktop\\Kaggle")
import pandas as pd
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df['Gender'].mode()
df['Gender'].fillna(method = 'ffill', inplace = True)
df_test['Gender'].fillna(method = 'ffill', inplace = True)

df['Married'].mode()
df['Married'].fillna(method = 'ffill', inplace = True,limit = 1)
df_test['Married'].fillna(method = 'ffill', inplace = True)

df['Dependents'].mode()
df['Dependents'].fillna(method = 'ffill', inplace = True)
df_test['Dependents'].fillna(method = 'ffill', inplace = True)

df['Education'].mode()
df['Education'].fillna(method = 'ffill', inplace = True)
df_test['Education'].fillna(method = 'ffill', inplace = True)

df['Self_Employed'].mode()
df['Self_Employed'].fillna(method = 'ffill', inplace = True)
df_test['Self_Employed'].fillna(method = 'ffill', inplace = True)

df['ApplicantIncome'].mode()
df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean(), inplace = True)
df_test['ApplicantIncome'].fillna(df['ApplicantIncome'].mean, inplace = True)

df['CoapplicantIncome'].mode()
df['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean(), inplace = True)
df_test['CoapplicantIncome'].fillna(df['CoapplicantIncome'].mean, inplace = True)

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace = True)
df_test['LoanAmount'].fillna(df_test['LoanAmount'].mean(),inplace = True)


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace = True)
df_test['Loan_Amount_Term'].fillna(df_test['Loan_Amount_Term'].mean(),inplace = True)

df['Credit_History'].mode()
df['Credit_History'].fillna(method = 'ffill', inplace = True)
df_test['Credit_History'].fillna(method = 'ffill', inplace = True)



df_test.isnull().sum()
df.isnull().sum()


x_train = df.drop(['Loan_ID','Loan_Status'],axis = 1)

x_test = df_test.drop(['Loan_ID'],axis = 1)

y_train = df['Loan_Status']


x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y_train = labelencoder.fit_transform(y_train)


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


