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
df=pd.read_csv('/kaggle/input/traincsv/train_new.csv')

df.head()
df.isnull().sum()
df.describe()
X=df.drop(['Loan_Status','Loan_ID'], axis=1)

y=df['Loan_Status']

print(X)
X.info()
X['Gender'].value_counts()
X['Gender'].fillna("Male", inplace=True)

X.info()
X['Married'].value_counts()
X['Married'].fillna("Yes", inplace=True)

X.info()
X['Education'].value_counts()

X.info()
X['Dependents'].value_counts()

X['Dependents'].fillna(0,inplace=True)   

X.info()
X['Self_Employed'].value_counts() 
X['Self_Employed'].fillna('No',inplace=True)

X.info()
mean_loan=X['LoanAmount'].mean()

X['LoanAmount'].fillna(mean_loan,inplace=True)

X.isnull().sum()
mean_loan=X['LoanAmount'].mean()

X['LoanAmount'].fillna(mean_loan,inplace=True)

X.isnull().sum()
X['Loan_Amount_Term'].fillna(X['Loan_Amount_Term'].mean(),inplace=True)

X['Credit_History'].fillna(X['Credit_History'].mean(),inplace=True)

X.isnull().sum()
X=pd.get_dummies(X)

X.head()
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)
y_train.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



lr.fit(X_train,y_train)
from sklearn.svm import SVC

svc = SVC()



svc.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

dtf = DecisionTreeClassifier()

dtf.fit(X_train, y_train)
from sklearn.naive_bayes import GaussianNB

n_b = GaussianNB()

n_b.fit(X_train, y_train)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()  

knn.fit(X_train, y_train)
print(lr.score(X_train, y_train),"   ", lr.score(X_test, y_test))

print(dtf.score(X_test, y_test))

print(n_b.score(X_test, y_test))

print(knn.score(X_test, y_test))

print(svc.score(X_test, y_test))
gender=input("What is your gender (Male or Female):")

married=input("Married (Enter Yes or No):")

dependents=int(input("dependents value (Enter 0 or 1 or 2 or 3+):"))

Education=input("enter your education (Enter Graduate or Not Graduate)")

SelfEmployed=input("Self Employed (Enter Yes or No):")

Applicantincome=int(input("enter applicant income (Enter between 150 to 81000)"))

coapplicantincome=int(input("enter co applicant income(Enter ):"))

loanamount=int(input("enter loan amount:"))

loanamountterm=int(input("enter loan amount term:"))

credithistory=int(input("enter credit history:"))

propertyarea=input("enter property area:")



data = [[gender,married,dependents,Education,SelfEmployed,Applicantincome,coapplicantincome,loanamount,loanamountterm,credithistory,propertyarea]]
newdf = pd.DataFrame(data, columns = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])
newdf = pd.get_dummies(newdf)
missing_cols = set( X_train.columns ) - set( newdf.columns )
for c in missing_cols:

    newdf[c] = 0
newdf = newdf[X_train.columns]
yp=n_b.predict(X_test[0:5])

print(yp)
if (yp[0]=='Y'):

    print("Your Loan is approved, Please contact at HDFC Bank Any Branch for further processing")

else:

    print("Sorry ! Your Loan is not approved")
yp1=n_b.predict(newdf)

yp2=lr.predict(newdf)

yp3=dtf.predict(newdf)

yp4=knn.predict(newdf)

yp5=svc.predict(newdf)

print(yp1)



print(yp2)

print(yp3)

print(yp4)

print(yp5)
from sklearn.metrics import confusion_matrix 

y_predict=lr.predict(X_test)

results = confusion_matrix(y_test, y_predict) 

print('Confusion Matrix :')

print(results)

for i in range(5):

    print(y_predict[i],y_test.iloc[i])

for i in range(5):

    print(y_predict[i],"     ", y_test.iloc[i])
print(lr.predict_proba(X_test[0:5]))