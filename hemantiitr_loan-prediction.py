import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from pylab import rcParams

import seaborn  as sb
%matplotlib inline

rcParams['figure.figsize']=7,5

plt.style.use('seaborn-whitegrid')
data = pd.read_csv('../input/loan-prediction.csv') 

data.head() #overview of our dataset

data.info()
data.isnull().sum()



data.describe()
sb.distplot(data['ApplicantIncome'])
sb.distplot(data[data['ApplicantIncome']<22000]['ApplicantIncome'])
data[data['ApplicantIncome']<22000]['ApplicantIncome'].describe()
# std decreased nearly half.So,here we are assuming that 8 values are outliers.

data[data['ApplicantIncome']>=22000]['ApplicantIncome'].index
data=data.drop([126, 155, 171, 183, 185, 333, 409, 443],axis=0)
sb.distplot(data['CoapplicantIncome'])
sb.distplot(data[data['CoapplicantIncome']<12000]['CoapplicantIncome'])
data[data['CoapplicantIncome']<12000]['CoapplicantIncome'].describe()
data[data['CoapplicantIncome']>=12000]['CoapplicantIncome'].index
data=data.drop([402, 417, 581, 600],axis=0)
data.boxplot(column = 'LoanAmount',showmeans=True)
rcParams['figure.figsize']=7,5



plt.subplot(1,2,1)

data['modified_LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean() )

data.boxplot(column='modified_LoanAmount',showmeans=True)

plt.title('Filled with mean')



plt.subplot(1,2,2)

data['modified_LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].median() )

data.boxplot(column='modified_LoanAmount',showmeans=True)

plt.title('Filled with Median')

#we are filling with median.

data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].mean() )

data.drop(['modified_LoanAmount'],axis=1,inplace=True)

sb.distplot(data['LoanAmount'])
## After data visualization,most of velues are arround 360(median) so, median will be better for missing velues.

data['Loan_Amount_Term']=data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median() )

sb.distplot(data['Loan_Amount_Term'])
#and also for credit history median fits best.

data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].median() )

sb.distplot(data['Credit_History'])
#For all categorical variables,we are using Mode to remove our Nan from Dataset



data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])

data['Married']=data['Married'].fillna(data['Married'].mode()[0])

data['Dependents']=data['Dependents'].fillna(data['Dependents'].mode()[0])

data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data.isnull().sum()
sb.countplot(x='Gender',data=data,hue='Loan_Status')
sb.countplot(x='Dependents',data=data,hue='Loan_Status')
sb.countplot(x='Education',data=data,hue='Loan_Status')

sb.countplot(x='Married',data=data,hue='Loan_Status')
data = pd.get_dummies(data, columns = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status'],drop_first = True)

data.head()
import sklearn

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier
Y=data['Loan_Status_Y']

X=data.drop(columns=['Loan_Status_Y','Loan_ID'])

# split the train and test dataset where test set is 30% of dataset

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3) 
model= RandomForestClassifier(max_depth=5) 

model=model.fit(xtrain,ytrain) 
model.score(xtest,ytest)
from sklearn.metrics import confusion_matrix



ypred=model.predict(xtest)



confusion_matrix(ytest,ypred)