# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib as plt

data=pd.read_csv("../input/loanPred_train_data.csv")





# Any results you write to the current directory are saved as output.
data.head(10)
data.describe()
data.info()
data['ApplicantIncome'].hist()
data['ApplicantIncome'].hist(bins=50)
data.boxplot('ApplicantIncome')
data.boxplot('ApplicantIncome',by='Education')
data['LoanAmount'].hist(bins=100)
data.boxplot('LoanAmount')
temp1 = data['Credit_History'].value_counts(ascending=True)

temp2 = data.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:')

print(temp1)
print ('\nProbility of getting loan for each Credit History class:')

print (temp2)
data.apply(lambda x: sum(x.isnull()),axis=0)
#Addin LoanAmount missing values

data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)
#Verify the Missing Values in LoanAmount

data.apply(lambda x: sum(x.isnull()),axis=0)
data.fillna(data.mean(),inplace=True)
#Verify the Missing Values

data.apply(lambda x: sum(x.isnull()),axis=0)
data['Self_Employed'].fillna('No',inplace=True)
#Verify the Missing Values

data.apply(lambda x: sum(x.isnull()),axis=0)
data['LoanAmount_log']=np.log(data['LoanAmount'])



#Histogram

data['LoanAmount_log'].hist()
#combining incomes

data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']
#Lets look at top 5 values of combined incomes

data['TotalIncome'].head(5)
#Log Transformation

data['TotalIncome_log']=np.log(data['TotalIncome'])
#Histogram

data['TotalIncome_log'].hist()
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)

data['Married'].fillna(data['Married'].mode()[0],inplace=True)

data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)

data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)

data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
#Check missing values

data.apply(lambda x: sum(x.isnull()),axis=0)
from sklearn.preprocessing import LabelEncoder

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

le = LabelEncoder()

for i in var_mod:

    data[i] = le.fit_transform(data[i])

data.dtypes 
#Import models from scikit learn module:

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics
#Function for making a classification model and verifying the performance

def classification_model(model, data, predictors, outcome):

  #Fit the model:

  model.fit(data[predictors],data[outcome])

  

  #Make predictions on training set:

  predictions = model.predict(data[predictors])

  

  #Print accuracy

  accuracy = metrics.accuracy_score(predictions,data[outcome])

  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))



  #Fit the model again so that it can be refered outside the function:

  model.fit(data[predictors],data[outcome]) 
outcome_var = 'Loan_Status'

model = LogisticRegression()

predictor_var = ['Credit_History']

classification_model(model, data,predictor_var,outcome_var)
#We can try different combination of variables:

predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']

classification_model(model, data,predictor_var,outcome_var)
model = DecisionTreeClassifier()

predictor_var = ['Credit_History','Gender','Married','Education']

classification_model(model, data,predictor_var,outcome_var)