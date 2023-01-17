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
#loading train dataset

train_data=pd.read_csv("/kaggle/input/bank-loan2/madfhantr.csv")

train_data.head()
#loading test dataset

test_data=pd.read_csv("/kaggle/input/bank-loan2/madhante.csv")

test_data.head()
#No. of null values

train_data.isnull().sum()
#filling missing values for train_data

train_data['Gender'].fillna(train_data['Gender'].mode().values[0],inplace=True)

train_data['Married'].fillna(train_data['Married'].mode().values[0],inplace=True)

train_data['Dependents'].fillna(train_data['Dependents'].mode().values[0],inplace=True)

train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode().values[0],inplace=True)

train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mean(),inplace=True)

train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(),inplace=True)

train_data['Credit_History'].fillna(train_data['Credit_History'].mean(),inplace=True)

train_data.isnull().sum()
#No. of null values of test_data

test_data.isnull().sum()
#filling missing values for test_data

test_data['Gender'].fillna(test_data['Gender'].mode().values[0],inplace=True)

test_data['Married'].fillna(test_data['Married'].mode().values[0],inplace=True)

test_data['Dependents'].fillna(test_data['Dependents'].mode().values[0],inplace=True)

test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode().values[0],inplace=True)

test_data['Loan_Amount_Term'].fillna(test_data['Loan_Amount_Term'].mean(),inplace=True)

test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(),inplace=True)

test_data['Credit_History'].fillna(test_data['Credit_History'].mean(),inplace=True)

test_data.isnull().sum()
from sklearn.preprocessing import LabelEncoder

encoder  = LabelEncoder()

train_data['Gender']=encoder.fit_transform(train_data['Gender'])

train_data['Married']=encoder.fit_transform(train_data['Married'])

train_data['Education']=encoder.fit_transform(train_data['Education'])

train_data['Self_Employed']=encoder.fit_transform(train_data['Self_Employed'])

train_data['Property_Area']=encoder.fit_transform(train_data['Property_Area'])

train_data['Loan_Status']=encoder.fit_transform(train_data['Loan_Status'])

train_data['Loan_ID']=encoder.fit_transform(train_data['Loan_ID'])

train_data['Dependents']=encoder.fit_transform(train_data['Dependents'])

train_data.head()
# Now LabelEncoding test_data

test_data['Loan_ID']=encoder.fit_transform(test_data['Loan_ID'])

test_data['Gender']=encoder.fit_transform(test_data['Gender'])

test_data['Married']=encoder.fit_transform(test_data['Married'])

test_data['Education']=encoder.fit_transform(test_data['Education'])

test_data['Self_Employed']=encoder.fit_transform(test_data['Self_Employed'])

test_data['Property_Area']=encoder.fit_transform(test_data['Property_Area'])

test_data['Dependents']=encoder.fit_transform(test_data['Dependents'])

test_data.head()
test_data.shape
train_data.shape
# correlation of dataset

train_data.corr()
from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report



X = train_data.iloc[:, 1:-1].values

y = train_data.iloc[:, -1].values

# Splitting the dataset into train and test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)

model = DecisionTreeClassifier(criterion = "entropy", random_state = 24,max_depth=3, min_samples_leaf=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix: ", confusion_matrix(y_test, y_pred)) 

print ("Accuracy : ", accuracy_score(y_test,y_pred)*100) 

print("Report : ", classification_report(y_test, y_pred)) 
sample=pd.read_csv("/kaggle/input/bank-loan2/sample_submission_49d68Cx.csv")

sample.head()
df1=pd.DataFrame(y_pred,columns=["Loan_Status"])

df1=df1.replace(1,'Y')

df1=df1.replace(0,'N')

df1.head()
sample=sample.drop(columns=['Loan_Status'],axis=1)

sample['Loan_Status']=df1['Loan_Status']

sample.head(20)