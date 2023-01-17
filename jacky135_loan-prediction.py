import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import AdaBoostClassifier

from catboost import CatBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,Activation

from keras.optimizers import RMSprop,SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import ModelCheckpoint

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/loanprediction/train_ctrUa4K.csv')

test = pd.read_csv('/kaggle/input/loanprediction/test_lAUu6dG.csv')
train
train.info()
train.isnull().sum()
train.drop(['Loan_ID'],axis= 1,inplace = True)
train
train[train['Loan_Status'] == 'N']['Loan_Amount_Term'].value_counts()
def impute_credit(col):

    credit=col[0]

    status = col[1]

    if pd.isnull(credit):

        if status == 'Y':

         return 1.0

        else :

         return 0.0

    else :

     return credit
train['Credit_History'] = train[['Credit_History','Loan_Status']].apply(impute_credit,axis = 1)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace = True)
def impute_loan(col):

    loan=col[0]

    status = col[1]

    if pd.isnull(loan):

        if status == 'Y':

         return 144

        else :

         return 151

    else :

     return loan
train['LoanAmount'] = train[['LoanAmount','Loan_Status']].apply(impute_loan,axis = 1)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace = True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(),inplace = True)
train['Married'].fillna(train['Married'].mode()[0],inplace = True)

test['Married'].fillna(test['Married'].mode()[0],inplace = True)
train['Gender'].fillna(train['Gender'].mode()[0],inplace = True)

test['Gender'].fillna(test['Gender'].mode()[0],inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)

test['Dependents'].fillna(test['Dependents'].mode()[0],inplace = True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)

test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace = True)
test['LoanAmount'].fillna(test['LoanAmount'].mode()[0],inplace = True)

test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace = True)
test.info()
test[train['Loan_Status']=='N']['ApplicantIncome'].mean()
train[['Loan_Amount_Term','Credit_History']]
train.info()
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 
train['Gender']= label_encoder.fit_transform(train['Gender'])

train['Married']= label_encoder.fit_transform(train['Married'])

train['Education']= label_encoder.fit_transform(train['Education'])

train['Dependents']= label_encoder.fit_transform(train['Dependents'])

train['Loan_Status']= label_encoder.fit_transform(train['Loan_Status'])

train['Self_Employed']=label_encoder.fit_transform(train['Self_Employed'])

test['Gender']= label_encoder.fit_transform(test['Gender'])

test['Married']= label_encoder.fit_transform(test['Married'])

test['Education']= label_encoder.fit_transform(test['Education'])

test['Dependents']= label_encoder.fit_transform(test['Dependents'])

test['Self_Employed']=label_encoder.fit_transform(test['Self_Employed'])

train['Property_Area']= label_encoder.fit_transform(train['Property_Area'])
test['Property_Area']= label_encoder.fit_transform(test['Property_Area'])
train.drop(['Married','Education','Dependents'],axis = 1 ,inplace= True)
test.drop(['Married','Education','Dependents'],axis = 1 ,inplace= True)
id1 = test['Loan_ID']
test.drop(['Loan_ID'],axis = 1 ,inplace = True)
test
train
y = train['Loan_Status']

train.drop(['Loan_Status'],axis = 1 ,inplace = True)
min_max = preprocessing.MinMaxScaler().fit(train)

train_minmax  = min_max.transform(train)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_minmax, y, test_size=0.33, random_state=42)
from sklearn.metrics import accuracy_score

model = CatBoostClassifier(iterations=500)

model.fit(train_minmax,y)

y_predict = model.predict(X_test)

accuracy = accuracy_score(y_predict,y_test)

print(accuracy)


test_minmax  = min_max.transform(test)
y_pred = model.predict(test_minmax)
Y_pred = pd.Series(y_pred,name="Loan_Status")

submission = pd.concat([pd.Series(id1,name="Loan_ID"),Y_pred],axis = 1)

def impute_credit(col):

    credit=col

    if col == 1:

         return 'Y'

    else :

         return 'N'
submission['Loan_Status'] = submission['Loan_Status'].apply(impute_credit)
submission['Loan_Status'].value_counts()
submission.to_csv("loan.csv",index=False)