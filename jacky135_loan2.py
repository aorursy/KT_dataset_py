# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/loanprediction/train_ctrUa4K.csv')

test = pd.read_csv('/kaggle/input/loanprediction/test_lAUu6dG.csv')
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace = True)

test['Dependents'].fillna(test['Dependents'].mode()[0],inplace = True)

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace = True)

test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace = True)

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean(),inplace = True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(),inplace = True)

train['LoanAmount'].fillna(train['LoanAmount'].mode()[0],inplace = True)

test['LoanAmount'].fillna(test['LoanAmount'].mode()[0],inplace = True)

train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace = True)

test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace = True)

train = pd.get_dummies(train,columns = ['Gender','Married','Dependents','Credit_History','Education','Self_Employed','Property_Area'])

test = pd.get_dummies(test,columns = ['Gender','Married','Dependents','Credit_History','Education','Self_Employed','Property_Area'])
for  minmax in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']:

    mini,maxa= train[minmax].min(),train[minmax].max()

    train.loc[:,minmax] = (train[minmax]-mini)/(maxa-mini)

   
for  minmax in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']:

    mini,maxa = test[minmax].min(),test[minmax].max()

    test.loc[:,minmax] = (test[minmax]-mini)/(maxa-mini)

   
train
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

train['Loan_Status'] = labelencoder.fit_transform(train['Loan_Status'])
y = train['Loan_Status']

y_train = to_categorical(y, num_classes = 2)
y_train


train.drop(['Loan_Status','Loan_ID'],axis = 1,inplace = True)

id1 = test['Loan_ID']

test.drop(['Loan_ID'],axis = 1,inplace = True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.33, random_state=42)
model = Sequential()

model.add(Dense(128,input_dim=X_train.shape[1]))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.45))

model.add(Dense(2, activation = "softmax"))

optimizer = RMSprop(lr=0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(X_train, y_train,epochs =100, 

         validation_data = (X_test, y_test),callbacks = [checkpoint], verbose = 2)
result = model.predict(test)

result
result = np.argmax(result,axis = 1)
result = pd.Series(result,name="Loan_Status")

submission = pd.concat([pd.Series(id1,name = "Loan_ID"),result],axis = 1)
submission['Loan_Status'] = submission['Loan_Status'].map({0:'N',1:'Y'})
submission
submission.to_csv("Loan.csv",index=False)