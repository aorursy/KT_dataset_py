import numpy as np

import pandas as pd

from pandas.api.types import is_numeric_dtype #to check if the numpy data type is numeric or not

from sklearn.naive_bayes import GaussianNB #Naive Bayes classifier

from sklearn.preprocessing import LabelEncoder # to convert the variable values to numeric data type

from sklearn.model_selection import train_test_split # to separate train and test data

from sklearn.metrics import accuracy_score #to check the accuracy

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

    

incomeData = pd.read_csv("../input/income-datatxt/income_data.txt",header=None) #Since the dataset does not have names for the columns, we need to include "header=None". The output is a pandas dataframe.



incomeData.head()

number = LabelEncoder()

# it is necessary to encode the string values to numeric values

for i in range(0,15):

  if not is_numeric_dtype(incomeData[i]):

    incomeData[i] = number.fit_transform(incomeData[i])

incomeData.head()

features = [i for i in range(0,14)] #in this example, we are considering all the features

target = 14

#Split the given dataset to training data and testing data

features_train, features_test, target_train, target_test = train_test_split(incomeData[features],incomeData[target],test_size = 0.30,random_state = 54)

model = GaussianNB()#create the Naive Bayes model object

model.fit(features_train, target_train) #train the model with the training input and output

pred = model.predict(features_test) #test the model with the testing data

con_mat = confusion_matrix(target_test,pred)

print(con_mat)

print(classification_report(target_test,pred))

accuracy = accuracy_score(target_test, pred) 

print("Accuracy:",accuracy)

import numpy as np

import pandas as pd

from pandas.api.types import is_numeric_dtype #to check if the numpy data type is numeric or not

from sklearn.svm import SVC #Support Vector Classifier

from sklearn.preprocessing import LabelEncoder # to convert the variable values to numeric data type

from sklearn.model_selection import train_test_split # to separate train and test data

from sklearn.metrics import accuracy_score #to check the accuracy

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



incomeData = pd.read_csv("../input/income-datatxt/income_data.txt",header=None) #Since the dataset does not have names for the columns, we need to include "header=None". The output is a pandas dataframe.

number = LabelEncoder()

# it is necessary to encode the string values to numeric values

for i in range(0,15):

  if not is_numeric_dtype(incomeData[i]):

    incomeData[i] = number.fit_transform(incomeData[i])





features = [i for i in range(0,14)] #in this example, we are considering all the features

target = 14

#Split the given dataset to training data and testing data

features_train, features_test, target_train, target_test = train_test_split(incomeData[features],incomeData[target],test_size = 0.30,random_state = 54)

svc_classifier = SVC(kernel="rbf")

svc_classifier.fit(features_train,target_train)

pred = svc_classifier.predict(features_test)

c_mat = confusion_matrix(pred,target_test)

print(c_mat)

print(classification_report(pred,target_test))

print("Accuracy:",accuracy_score(pred,target_test))

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

import sklearn.model_selection as sms



incomeData = pd.read_csv("../input/income-datatxt/income_data.txt",header=None) #Since the dataset does not have names for the columns, we need to include "header=None". The output is a pandas dataframe.

number = LabelEncoder()

# it is necessary to encode the string values to numeric values

for i in range(0,15):

  if not is_numeric_dtype(incomeData[i]):

    incomeData[i] = number.fit_transform(incomeData[i])



incomeData.head()

features = [i for i in range(0,14)] #in this example, we are considering all the features

target = 14

#Split the given dataset to training data and testing data

features_train, features_test, target_train, target_test = train_test_split(incomeData[features],incomeData[target],test_size = 0.30,random_state = 54)

params = {'n_estimators':100,'max_depth':4,'random_state':0}

classifier = RandomForestClassifier(**params)

classifier.fit(features_train,target_train)

pred = classifier.predict(features_test)

c_mat = confusion_matrix(pred,target_test)

print(c_mat)

print(classification_report(pred,target_test))

print("Accuracy:",accuracy_score(pred,target_test))

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

import sklearn.model_selection as sms



incomeData = pd.read_csv("../input/income-datatxt/income_data.txt",header=None) #Since the dataset does not have names for the columns, we need to include "header=None". The output is a pandas dataframe.

number = LabelEncoder()

# it is necessary to encode the string values to numeric values

for i in range(0,15):

  if not is_numeric_dtype(incomeData[i]):

    incomeData[i] = number.fit_transform(incomeData[i])



incomeData.head()

features = [i for i in range(0,14)] #in this example, we are considering all the features

target = 14

#Split the given dataset to training data and testing data

features_train, features_test, target_train, target_test = train_test_split(incomeData[features],incomeData[target],test_size = 0.30,random_state = 54)

params = {'n_estimators':100,'max_depth':4,'random_state':0}

classifier = ExtraTreesClassifier(**params)

classifier.fit(features_train,target_train)

pred = classifier.predict(features_test)

c_mat = confusion_matrix(pred,target_test)

print(c_mat)

print(classification_report(pred,target_test))

print("Accuracy:",accuracy_score(pred,target_test))

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_validate

from sklearn.tree import DecisionTreeClassifier

import sklearn.model_selection as sms



incomeData = pd.read_csv("../input/income-datatxt/income_data.txt",header=None) #Since the dataset does not have names for the columns, we need to include "header=None". The output is a pandas dataframe.

number = LabelEncoder()

# it is necessary to encode the string values to numeric values

for i in range(0,15):

  if not is_numeric_dtype(incomeData[i]):

    incomeData[i] = number.fit_transform(incomeData[i])



incomeData.head()

features = [i for i in range(0,14)] #in this example, we are considering all the features

target = 14

#Split the given dataset to training data and testing data

features_train, features_test, target_train, target_test = train_test_split(incomeData[features],incomeData[target],test_size = 0.30,random_state = 54)

params = {'random_state':0,'max_depth':4}

classifier = DecisionTreeClassifier(**params)

classifier.fit(features_train,target_train)

pred = classifier.predict(features_test)

c_mat = confusion_matrix(pred,target_test)

print(c_mat)

print(classification_report(pred,target_test))

print("Accuracy:",accuracy_score(pred,target_test))
