# Loading libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")
# importing data

traindata = pd.read_csv('/kaggle/input/titanic/train.csv')

testdata = pd.read_csv('/kaggle/input/titanic/test.csv')
traindata.head()
# describing data

traindata.describe()
# encoding string data columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

traindata['Sex'] = le.fit_transform(traindata['Sex'])

testdata['Sex'] = le.fit_transform(testdata['Sex'])

traindata['Embarked'] = le.fit_transform(traindata['Embarked'].astype(str))

testdata['Embarked'] = le.fit_transform(testdata['Embarked'].astype(str))
# feature selection

X = traindata[['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']].values

y = traindata.iloc[:,1].values

X_real_test = testdata[['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked']].values
# checking missing values

print(traindata.isnull().sum())
# handling missing values

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer()

X_transformed = imputer.fit_transform(X)

X_real_test = imputer.fit_transform(X_real_test)
# Splitting the dataset into the Training set and Test set for evaluation

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_transformed,y,test_size=0.25,random_state=1313)
# Feature scaling for better performance

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)

X_real_test = sc.fit_transform(X_real_test)
# SVM classifier with Gaussian RBF kernel

from sklearn.svm import SVC

classifier = SVC(kernel='rbf',random_state=0)

classifier.fit(X_train,y_train)

# predict with splitted test data

y_pred = classifier.predict(X_test)

# evaluation using confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
# generating predictions on provided test data

y_pred_test = classifier.predict(X_real_test)

pid = testdata[['PassengerId']].values

res = np.expand_dims(y_pred_test,axis=1)

f = np.hstack((pid,res))

df = pd.DataFrame(f, columns = ['PassengerId', 'Survived']) 

df.to_csv('gender_submission.csv', index=False)