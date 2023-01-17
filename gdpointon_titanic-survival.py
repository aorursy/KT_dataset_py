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
#survival data

survival_index = pd.read_csv("../input/titanic/gender_submission.csv")



#test data

test = pd.read_csv("../input/titanic/test.csv")

test.head()



#training data

train = pd.read_csv("../input/titanic/train.csv")

train.head()

#merging survival index with train data

train = pd.merge(train, survival_index, how = 'left')

train.head()





#merging survival index with test dat

test = pd.merge(test, survival_index, how = 'left')

test.head()
#handle missing age data

#https://datascience.stackexchange.com/questions/51890/how-to-use-simpleimputer-class-to-replace-missing-values-with-mean-values-using



#simple imputer option

#from sklearn.impute import SimpleImputer

#imp = SimpleImputer(missing_values=np.nan, strategy='mean')

#imp.fit(train[['Age']])

#train['Age'] = imp.transform(train[['Age']]).ravel()



#multiple imputer option

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



mimp = IterativeImputer(max_iter=20, random_state=27)

mimp.fit(train[['Age']])

train['Age'] = mimp.transform(train[['Age']]).ravel()



mimp.fit(test[['Age']])

test['Age'] = mimp.transform(test[['Age']]).ravel()
#DUMMY CODING

#create dummy codes for embark; queenstown vs. cherbourg

train['embark_qc'] = np.where(train['Embarked'] == "C", 1, 0)

test['embark_qc'] = np.where(test['Embarked'] == "C", 1, 0)



#create dummy codes for embark; queenstown vs. southampton

train['embark_qs'] = np.where(train['Embarked'] == "S", 1, 0)

test['embark_qs'] = np.where(test['Embarked'] == "S", 1, 0)



#create dummy codes for Pclass; pclass 1 vs pclass 2

train['pclass_2'] = np.where(train['Pclass'] == 2, 1, 0)

test['pclass_2'] = np.where(test['Pclass'] == 2, 1, 0)



#create dummy codes for Pclass; pclass 2 vs pclass 3

train['pclass_3'] = np.where(train['Pclass'] == 3, 1, 0)

test['pclass_3'] = np.where(test['Pclass'] == 3, 1, 0)



#create dummy codes for sex

train['sex'] = np.where(train['Sex'] == "male", -0.5, 0.5)

test['sex'] = np.where(test['Sex'] == "male", -0.5, 0.5)



#create interaction for age * sex; younger females may be more likely to survive compared to younger males

train['sex_age'] = train['sex']*train['Age']

test['sex_age'] = test['sex']*test['Age']





#create dummy code for 'in travel group'; wouldn't actually be able to do this since this requires knowledge of train & test set

#create dummy code for 'youngest in travel group'; also not able to do this because we need data from both train & test set

#organizing train data

train = train.loc[:, ['Age', 'SibSp','Parch','embark_qc','embark_qs','pclass_2','pclass_3', 'sex', 'sex_age', 'Survived']]



#organizing test data

test = test.loc[:, ['Age', 'SibSp','Parch','embark_qc','embark_qs','pclass_2','pclass_3', 'sex', 'sex_age', 'Survived']]

train.head()
test.head()
#setup data for ML

X_train = train.iloc[:, :-1].values

y_train = train.iloc[:, -1].values

X_test = test.iloc[:, :-1].values

y_test = test.iloc[:, -1].values





#logistic regression

from sklearn.linear_model import LogisticRegression

lg_reg = LogisticRegression(random_state=0, max_iter = 1000)

lg_reg = lg_reg.fit(X_train, y_train)



#calculate predicted values

y_pred = lg_reg.predict(X_test)



#see confusion matrix and accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
#build model on test set

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski")

knn_classifier.fit(X_train, y_train)



#calculate predicted values

y_pred = knn_classifier.predict(X_test)



#see confusion matrix and accuracy

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
#Random forest

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)



#calculate predicted values

y_pred = classifier.predict(X_test)



#see confusion matrix and accuracy score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
#Naive bayes

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



#calculate predicted values

y_pred = classifier.predict(X_test)



#see confusion matrix and accuracy score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
#Support vector machine

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)



#calculate predicted values

y_pred = classifier.predict(X_test)



#see confusion matrix and accuracy score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
# Kernel SVM model 

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)



#calculate predicted values

y_pred = classifier.predict(X_test)



#see confusion matrix and accuracy score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)

#redo SVM model

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y_train)



#calculate predicted values

y_pred = classifier.predict(X_test)



#see confusion matrix and accuracy score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)



#make submission data

test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred}) #, 'Survived_og_test': test.Survived

output.head()

output.to_csv('gdpointon_titanic_submission.csv', index = False)

print("Your submission was successfully saved!")