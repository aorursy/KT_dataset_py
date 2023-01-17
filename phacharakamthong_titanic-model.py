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
import pandas as pd

# data = pd.read_csv('/kaggle/input/titanic/train.csv', sep=",")

train_data = pd.read_csv('/kaggle/input/dataset/train.csv', sep=",")

test_data = pd.read_csv('/kaggle/input/dataset/test.csv', sep=",")
train_data = train_data.drop(['Name','Ticket','Fare', 'Cabin', 'Embarked'], axis=1)

test_data = test_data.drop(['Name','Ticket','Fare', 'Cabin', 'Embarked'], axis=1)
print(train_data.head())

print(test_data.head())
train_data['Age'] = train_data.Age.fillna(value=train_data.Age.mean())

train_data['Sex'] = train_data.Sex.replace(to_replace="male", value=0)

train_data['Sex'] = train_data.Sex.replace(to_replace="female", value=1)

test_data['Age'] = test_data.Age.fillna(value=test_data.Age.mean())

test_data['Sex'] = test_data.Sex.replace(to_replace="male", value=0)

test_data['Sex'] = test_data.Sex.replace(to_replace="female", value=1)
test_data.head()

# train_data.to_csv('latest_Submission.csv', index = False)
import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

print(test_data)



#split dataset in features and target variable

feature_cols = ['Pclass', 'Sex', 'Age']

target = 'Survived'



X_train = train_data[feature_cols] # Features

y_train = train_data.Survived # Target variable

X_test = test_data[feature_cols]

y_test = test_data[target]
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix



# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)





# Evaluating Model ================================================================================================================================================================

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



# Make predictions

preds = clf.predict(X_test)

print("Predicted")

print(preds)

print('target : ',len(preds))



print(X_test)



# Calculate and print a confusion Matrix below ...

cm1 = confusion_matrix(y_test, preds)

print ("Confusion Matrix: \n", cm1)



total1 = sum(sum(cm1))

#print("Total test cases =", total1)



#print(cm1)

TP = cm1[0][0]

FN = cm1[0][1]

FP = cm1[1][0]

TN = cm1[1][1]

print('TP : ',TP)

print('FN : ',FN)

print('FP : ',FP)

print('TN : ',TN)

print('----------')



Myaccuracy_rate =  (TP + TN)/(TP + TN + FP + FN)

Myerror_rate =  (FP + FN)/(TP + TN + FP + FN)

MySensitivity = TP/(TP + FN)

MySpecificity = TN/(TN + FP)

MyPrecision = TP/(TP + FP)

MyRecall = TP/(TP + FN)



# Add codes to compute and print accuracy rate

print("accuracy rate : ", Myaccuracy_rate)



# Add codes to compute and print error rate

print("Error rate : ", Myerror_rate)



# Add codes to compute and print Sensitivity

print("Sensitivity : ", MySensitivity)



# Add codes to compute and print Specificity

print("Specificity : ", MySpecificity)



# Add codes to compute and print Precision

print("Precision : ", MyPrecision)



# Add codes to compute and print Recall

print("Recall : ", MyRecall)



DT = metrics.accuracy_score(y_test, y_pred)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':preds})

submission



submission.to_csv('submission.csv',index=False)
# Initialize our classifier

gnb = GaussianNB()



# Train our classifier

model = gnb.fit(X_train,y_train)



# print the classifier method used + test data set size

print ("GaussianNB and Test Data Size = ( 10% )")



# print("Actual class")

# print('test_labels : ',y_test)



# Evaluating Model ================================================================================================================================================================

# Model Accuracy, how often is the classifier correct?

print("============================================================")

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



# Make predictions

preds = gnb.predict(X_test)

print("Predicted")

print(preds)

print(len(preds))

print('X_test : ',X_test.head())

print('X_test.len : ',len(X_test))



# pd.DataFrame({

#     'PassengerId' : 

# })



# Calculate and print a confusion Matrix below ...

cm1 = confusion_matrix(y_test, preds)

print ("Confusion Matrix: \n", cm1)



total1 = sum(sum(cm1))

#print("Total test cases =", total1)



#print(cm1)

TP = cm1[0][0]

FN = cm1[0][1]

FP = cm1[1][0]

TN = cm1[1][1]

print('TP : ',TP)

print('FN : ',FN)

print('FP : ',FP)

print('TN : ',TN)

print('----------')



Myaccuracy_rate =  (TP + TN)/(TP + TN + FP + FN)

Myerror_rate =  (FP + FN)/(TP + TN + FP + FN)

MySensitivity = TP/(TP + FN)

MySpecificity = TN/(TN + FP)

MyPrecision = TP/(TP + FP)

MyRecall = TP/(TP + FN)



# Add codes to compute and print accuracy rate

print("accuracy rate : ", Myaccuracy_rate)



# Add codes to compute and print error rate

print("Error rate : ", Myerror_rate)



# Add codes to compute and print Sensitivity

print("Sensitivity : ", MySensitivity)



# Add codes to compute and print Specificity

print("Specificity : ", MySpecificity)



# Add codes to compute and print Precision

print("Precision : ", MyPrecision)



# Add codes to compute and print Recall

print("Recall : ", MyRecall)



NB = metrics.accuracy_score(y_test, y_pred)
print(DT)

print(NB)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print("Accuracy:",random_forest.score(X_train, y_train))

# Make predictions

preds = random_forest.predict(X_test)

print("Predicted")

print(preds)





# Calculate and print a confusion Matrix below ...

cm1 = confusion_matrix(y_test, preds)

print ("Confusion Matrix: \n", cm1)



total1 = sum(sum(cm1))

#print("Total test cases =", total1)



#print(cm1)

TP = cm1[0][0]

FN = cm1[0][1]

FP = cm1[1][0]

TN = cm1[1][1]

print('TP : ',TP)

print('FN : ',FN)

print('FP : ',FP)

print('TN : ',TN)

print('----------')



Myaccuracy_rate =  (TP + TN)/(TP + TN + FP + FN)

Myerror_rate =  (FP + FN)/(TP + TN + FP + FN)

MySensitivity = TP/(TP + FN)

MySpecificity = TN/(TN + FP)

MyPrecision = TP/(TP + FP)

MyRecall = TP/(TP + FN)



# Add codes to compute and print accuracy rate

print("accuracy rate : ", Myaccuracy_rate)



# Add codes to compute and print error rate

print("Error rate : ", Myerror_rate)



# Add codes to compute and print Sensitivity

print("Sensitivity : ", MySensitivity)



# Add codes to compute and print Specificity

print("Specificity : ", MySpecificity)



# Add codes to compute and print Precision

print("Precision : ", MyPrecision)



# Add codes to compute and print Recall

print("Recall : ", MyRecall)
# Make predictions

preds = random_forest.predict(X_test)

print("Predicted")

print(preds)





# Calculate and print a confusion Matrix below ...

cm1 = confusion_matrix(y_test, preds)

print ("Confusion Matrix: \n", cm1)



total1 = sum(sum(cm1))

#print("Total test cases =", total1)



#print(cm1)

TP = cm1[0][0]

FN = cm1[0][1]

FP = cm1[1][0]

TN = cm1[1][1]

print('TP : ',TP)

print('FN : ',FN)

print('FP : ',FP)

print('TN : ',TN)

print('----------')



Myaccuracy_rate =  (TP + TN)/(TP + TN + FP + FN)

Myerror_rate =  (FP + FN)/(TP + TN + FP + FN)

MySensitivity = TP/(TP + FN)

MySpecificity = TN/(TN + FP)

MyPrecision = TP/(TP + FP)

MyRecall = TP/(TP + FN)



# Add codes to compute and print accuracy rate

print("accuracy rate : ", Myaccuracy_rate)



# Add codes to compute and print error rate

print("Error rate : ", Myerror_rate)



# Add codes to compute and print Sensitivity

print("Sensitivity : ", MySensitivity)



# Add codes to compute and print Specificity

print("Specificity : ", MySpecificity)



# Add codes to compute and print Precision

print("Precision : ", MyPrecision)



# Add codes to compute and print Recall

print("Recall : ", MyRecall)
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

y_pred = logistic_regression.predict(X_test)

logistic_regression.score(X_train, y_train)

print("Accuracy:",logistic_regression.score(X_train, y_train))



# Make predictions

preds = logistic_regression.predict(X_test)

print("Predicted")

print(preds)





# Calculate and print a confusion Matrix below ...

cm1 = confusion_matrix(y_test, preds)

print ("Confusion Matrix: \n", cm1)



total1 = sum(sum(cm1))



TP = cm1[0][0]

FN = cm1[0][1]

FP = cm1[1][0]

TN = cm1[1][1]

print('TP : ',TP)

print('FN : ',FN)

print('FP : ',FP)

print('TN : ',TN)

print('----------')



Myaccuracy_rate =  (TP + TN)/(TP + TN + FP + FN)

Myerror_rate =  (FP + FN)/(TP + TN + FP + FN)

MySensitivity = TP/(TP + FN)

MySpecificity = TN/(TN + FP)

MyPrecision = TP/(TP + FP)

MyRecall = TP/(TP + FN)



# Add codes to compute and print accuracy rate

print("accuracy rate : ", Myaccuracy_rate)



# Add codes to compute and print error rate

print("Error rate : ", Myerror_rate)



# Add codes to compute and print Sensitivity

print("Sensitivity : ", MySensitivity)



# Add codes to compute and print Specificity

print("Specificity : ", MySpecificity)



# Add codes to compute and print Precision

print("Precision : ", MyPrecision)



# Add codes to compute and print Recall

print("Recall : ", MyRecall)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:",knn.score(X_train, y_train))



# Make predictions

preds = knn.predict(X_test)

print("Predicted")

print(preds)





# Calculate and print a confusion Matrix below ...

cm1 = confusion_matrix(y_test, preds)

print ("Confusion Matrix: \n", cm1)



total1 = sum(sum(cm1))

TP = cm1[0][0]

FN = cm1[0][1]

FP = cm1[1][0]

TN = cm1[1][1]

print('TP : ',TP)

print('FN : ',FN)

print('FP : ',FP)

print('TN : ',TN)

print('----------')



Myaccuracy_rate =  (TP + TN)/(TP + TN + FP + FN)

Myerror_rate =  (FP + FN)/(TP + TN + FP + FN)

MySensitivity = TP/(TP + FN)

MySpecificity = TN/(TN + FP)

MyPrecision = TP/(TP + FP)

MyRecall = TP/(TP + FN)



# Add codes to compute and print accuracy rate

print("accuracy rate : ", Myaccuracy_rate)



# Add codes to compute and print error rate

print("Error rate : ", Myerror_rate)



# Add codes to compute and print Sensitivity

print("Sensitivity : ", MySensitivity)



# Add codes to compute and print Specificity

print("Specificity : ", MySpecificity)



# Add codes to compute and print Precision

print("Precision : ", MyPrecision)



# Add codes to compute and print Recall

print("Recall : ", MyRecall)