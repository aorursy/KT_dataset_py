import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import math





#ইনলাইন প্লটিংয়ের জন্য জুপিটার নোটবুকের ম্যাজিক ফাংশন (আলাদা উইন্ডোতে প্লট শো করতে চাচ্ছি না আমরা)

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)
train.head(3)
test.head(3)
Y_train = train["Survived"]



g = sns.countplot(Y_train)



Y_train.value_counts()
train.isnull().sum()
train.isnull().values.any()
train["Sex"] = train["Sex"].map({"male": 0, "female":1})
import math

train["Age"] = train["Age"].replace(np.NaN, math.floor(train['Age'].mean()))
test.isnull().sum()
test["Sex"] = test["Sex"].map({"male": 0, "female":1})
test["Age"] = test["Age"].replace(np.NaN, math.floor(test['Age'].mean()))
g = sns.heatmap(train[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
num_true = len(train.loc[train['Survived'] == 0])

num_false = len(train.loc[train['Survived'] == 1])

print ("Number of Survived People: {0} ({1:2.2f}%)".format(num_true, (num_true / (num_true + num_false)) * 100))

print ("Number of Dead People: {0} ({1:2.2f}%)".format(num_false, (num_false / (num_true + num_false)) * 100))
train.columns
from sklearn.model_selection import train_test_split



feature_column_names = ['Pclass', 'Sex', 'Age', 'Parch', 'SibSp']



predicted_class_name = ['Survived']



# Getting feature variable values



X = train[feature_column_names].values

y = train[predicted_class_name].values



# Saving 30% for testing

split_test_size = 0.30



# Splitting using scikit-learn train_test_split function



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_test_size, random_state = 42)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics





# Create a RandomForestClassifier object

rf_model = RandomForestClassifier(random_state=42)



rf_model.fit(X_train, y_train.ravel())
rf_predict_train = rf_model.predict(X_train)



#get accuracy

rf_accuracy = metrics.accuracy_score(y_train, rf_predict_train)



#print accuracy

print ("Accuracy: {0:.4f}".format(rf_accuracy))
rf_predict_test = rf_model.predict(X_test)



#get accuracy

rf_accuracy_testdata = metrics.accuracy_score(y_test, rf_predict_test)



#print accuracy

print ("Accuracy: {0:.4f}".format(rf_accuracy_testdata))
print ("Confusion Matrix for Random Forest")



# labels for set 1=True to upper left and 0 = False to lower right

print ("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0])))
sub_test = test[feature_column_names].values
pred_test = rf_model.predict(sub_test)
submission = test.copy()

submission['Survived'] = pred_test

submission.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)



submission[['PassengerId', 'Survived']].head(15)