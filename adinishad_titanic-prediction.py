# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load the dataset
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
# First 5 row
train.head()
test.head()
submission.head()
# Shape of train and test dataset
print(train.shape)
print(test.shape)
# important information of dataset
print(train.info())
# total null value
print(train.isna().sum())
# function for drop column from train and test
def drop_col(trainORtest, *name):
    drop_list = [*name]
    trainORtest.drop(drop_list, axis=1, inplace=True)
    
# column name
drop_col(train, "Name", "Ticket", "Cabin", "Fare")
drop_col(test, "Name", "Ticket", "Cabin", "Fare")
# updated first 5 row
train.head()
# Unique value
train["Sex"].unique()
test.head()
# Label Encoder for encoding column
from sklearn.preprocessing import LabelEncoder
# encode
encoder = LabelEncoder()

train["Sex"] = encoder.fit_transform(train["Sex"])
test["Sex"] = encoder.fit_transform(test["Sex"])
train.head()
train["Sex"].nunique()
# Port of Embarkation
train["Embarked"].unique()
# Port of Embarkation fill value
train["Embarked"] = train["Embarked"].fillna("Q")
# encode Embarked column
train["Embarked"] = encoder.fit_transform(train["Embarked"])
test["Embarked"] = encoder.fit_transform(test["Embarked"])
# mean value of age column and fill
mean = train["Age"].mean()
train["Age"] = train["Age"].fillna(mean)

mean = test["Age"].mean()
test["Age"] = test["Age"].fillna(mean)
train.head()
# correlation
correlation = train.corr()


# heatmap
plt.subplots(figsize=(10,6))
sns.heatmap(correlation, annot=True)

# Correlation with Sex column 
correlation["Sex"].sort_values(ascending=False)
plt.subplots(figsize=(8,5))
sns.countplot(x=train["Pclass"], hue=train["Survived"], data=train)
plt.subplots(figsize=(8,5))
sns.countplot(x=train["Sex"], hue=train["Survived"], data=train)
plt.subplots(figsize=(8,5))
sns.countplot(x=train["Pclass"])
plt.subplots(figsize=(8,5))
sns.countplot(x=train["Pclass"], hue=train["Embarked"], data=train)
plt.subplots(figsize=(8,5))
sns.countplot(x=train["Parch"], data=train)

plt.subplots(figsize=(8,5))
sns.countplot(x=train["SibSp"], hue=train["Pclass"], data=train)
X = train.drop("Survived", axis=1)
y = train["Survived"].copy()
# Split the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
# different types of model and accuracy
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(n_estimators=700),
    LogisticRegression(),
    SGDClassifier(),
    GaussianNB(),
#     SVC(kernel = 'linear')
]

models = zip(names, classifiers)
for name, model in models:
    model_name = model
    model_name.fit(X_train, y_train)
    predict = model_name.predict(X_test)
    accuracy1 = accuracy_score(predict, y_test)
    accuracy2 = confusion_matrix(predict, y_test)
    accuracy3 = classification_report(predict, y_test)
    scores = cross_val_score(model, X_train, y_train, cv = 10, scoring = "accuracy")

    print(f"{name} model Accuracy Score {accuracy1}")
    
    print("******************************")
    
    print(f"confusion matrix {accuracy2}")
    
    print("******************************")
    
    print(f"classification report {accuracy3}")
    
    print("******************************")
    
    print(f"{name} and prediction {predict}")
    
    print("******************************")
    
    print(f"cross val score {scores}")
    
    print("*******************************")

# final prediction
a = list(submission["Survived"])
models = zip(names, classifiers)
for name, model in models:
    final_predict = model.predict(test)
    print(f"{name}: {final_predict}")
    print(f"submission : {a}")

# predict from random forest
models = zip(names, classifiers)
model_select = classifiers[2]
final_predict = model_select.predict(test)

# create submission dataset
'''
index = test.PassengerId
newFrame = pd.DataFrame({"PassengerId":index, "Survived":final_predict})
newFrame.to_csv("new_titanic_submissio_rf.csv", index=False)
'''
