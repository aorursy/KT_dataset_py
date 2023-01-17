import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Additional packages for data preprocessing, model building, data visualisation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

combined = pd.concat([train,test],axis = 0)
# Basic properties of data
train.info()
train.describe()
test.info()
test.describe()
# Investigating NA values in each column
#train.loc[train['Fare'] == 0]
#train.loc[train['Age'].isna()]
#train.loc[train['Cabin'].notna()]
train.loc[train['Embarked'].isna()]
# Removing NA values from data
train_pre = train.dropna(subset = ["Age"],inplace = False)
train_pre.info()
sns.swarmplot(x = "Pclass",y="Age",hue = "Survived",data = train_pre)
sns.swarmplot(x = "Survived",y="Age",hue = "Sex",data = train_pre)
sns.heatmap(data = train_pre.corr(),cmap = "Spectral",center = 0)
train_pre.dropna(subset = ['Cabin','Embarked'],inplace = True)
train_pre.info()
train_pre.describe()
# Filling NA values for Age
meanAge = round(combined["Age"].mean(skipna = True))
train['Age'].fillna(meanAge,inplace = True)
test['Age'].fillna(meanAge,inplace = True)
proxyCabin = "Unknown"
train['Cabin'].fillna(proxyCabin,inplace = True)
test['Cabin'].fillna(proxyCabin,inplace = True)
modeEmbark = combined["Embarked"].mode()[0]
train['Embarked'].fillna(modeEmbark,inplace = True)
test['Embarked'].fillna(modeEmbark,inplace = True)
meanFare = round(combined["Fare"].mean(skipna = True))
test['Fare'].fillna(meanFare,inplace = True)
train.info()
train.head()
# Dropping columns
train.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1,inplace = True)
test.drop(['Name','Ticket','Cabin'],axis = 1,inplace = True)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

train.drop(['SibSp','Parch'],axis = 1,inplace = True)
test.drop(['SibSp','Parch'],axis = 1,inplace = True)
train.info()
train.head()
# Encoding categorical columns 
trainPclassAdd = pd.get_dummies(train['Pclass'].reset_index(drop = True),prefix = 'Pclass',dtype = int)
trainEmbarkedAdd = pd.get_dummies(train['Embarked'].reset_index(drop = True).astype(str),prefix = 'Embarked',dtype = int)

testPclassAdd = pd.get_dummies(test['Pclass'].reset_index(drop = True),prefix = 'Pclass',dtype = int)
testEmbarkedAdd = pd.get_dummies(test['Embarked'].reset_index(drop = True).astype(str),prefix = 'Embarked',dtype = int)

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train = pd.concat([train.reset_index(drop = True),trainPclassAdd,trainEmbarkedAdd],axis = 1)
test = pd.concat([test.reset_index(drop = True),testPclassAdd,testEmbarkedAdd],axis = 1)

train.drop(["Embarked","Pclass"],axis = 1, inplace = True)
test.drop(["Embarked","Pclass"],axis = 1, inplace = True)
train.head()
test.head()
X_train = train.drop(['Survived'],axis = 1)
y_train = train['Survived']

X_test = test.drop(['PassengerId'],axis = 1)
X_train.shape, y_train.shape, X_test.shape
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
print("Training accuracy is " + str(logreg.score(X_train,y_train)* 100))
logreg.coef_[0]
# Support Vector Machine
svm = SVC()
svm.fit(X_train,y_train)
print("Training accuracy is " + str(svm.score(X_train,y_train)* 100))
# Decision tree classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
print("Training accuracy is " + str(dtc.score(X_train,y_train)* 100))