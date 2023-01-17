import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline
data = pd.read_csv("../input/train.csv")
data.head()
print("# of passengers in dataset:"+str(len(data)))
sns.countplot(x = "Survived", data = data);
sns.countplot(x = "Survived", hue = "Sex", data = data);
sns.countplot(x = "Survived", hue ="Pclass", data = data);
data["Age"].plot.hist();
data["Fare"].plot.hist(bins = 20, figsize=(10,5));
data.info()
sns.countplot(x="SibSp", data = data);
data.isnull()
data.isnull().sum()
sns.heatmap(data.isnull(), yticklabels = False, cmap="viridis")
sns.boxplot(x = "Pclass", y = "Age", data = data);
data.drop("Cabin", axis = 1, inplace = True)
data.dropna(inplace = True)
sns.heatmap(data.isnull(), yticklabels = False, cbar = False);
data.isnull().sum()
pd.get_dummies(data['Sex'])
sex = pd.get_dummies(data['Sex'], drop_first = True)
sex.head()
embark = pd.get_dummies(data['Embarked'])
embark.head()
embark = pd.get_dummies(data['Embarked'], drop_first=True)
embark.head(5)
pcl = pd.get_dummies(data['Pclass'])
pcl.head()
pcl = pd.get_dummies(data['Pclass'], drop_first = True)
pcl.head()
data = pd.concat([data, sex, embark, pcl], axis = 1)
data.head()
data.drop(['Sex', 'Embarked', 'Pclass', 'PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
data.head()
X = data.drop("Survived", axis = 1)
Y = data["Survived"]  # this is our target variable
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(Y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, predictions)