import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score

data = pd.read_csv("../input/mushrooms.csv")
data.head()
# Structure of the data (shows 8124 rows and 23 columns)

data.shape
data.describe().T
# Check for missing values (No Missing values present)



data.isnull().values.any()
#Check for uniques values in for class



data['class'].unique()

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 

data.head()
# Checking for unique values after labeling to integers

data['stalk-color-above-ring'].unique()
#Check values for target variable

print(data.groupby('class').size())
#Exploratory Data analysis for target variable



ax = sns.boxplot(x='class', y='stalk-color-above-ring', data=data)
#Checking for correlation (color closer to 1 means they are highly correlated)

corr = data.corr()

corr = data.corr()

ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
X = data.drop(["class"], axis=1)

y = data["class"]

X = pd.get_dummies(X)



le = LabelEncoder()

y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)



clf = LogisticRegression(solver="lbfgs").fit(X_train,y_train)

predicted = clf.predict(X_test)

predicted_proba = clf.predict(X_test)



print("Accuracy is: "+ str(clf.score(X_test,y_test)))

print("Recall score is: " + str(round(recall_score(y_test, predicted),3)))

print("Precision score is: " + str(round(precision_score(y_test, predicted),3)))

print("F1 score is: " + str(round(f1_score(y_test, predicted),3)))

print("\nConfusion matrix:")

print(confusion_matrix(y_test, predicted))