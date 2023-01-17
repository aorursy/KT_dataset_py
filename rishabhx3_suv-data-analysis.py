import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/suv-data/suv_data.csv")

data.head(10)
print("Number of customers: ", len(data))
data.info()
sns.countplot(x='Purchased', data = data)
sns.countplot(x='Purchased', hue = 'Gender', data = data)
data['Age'].plot.hist()
data['EstimatedSalary'].plot.hist()
plt.figure(figsize = (5,5))

sns.distplot(data[data['Purchased']==1]['Age'])
plt.figure(figsize = (20,10))

sns.barplot(x=data['Age'],y=data['Purchased'])
plt.figure(figsize = (5,5))

sns.distplot(data[data['Purchased']==1]['EstimatedSalary'])
plt.figure(figsize = (20,7))

sns.lineplot(x=data['EstimatedSalary'],y=data['Purchased'])
Gender = pd.get_dummies(data['Gender'], drop_first = True)

Gender.head(5)
data = pd.concat([data, Gender], axis = 1)
data.drop(['User ID', 'Gender'], axis = 1, inplace = True)

data.head()
X = data.drop('Purchased', axis = 1)

y = data['Purchased']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
model = LogisticRegression(solver = 'liblinear')
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print("Confusion Matrix: \n",confusion_matrix(y_test, predictions))
print("Accuracy: ",accuracy_score(y_test, predictions))