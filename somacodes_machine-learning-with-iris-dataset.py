

## We will be importing the major libraries that we will need for our EDA

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns





from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



data_df = pd.read_csv('../input/Iris.csv')
## We will first have a look at the top five records. Note that since the first 50 records belong to Iris-sentosa

## so will will not be seeing any other records apart from those belonging to Iris-sentosa

data_df.head()
## Next we are interested in finding out a little more about the dataset like the data columns/features,their data types 

## and also total number of values for each feature.

data_df.info()
## Next we will be exploring the count,mean,standard deviation,min/max and the percentiles off the numeric values.

## On observation it is found that the mean of the features lie within the 50th percentile.



data_df.describe()
## Next we will have a look at our label column

## using the function value_counts(), we will see the total unique count of records

## So we see that there are 50 labels for Iris-setosa, 50 fir Iris-versicolor and 50 for Iris-virginica



data_df['Species'].value_counts()
data = data_df.drop('Id', axis = 1)

## axis = 1 means, drop column

## axis = 0 means, drop labels
data.corr(method = 'pearson')
correlation = data.corr(method = 'pearson')

heat_map = sns.heatmap(correlation,annot = True, cmap = 'coolwarm', linewidth = .5)

plt.show()
## Also we will be demonstrating the pair-wise feature correlation using pairplot from the seaborn library

## We are giving the parameter hue = 'Species' so that for each species, the colour marker will be different.



sns.pairplot(data, hue = 'Species')
g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')

plt.show()
## In X, we are storing all the features while in y we are storing the lables

X = data.drop(['Species'], axis=1)

y = data['Species']



## 50% training date, 50% test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# experimenting with different n values

k_range = list(range(1,51))

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))



plt.plot(k_range, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.grid(True)

plt.show()

## K Nearest Neighbour

knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
## Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
## Decision Tree

tree = DecisionTreeClassifier(max_depth = 3)

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

metrics.accuracy_score(y_test,y_pred)
## Now lets split the train and test data into 80% training data and 20% testing data 

## And run our tests again



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=5)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(X_test.shape)
## K Nearest Neighbour

knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
## Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

metrics.accuracy_score(y_test, y_pred)
## Decision Tree

tree = DecisionTreeClassifier(max_depth = 3)

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

metrics.accuracy_score(y_test,y_pred)