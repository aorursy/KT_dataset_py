# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl')

import matplotlib.pyplot as plt

%matplotlib inline
#Load Data

iris = pd.read_csv('../input/Iris.csv')
iris.shape
# View Top 5 records

iris.head()
# View bottom 5 records

iris.tail()
iris.info()
iris.describe()
iris['Species'].value_counts()
iris1 = iris.drop('Id', axis=1)

g = sns.pairplot(iris1, hue='Species', markers='+')

plt.show()
fig = iris1[iris1.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris1[iris1.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris1[iris1.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
fig = iris1[iris1.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris1[iris1.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

iris1[iris1.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()
iris1.hist(edgecolor='black')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
g = sns.violinplot(y='Species', x='SepalLengthCm', data=iris1, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='SepalWidthCm', data=iris1, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalLengthCm', data=iris1, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalWidthCm', data=iris1, inner='quartile')

plt.show()
plt.figure(figsize=(10,8)) 

sns.heatmap(iris1.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())

plt.show()
X = iris.drop(['Id', 'Species'], axis=1)

y = iris['Species']

# print(X.head())

print(X.shape)

# print(y.head())

print(y.shape)
from sklearn.model_selection import train_test_split  #to split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
# Importing alll the necessary packages to use the various classification algorithms



from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import metrics #for checking the model accuracy
logr = LogisticRegression()

logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

acc_log = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the Logistic Regression is', acc_log)
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

acc_dt = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the Decision Tree is', acc_dt)
sv = svm.SVC() #select the algorithm

sv.fit(X_train,y_train) # we train the algorithm with the training data and the training output

y_pred = sv.predict(X_test) #now we pass the testing data to the trained algorithm

acc_svm = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the SVM is:', acc_svm)
knc = KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

knc.fit(X_train,y_train)

y_pred = knc.predict(X_test)

acc_knn = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the KNN is', acc_knn)
a_index = list(range(1,11))

a = pd.Series()

x = [1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    kcs = KNeighborsClassifier(n_neighbors=i) 

    kcs.fit(X_train,y_train)

    y_pred = kcs.predict(X_test)

    a=a.append(pd.Series(metrics.accuracy_score(y_pred,y_test)))

plt.plot(a_index, a)

plt.xticks(x)
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machines',

              'K-Nearest Neighbours'],

    'Score': [acc_log, acc_dt, acc_svm, acc_knn]})

models.sort_values(by='Score', ascending=False)