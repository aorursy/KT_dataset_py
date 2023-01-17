import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier 

from sklearn import svm 

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier  

from sklearn import metrics 
data=pd.read_csv('../input/iris/Iris.csv')
data.head()
data.tail()
data.shape
data.info()
data.isna().sum()
data.describe().T
plt.figure(figsize=(5,5))

cor=data.corr()

sns.heatmap(cor,annot=True)
data["Species"].value_counts()
sns.pairplot(data,hue='Species')
fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

plt.legend()
fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title("Petal Length VS Width")

plt.legend()
data.hist()

fig=plt.gcf()

fig.set_size_inches(14,8)

plt.show()
x = data.drop(['Id', 'Species'], axis=1)

y = data['Species']

print(x.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
logr=LogisticRegression(solver="lbfgs")

logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)

logacc = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the Logistic Regression is: ', logacc)
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)

dtacc = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the Decision Tree is: ', dtacc)
knn = KNeighborsClassifier(n_neighbors=3) #

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

knnacc = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the KNN is', knnacc)
sv = svm.SVC()

sv.fit(X_train,y_train) 

y_pred = sv.predict(X_test) 

svmacc = metrics.accuracy_score(y_pred,y_test)

print('The accuracy of the SVM is:', svmacc)
models=pd.DataFrame({'Models':['Logistic Regression','Decision Tree','K-NN','SVM'],'Accuracies':[logacc,dtacc,knnacc,svmacc]})
models.sort_index(1)