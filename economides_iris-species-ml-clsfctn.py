import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile

# cwd = os.getcwd() #getting current working directory

# #unzipping the file
# with zipfile.ZipFile('iris.zip', 'r') as zip_ref:
#     zip_ref.extractall(cwd)

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


iris = pd.read_csv("../input/iris/Iris.csv")
#iris = pd.read_csv("iris.csv")
iris.drop('Id', axis=1, inplace=True) #dropping unessecery column

print(iris.shape,"\n")
print(set(iris["Species"]),"\n")
print(iris.head())

fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()
iris.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
plt.figure(figsize=(7,4)) 
sns.heatmap(iris.corr(), annot=True, cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()
train, test = train_test_split(iris, test_size = 0.3)

train_X = train[iris.columns[:-1]]  # taking the training data features
train_y = train.Species             # output of our training data
test_X = test[iris.columns[:-1]]    # taking test data features
test_y = test.Species
model = svm.SVC()
model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("The accuracy of the SVM is: ", 
      metrics.accuracy_score(prediction, test_y))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, prediction)
print(cm)
model = LogisticRegression()
model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("The accuracy of the LogisticRegression is: ", 
      metrics.accuracy_score(prediction, test_y))

model = DecisionTreeClassifier()
model.fit(train_X, train_y)

prediction = model.predict(test_X)

print("The accuracy of the Decision Tree is: ", 
      metrics.accuracy_score(prediction, test_y))
a_index = list(range(1,11))
a = pd.Series()
x = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]

for i in list(range(1, 11)):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    a = a.append(pd.Series(metrics.accuracy_score(prediction, test_y)))
    
plt.plot(a_index, a);
plt.xticks(x);
