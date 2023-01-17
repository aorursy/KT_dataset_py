# import packages that will be used:

# for data manipulation:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
iris = pd.read_csv('../input/Iris.csv') # Read in dataset

iris.head(2)
# show a summary of iris dataset, there are 150 entries in this dataset and the lengths of all fields are 150, which means there is no null values in this dataset.

iris.info() 
iris.head()
# Drop the first column 'Id', which is not useful

iris.drop('Id', axis=1).head(3)
iris.head(3)
# Try a different 'drop', in which set inplace to be True

iris.drop('Id', axis=1, inplace=True)

iris.head(3)
iris_setosa = iris[iris.Species == 'Iris-setosa']

iris_versicolor = iris[iris.Species == 'Iris-versicolor']

iris_virginica = iris[iris.Species == 'Iris-virginica']

# plot all data in a single chart, the style is set to be scatter, 

# ax=fig to make sure that the plot apear on 'fig'

# the size of figure is set in function dataframe.plot()

fig = iris_setosa.plot(x='SepalLengthCm', y='SepalWidthCm', kind='scatter', color='orange', 

                       label='Setosa', figsize=(10, 5))

iris_versicolor.plot(x='SepalLengthCm', y='SepalWidthCm', kind='scatter', color='green', label='Versicolor', ax=fig)

iris_virginica.plot(x='SepalLengthCm', y='SepalWidthCm', kind='scatter', color='blue', label='Virginica', ax=fig)

fig.set_xlabel('Sepal Length in centermeter')

fig.set_ylabel('Sepal Width in centermeter')

fig.set_title("Sepal Length VS Width")

plt.show()
fig = iris_setosa.plot(x='PetalLengthCm', y='PetalWidthCm', kind='scatter', color='orange', 

                       label='Setosa', figsize=(10, 5))

iris_versicolor.plot(x='PetalLengthCm', y='PetalWidthCm', kind='scatter', color='green', label='Versicolor', ax=fig)

iris_virginica.plot(x='PetalLengthCm', y='PetalWidthCm', kind='scatter', color='blue', label='Virginica', ax=fig)

fig.set_xlabel('Petal Sepal Length in centermeter')

fig.set_ylabel('Petal Width in centermeter')

fig.set_title("Petal Sepal Length VS Width")

plt.show()
iris.hist(edgecolor='black', linewidth=1, color='orange')

plt.show()
plt.figure(figsize=(10, 7))

sns.heatmap(iris.corr(), annot=True, cmap='Purples')

plt.show()
# import packages that will be used:

# for data manipulation for machine learning, mainly we need sklearn

from sklearn.cross_validation import train_test_split

from sklearn import svm

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

iris.shape
# splitting dataset into two parts: train and test, 30% are test data

(train, test) = train_test_split(iris, test_size=0.3)

print('----Head of training data----')

print(train.head(3))

print('----dimensions of training data----')

print(train.shape)

print('----Head of testing data----')

print(test.head(3))

print('----dimensions of testing data----')

print(test.shape)
print('---- training data info ----')

train_X = train.drop('Species', axis=1)

print(train_X.head(3))

train_Y = train[['Species']]

print(train_Y.head(3))

print('---- testing data info ----')

test_X = test.drop('Species', axis=1)

print(test_X.head(3))

test_Y = test[['Species']]

print(test_Y.head(3))
# SVM

# using SVM to train a model and see its accuracy

model = svm.SVC()

model.fit(train_X, train_Y.values.ravel())

prediction = model.predict(test_X)

print('The accuracy of the SVM is: ', metrics.accuracy_score(prediction, test_Y.values.ravel()))
# Logistic Regression

# using Logistic Regression to train a model and see its accuracy

model = LogisticRegression()

model.fit(train_X, train_Y.values.ravel())

prediction = model.predict(test_X)

print('The accuracy of the Logistic Resgression is: ', 

      metrics.accuracy_score(prediction, test_Y.values.ravel()))
labels=['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']

cm = metrics.confusion_matrix(y_true=np.array(test_Y.values.ravel()), y_pred=prediction,

                              labels = labels)

# print(cm)

df_cm = pd.DataFrame(cm, index = [i for i in labels],

                  columns = [i for i in labels])

plt.figure(figsize = (8,5))

sns.heatmap(df_cm, annot=True)

plt.show()
# K-Nearest Neighbours 

model = KNeighborsClassifier(n_neighbors=3)

model.fit(train_X, train_Y.values.ravel())

prediction = model.predict(test_X)

print('The accuracy of the K-nearest neighbours is: ', 

      metrics.accuracy_score(prediction, test_Y.values.ravel()))
# plot the confusion matrix

labels=['Iris-virginica', 'Iris-setosa', 'Iris-versicolor']

cm = metrics.confusion_matrix(y_true=np.array(test_Y.values.ravel()), y_pred=prediction,

                              labels = labels)



df_cm = pd.DataFrame(cm, index = [i for i in labels],

                  columns = [i for i in labels])

plt.figure(figsize = (8,5))

sns.heatmap(df_cm, annot=True)

plt.show()