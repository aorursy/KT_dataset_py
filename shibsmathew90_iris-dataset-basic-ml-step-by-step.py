import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
# The Python Pandas packages helps us work with our datasets. We start by acquiring the datasets into Pandas DataFrames

iris_dataset = pd.read_csv('../input/Iris.csv')
# Preview the data

iris_dataset.head()
# Check whether there are any missing values in the dataset

iris_dataset.info()
# Id column in the dataset does not contribute hence dropping the 'ID' field.

iris_dataset.drop(['Id'], axis=1, inplace=True)



iris_dataset.head()
# Check the unique values present in the Species column

print(iris_dataset['Species'].unique())
# Plot the relationship between Sepal Length and Sepal width for all the species using matplotlib

fig = iris_dataset[iris_dataset.Species=='Iris-setosa'].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "orange", label='Setosa', marker='x')

iris_dataset[iris_dataset.Species=='Iris-versicolor'].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "green", label='Versicolor', ax = fig, marker='o')

iris_dataset[iris_dataset.Species=='Iris-virginica'].plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", color = "blue", label='Virginica', ax = fig, marker = 's')

fig.set_xlabel('Sepal Lenth in Cm')

fig.set_ylabel('Sepal Width in cm')

fig.set_title('Sepal Length vs Width')

fig = plt.gcf()

fig.set_size_inches(10,6)

plt.show()
# Plot the relationship between Petal Length and Petal width for all the species using seaborn package.



plt.figure(figsize=(10,6))

ax = sns.scatterplot(x=iris_dataset.PetalLengthCm, y=iris_dataset.PetalWidthCm, hue=iris_dataset.Species, style=iris_dataset.Species)

plt.title('Petal Length vs Width')

# Plotting boxplot 

f,axis = plt.subplots(2,2, figsize = [20,20])

plt.subplot(2,2,1)

sns.boxplot(data=iris_dataset, x = 'Species', y = 'SepalLengthCm')

plt.subplot(2,2,2)

sns.boxplot(data=iris_dataset, x = 'Species', y = 'SepalWidthCm')

plt.subplot(2,2,3)

sns.boxplot(data=iris_dataset, x = 'Species', y = 'PetalLengthCm')

plt.subplot(2,2,4)

sns.boxplot(data=iris_dataset, x = 'Species', y = 'PetalWidthCm')
# Generating violin plot to provide a visual distribution of data and its probability density. This provides a combination of Box plot and Density plot.

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

sns.violinplot('Species', 'SepalLengthCm', data = iris_dataset)

plt.subplot(2,2,2)

sns.violinplot('Species', 'SepalWidthCm', data = iris_dataset)

plt.subplot(2,2,3)

sns.violinplot('Species', 'PetalLengthCm', data = iris_dataset)

plt.subplot(2,2,4)

sns.violinplot('Species', 'PetalWidthCm', data = iris_dataset)
# Boxplot to get a better picture of how the data is distributed.

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

sns.distplot(iris_dataset['SepalLengthCm'], bins=10)

plt.subplot(2,2,2)

sns.distplot(iris_dataset['SepalWidthCm'], bins = 10)

plt.subplot(2,2,3)

sns.distplot(iris_dataset['PetalLengthCm'], bins = 10)

plt.subplot(2,2,4)

sns.distplot(iris_dataset['PetalWidthCm'], bins = 10)
# Generating Heatmap

plt.figure(figsize=(12,8))

sns.heatmap(iris_dataset.corr(), annot=True, linewidths=0.4)   # annot displays the value of each cell in the heatmat

plt.show()
# From the above heatmap its evident that the Petal Length and Petal Width are highly correlated. On the other hand, the Sepal Length and Sepal Width are not correlated.
# Loading the ML algorithms

from sklearn.linear_model import LogisticRegression  # For Logistic Regression

from sklearn.model_selection import train_test_split  # To split the data set into training and testing 

from sklearn.neighbors import KNeighborsClassifier  # for K Nearest neighbors

from sklearn import svm  # for SVM (Support Vector Machines) algorithm

from sklearn import metrics  # for checking model accuracy

from sklearn.tree import DecisionTreeClassifier  # for using Decision Tree Algorithm
# Splitting the dataset into Training and Testing

X = iris_dataset.drop(['Species'], axis=1)

y = iris_dataset['Species']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train.head()
y_train.head()
# Logistic Regression

model = LogisticRegression()

model.fit(X_train, y_train)

prediction_log_reg = model.predict(X_test)



print('The accuracy of Logistic Regression', metrics.accuracy_score(prediction_log_reg, y_test))
# K-nearest neigbors

model = KNeighborsClassifier(n_neighbors=10)

model.fit(X_train,  y_train)

prediction_KNear = model.predict(X_test)



print('The accuracy of K-nearest neighbors:', metrics.accuracy_score(prediction_KNear, y_test))
a_index = list(range(1,11))

a = pd.Series()

x = [1,2,3,4,5,6,7,8,9,10]

for i in a_index:

    model = KNeighborsClassifier(n_neighbors=i)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    a = a.append(pd.Series(metrics.accuracy_score(prediction, y_test)))

plt.plot(a_index, a)
# The above graph shows accuracy levels of KNN models for different values of n
# We had used all the features of the Iris. Now we will use Petals and Sepals seperately
petals = iris_dataset[['PetalLengthCm', 'PetalWidthCm', 'Species']]

sepals = iris_dataset[['SepalLengthCm', 'SepalWidthCm', 'Species']]
petals.head()
sepals.head()
# Splitting the Petals and Sepals dataset into Training and Testing



petals_x = petals.drop(['Species'], axis = 1)

petals_y = petals['Species']



sepals_x = sepals.drop(['Species'], axis = 1)

sepals_y = sepals['Species']



x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(petals_x, petals_y, test_size = 0.3)

x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(sepals_x, sepals_y, test_size = 0.3)
print(x_train_p.shape, x_test_p.shape, y_train_p.shape, y_test_p.shape)

print(x_train_s.shape, x_test_s.shape, y_train_s.shape, y_test_s.shape)
#Logistic Regression

model_log = LogisticRegression()

# 1. Petals

model_log.fit(x_train_p, y_train_p)

prediction = model_log.predict(x_test_p)



print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))



# 2. Sepals

model_log.fit(x_train_s, y_train_s)

prediction = model_log.predict(x_test_s)



print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))
# Decision Tree 

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(criterion='gini', max_depth=6)  # by defalt it takes Gini index as critera



# 1. Petals

model_dt.fit(x_train_p, y_train_p)

prediction = model_dt.predict(x_test_p)



print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))



# 2. Sepals

model_dt.fit(x_train_s, y_train_s)

prediction = model_dt.predict(x_test_s)



print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))
# Random Forest

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()



# 1. Petals

model_rf.fit(x_train_p, y_train_p)

prediction = model_rf.predict(x_test_p)



print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))



# 2. Sepals

model_rf.fit(x_train_s, y_train_s)

prediction = model_rf.predict(x_test_s)



print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))
# K-Nearest Neighbors

model_knn = KNeighborsClassifier(n_neighbors=9)



# 1. Petals

model_knn.fit(x_train_p, y_train_p)

prediction = model_knn.predict(x_test_p)



print('The accuracy of the model for Petals is:', metrics.accuracy_score(prediction, y_test_p))



# 2. Sepals

model_knn.fit(x_train_s, y_train_s)

prediction = model_knn.predict(x_test_s)



print('The accuracy of the model for Sepals is:', metrics.accuracy_score(prediction, y_test_s))