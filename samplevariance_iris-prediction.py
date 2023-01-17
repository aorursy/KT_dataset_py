#import libraries

import numpy as np

import pandas as pd

from pandas import get_dummies

from matplotlib import pyplot as plt

from scipy import stats

import seaborn as sns

#set default sns style

sns.set()

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

#ensure csv copied into folder

import os

print(os.listdir("../input"))
#load data

iris = pd.read_csv('../input/Iris.csv',header=0,index_col=0)

#what is the dataframe's shape?

print(iris.shape)

#datatypes?

print(iris.dtypes)

#list of features

print(iris.columns.values)

#describe dataframe

print(iris.describe())

#overview of top and bottom (5) rows

print(iris.head())

print(iris.tail())

#any column null?

print(iris.columns[iris.isnull().any()])
print(iris.groupby('Species').size())
n_data = len(iris['PetalLengthCm'])

#number of bins, we will use a pretty simple rule called "square root rule"

#Choose the number of bins to be the square root of the number of samples

n_bins = np.sqrt(n_data)

n_bins = int(n_bins)

iris.plot(kind='hist',subplots=True,layout=(2,2),bins=n_bins)

plt.tight_layout()

plt.suptitle("Figure 1: Feature Frequency")

plt.show()
_ = sns.swarmplot(x='PetalWidthCm',y='PetalLengthCm',hue='Species', data=iris)

plt.xlabel("Petal Width (CM)")

plt.ylabel("Petal Length (CM)")

plt.suptitle("Figure 2: Petal Width (CM) versus Petal Length (CM)")

plt.show()
_ = sns.swarmplot(x='SepalWidthCm',y='SepalLengthCm',hue='Species', data=iris)

plt.xlabel("Sepal Width (CM)")

plt.ylabel("Sepal Length (CM)")

plt.title("Figure 3: Sepal Width (CM) versus Sepal Length (CM)")

plt.show()
_ = sns.swarmplot(x='Species',y='PetalLengthCm',data=iris)

plt.xlabel("Species")

plt.ylabel("Petal Length (CM)")

plt.title("Figure 4: Petal Length (CM) by Species")

plt.show()
_ = sns.swarmplot(x='Species',y='PetalWidthCm',data=iris)

plt.xlabel("Species")

plt.ylabel("Petal Width (CM)")

plt.title("Figure 5: Petal Width (CM) by Species")

plt.show()
_ = sns.swarmplot(x='Species',y='SepalLengthCm',data=iris)

plt.xlabel("Species")

plt.ylabel("Sepal Length (CM)")

plt.title("Figure 6: Sepal Length (CM) by Species")

plt.show()
_ = sns.swarmplot(x='Species',y='SepalWidthCm',data=iris)

plt.xlabel("Species")

plt.ylabel("Sepal Width (CM)")

plt.title("Figure 7: Sepal Width (CM) by Species")

plt.show()
_ = sns.pairplot(iris, hue="Species")

plt.suptitle("Figure 8: Scatterplot Matrix")

plt.show()
iris.plot(kind='box',subplots=True,layout=(2,2))

plt.suptitle("Figure 9: Feature box plot")

plt.show()
sns.boxplot(x=iris['SepalWidthCm'])

plt.title("Figure 10: Sepal Width (CM) Box Plot")

plt.show()
print(iris.describe())
x = iris.iloc[:, :-1].values

y = iris.iloc[:, -1].values

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.5, random_state = 7,stratify=y)

print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
#let's create a model that looks at 6 cloest neighbors

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data

clf = knn.fit(x_train,y_train)

#isolate predicted target (y)

y_pred = clf.predict(x_test)

# Print the accuracy of test

print(knn.score(x_test, y_test))

#let's look at the confusion matrix

cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['setosa','versicolor','virginica'], 

                     columns = ['setosa','versicolor','virginica'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('kNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

#full report

print(classification_report(y_test, y_pred))
#let's find the best # of neighbors to train on

#I started with 6, let's double that to 12 and do a range of neighbors from 1-12.

neighbors = np.arange(1, 12)

#let's store the result in some arrays

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



#we want to do the same process as above, but for 1 to 12 in n_neighbors

#knn = KNeighborsClassifier(n_neighbors=[1-12])

#knn.fit(x_train,y_train)



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data

    knn.fit(x_train,y_train)

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(x_train, y_train)

    #Compute accuracy on the testing set

    test_accuracy[i] =  knn.score(x_test, y_test)

# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()
#change n_neighbors as per above

knn = KNeighborsClassifier(n_neighbors=9)

# Fit the classifier to the training data

clf = knn.fit(x_train,y_train)

#isolate predicted target (y)

y_pred = clf.predict(x_test)



# Print the accuracy of test

print(knn.score(x_test, y_test))

#let's look at the confusion matrix

#let's look at the confusion matrix

cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['setosa','versicolor','virginica'], 

                     columns = ['setosa','versicolor','virginica'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('kNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

#full report

print(classification_report(y_test, y_pred))
# Support Vector Machine

Model = SVC()

Model.fit(x_train, y_train)



y_pred = Model.predict(x_test)

# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

#confusion matrix

cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['setosa','versicolor','virginica'], 

                     columns = ['setosa','versicolor','virginica'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('SVC \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

#full report

print(classification_report(y_test, y_pred))
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 7,stratify=y)

print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
knn = KNeighborsClassifier(n_neighbors=8)

# Fit the classifier to the training data

clf = knn.fit(x_train,y_train)

#isolate predicted target (y)

y_pred = clf.predict(x_test)



# Print the accuracy of test

print(knn.score(x_test, y_test))

#let's look at the confusion matrix

#let's look at the confusion matrix

cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['setosa','versicolor','virginica'], 

                     columns = ['setosa','versicolor','virginica'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('kNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

#full report

print(classification_report(y_test, y_pred))
Model = SVC()

Model.fit(x_train, y_train)



y_pred = Model.predict(x_test)

# Accuracy score

print('accuracy is',accuracy_score(y_pred,y_test))

#confusion matrix

cm = confusion_matrix(y_test, y_pred) 

# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = ['setosa','versicolor','virginica'], 

                     columns = ['setosa','versicolor','virginica'])



plt.figure(figsize=(5.5,4))

sns.heatmap(cm_df, annot=True)

plt.title('SVC \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()

#full report

print(classification_report(y_test, y_pred))