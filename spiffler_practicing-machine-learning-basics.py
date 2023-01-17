# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')

iris['Species'] = iris['Species'].astype('category')

iris = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']]



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(iris['Species'])

iris_target = le.transform(iris['Species'])

iris_data = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].as_matrix() #i dont think ID is of any use

print(iris_target)
iris.info()

iris.describe()

iris["Species"].value_counts()
# ok just went and did a super quick check on what is the meaning of sepal! (I knew what petal is)

# Lets try and plot this to get some basic insights into the data



import warnings 

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)



sns.pairplot(data=iris, hue='Species')
# average petallength for every species

sns.boxplot(x='Species', y='PetalLengthCm', data=iris)
# average sepallength for every species

sns.boxplot(x='Species', y='SepalLengthCm', data=iris)
# average petalwidth for every species

sns.boxplot(x='Species', y='PetalWidthCm', data=iris)
# average sepalwidth for every species

sns.boxplot(x='Species', y='SepalWidthCm', data=iris)
# setting features to be used to be Petal Lengtha nd Width

features = iris.iloc[:,2:4].values

labels = iris.iloc[:,4].values
# Encoding the categorical Dependent Variable



from sklearn.preprocessing import LabelEncoder

labelencoder_labels = LabelEncoder()

labels = labelencoder_labels.fit_transform(labels)
# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 43)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

features_train = sc.fit_transform(features_train)

features_test = sc.transform(features_test)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('Naive Bayes (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Fitting KNN to the Training set

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('KNN (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Fitting SVM Linear to the Training set

from sklearn.svm import SVC

clf = SVC(kernel='linear', random_state=43)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('SVM linear (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Fitting SVM RBF to the Training set

from sklearn.svm import SVC

clf = SVC(kernel='rbf', random_state=43)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('SVM RBF (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Fitting SVM RBF to the Training set

from sklearn.svm import SVC

clf = SVC(kernel='rbf', random_state=43, gamma = 0.5)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('SVM RBF (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Visualising the TRAINING set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_train, labels_train

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('SVM RBF (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# It is ok to not use feature scaling when using decision trees - i will just keep it in for now



# Fitting Decision Tree to the Training set

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
iris.describe()
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('Decision Tree (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 7, criterion='entropy', random_state=43)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cmat = confusion_matrix(labels_test, labels_pred)

print(cmat)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Visualising the Test set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_test, labels_test

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('RandomForest (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# Visualising the TRAINING set results (based the Udemy course by Kirill Ermenko/ Hadelin + sklearn examples)

from matplotlib.colors import ListedColormap



# Create color maps

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])



X_set, y_set = features_train, labels_train

xx, yy = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.pcolormesh(xx, yy, clf.predict(np.array([xx.ravel(), yy.ravel()]).T).reshape(xx.shape),

             alpha = 0.75, cmap = cmap_light )

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = cmap_bold(i), label = j)

plt.title('RandomForest (Test set)')

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
# setting features to be used to be Sepal Length and Width

features = iris.iloc[:,0:2].values

labels = iris.iloc[:,4].values



# Encoding the categorical Dependent Variable

from sklearn.preprocessing import LabelEncoder

labelencoder_labels = LabelEncoder()

labels = labelencoder_labels.fit_transform(labels)



# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 43)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

features_train = sc.fit_transform(features_train)

features_test = sc.transform(features_test)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Fitting KNN to the Training set

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Fitting SVM Linear to the Training set

from sklearn.svm import SVC

clf = SVC(kernel='linear', random_state=43)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Fitting SVM RBF to the Training set

from sklearn.svm import SVC

clf = SVC(kernel='rbf', random_state=43)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Fitting Decision Tree to the Training set

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion = 'entropy', random_state=0)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 7, criterion='entropy', random_state=43)

clf.fit(features_train, labels_train)



# Predicting the Test set results

labels_pred = clf.predict(features_test)



# accuracy score

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test, labels_pred))