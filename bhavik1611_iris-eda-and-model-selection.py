import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



sns.set(style="white", color_codes=True, rc={'figure.figsize':(11.7,8.27)})
cd /kaggle/input/iris
!ls
iris = pd.read_csv('Iris.csv')
iris.dtypes
iris.describe()
iris.head()
print('Dataset has {n} instances.'.format(n = iris.shape[0]))

print('Dataset has {n} columns.'.format(n = iris.shape[1]))

cols = ', '.join(iris.columns) + '.'

print('Dataset has the following columns:',cols)
iris.drop(['Id'], axis=1, inplace=True)
iris['Species'].value_counts()
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
iris.plot(kind="scatter", x="PetalLengthCm", y="PetalWidthCm")
sns.FacetGrid(iris, hue="Species", size=7).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
sns.FacetGrid(iris, hue="Species", size=7).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
sns.boxplot(x="Species", y="PetalWidthCm", data=iris)
sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
sns.boxplot(x="Species", y="SepalWidthCm", data=iris)
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalWidthCm").add_legend()
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "SepalLengthCm").add_legend()
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "SepalWidthCm").add_legend()
sns.pairplot(iris, hue="Species", size=3.5)
X = iris.drop(['Species'], axis=1)

Y = iris['Species']
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

  

Y= le.fit_transform(Y) 
print(Y.shape)

Y
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)
print(X.shape)

print('\nMinMax Scaled Inputs:\n', X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
train_acc = []

test_acc = []

i = 0
from sklearn.linear_model import LogisticRegression

logR = LogisticRegression(random_state = 0).fit(X_train, y_train)



train_acc.append(np.mean(logR.predict(X_train) == y_train)*100)

test_acc.append(np.mean(logR.predict(X_test) == y_test)*100)

print('\nTrain Accuracy: {}\nTest Accuracy: {}'.format(train_acc[i], test_acc[i]))

i += 1

y_pred = logR.predict(X_test)
#F1 score

from sklearn.metrics import f1_score, confusion_matrix

print('\nTrain Confusion Matrix\n', confusion_matrix(y_train, logR.predict(X_train)))

print('\nTest Confusion Matrix\n', confusion_matrix(y_test, y_pred))

print('\nF1-score\n', f1_score(y_test, y_pred, average=None))

from sklearn import tree

dTree = tree.DecisionTreeClassifier(random_state=0).fit(X_train, y_train)



train_acc.append(np.mean(dTree.predict(X_train) == y_train)*100)

test_acc.append(np.mean(dTree.predict(X_test) == y_test)*100)

print('\nTrain Accuracy: {}\nTest Accuracy: {}'.format(train_acc[i], test_acc[i]))

i += 1

y_pred = dTree.predict(X_test)
import graphviz 

dot_data = tree.export_graphviz(dTree, out_file=None) 

graph = graphviz.Source(dot_data) 

dot_data = tree.export_graphviz(dTree, out_file=None,feature_names=iris.columns[:-1], 

                                class_names=['Iris-setosa', 'Iris-vesicolor', 'Iris-viriginica'],

                                filled=True, rounded=True, special_characters=True)  

graph = graphviz.Source(dot_data)  

graph 
#F1 score

from sklearn.metrics import f1_score, confusion_matrix

print('\nTrain Confusion Matrix\n', confusion_matrix(y_train, dTree.predict(X_train)))

print('\nTest Confusion Matrix\n', confusion_matrix(y_test, y_pred))

print('\nF1-score\n', f1_score(y_test, y_pred, average=None))
from sklearn.ensemble import RandomForestClassifier

ranF = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)



train_acc.append(np.mean(ranF.predict(X_train) == y_train)*100)

test_acc.append(np.mean(ranF.predict(X_test) == y_test)*100)

print('\nTrain Accuracy: {}\nTest Accuracy: {}'.format(train_acc[i], test_acc[i]))

i += 1

y_pred = ranF.predict(X_test)
#F1 score

from sklearn.metrics import f1_score, confusion_matrix

print('\nTrain Confusion Matrix\n', confusion_matrix(y_train, ranF.predict(X_train)))

print('\nTest Confusion Matrix\n', confusion_matrix(y_test, y_pred))

print('\nF1-score\n', f1_score(y_test, y_pred, average=None))

from sklearn import svm

svmPre = svm.SVC().fit(X_train, y_train)



train_acc.append(np.mean(svmPre.predict(X_train) == y_train)*100)

test_acc.append(np.mean(svmPre.predict(X_test) == y_test)*100)

print('\nTrain Accuracy: {}\nTest Accuracy: {}'.format(train_acc[i], test_acc[i]))

i += 1

y_pred = svmPre.predict(X_test)
#F1 score

from sklearn.metrics import f1_score, confusion_matrix

print('\nTrain Confusion Matrix\n', confusion_matrix(y_train, svmPre.predict(X_train)))

print('\nTest Confusion Matrix\n', confusion_matrix(y_test, y_pred))

print('\nF1-score\n', f1_score(y_test, y_pred, average=None))