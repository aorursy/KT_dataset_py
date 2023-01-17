# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import scipy

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/Iris.csv") 

df.head(2)
df.info()
df.describe()
# Displaying the number of Nan's in each column

labels= []

value = []

for col in df.columns:

    labels.append(col)

    value.append(df[col].isnull().sum())

    print(col,value[-1])

# Counting the number of species in the dataset

df['Species'].value_counts()
rel_df = pd.read_csv("../input/Iris.csv") 

del rel_df['Id']

rel = rel_df.corr()

sns.heatmap(rel,square= True)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, [2, 3]]

y = iris.target
# splitting the dataset

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)



print('There are {} samples in the training set and {} samples in the test set'.format(

X_train.shape[0], X_test.shape[0]))

print()
# scaling the dataset

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)



print('After standardizing our features, the first 5 rows of our data now look like this:\n')

X_train_print = pd.DataFrame(X_train_std)

X_train_print.head(5)

df.head(5)
from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train_std, y_train)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))



print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

knn.fit(X_train_std, y_train)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train_std, y_train)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test_std, y_test)))

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=2)

model.fit(X_train_std,y_train)

print('Training Accuracy Sepal = {}'.format(model.score(X_train_std, y_train)))

print('Testing  Accuracy Sepal = {}'.format(model.score(X_test_std, y_test)))


