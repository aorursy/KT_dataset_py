# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pdp

from sklearn.linear_model import LogisticRegression  #Logistic Regression

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  #K nearest neighbours

from sklearn import svm  #Support Vector Machine

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #Decision Tree



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

Iris=pd.read_csv('/kaggle/input/iris/Iris.csv')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Iris.columns
y=Iris['Species']

y.head(10)
Flower_feature=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

X=Iris[Flower_feature]

X.head(10)
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=.8, random_state = 0)

print(train_X.shape)

print(val_X.shape)

print(train_y.shape)

print(val_y.shape)
train_X.head(10)
train_y.head(10)
for i in train_X.columns:

    plt.hist(train_X[i])

    plt.title(i)

    plt.show()
print(train_X.corr());

sns.heatmap(train_X.corr());
print(train_y.value_counts())

species_count = train_y.value_counts()

sns.set(style="darkgrid")

sns.barplot(species_count.index, species_count.values, alpha=0.9,palette="PuOr")

plt.title('Frequency of Species')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Species', fontsize=12)

plt.show()
labels = train_y.astype('category').cat.categories.tolist()

counts = train_y.value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot

ax1.axis('equal')

plt.show()
from sklearn import preprocessing

X_train_scaled = preprocessing.scale(train_X)

X_val_scaled = preprocessing.scale(val_X)

model = LogisticRegression()

model.fit(X_train_scaled,train_y)

prediction=model.predict(X_val_scaled)

print("The model accuracy for Logisitc Regression is",metrics.accuracy_score(prediction,val_y))
model=DecisionTreeClassifier()

model.fit(X_train_scaled,train_y)

prediction=model.predict(X_val_scaled)

print("The model accuracy of the Decision Tree is",metrics.accuracy_score(prediction,val_y))
model=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

model.fit(X_train_scaled,train_y)

prediction=model.predict(X_val_scaled)

print('The model accuracy of the KNN is',metrics.accuracy_score(prediction,val_y))
a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(X_train_scaled,train_y)

    prediction=model.predict(X_val_scaled)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,val_y)))

plt.plot(a_index, a)

plt.xticks(x)
model=svm.SVC(kernel='linear')

model.fit(X_train_scaled,train_y)

prediction=model.predict(X_val_scaled)

print('The model accuracy of linear SVM is:',metrics.accuracy_score(prediction,val_y))
model=svm.SVC(kernel='poly',

             degree=3)

model.fit(X_train_scaled,train_y)

prediction=model.predict(X_val_scaled)

print('The model accuracy of linear SVM is:',metrics.accuracy_score(prediction,val_y))