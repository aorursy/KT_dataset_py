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
#import all the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Read the file

iris = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.head()
print('The different types of Iris Flower are ',np.unique(iris['Species']))
print('The shape of the IRIS dataset', iris.shape)

#It has 150 rows and 6 columns
#The information of iris dataset is as follows

iris.info()

#no missing data
iris.describe()
#Drop the unnecesary column



iris.drop('Id', axis=1, inplace=True)
iris.head()
#iris['SepalLengthCm'].hist(), iris['SepalWidthCm'].hist()

fig,ax = plt.subplots(2,2,figsize=(10,6))

ax[0,0].hist(iris['SepalLengthCm'])

ax[0,1].hist(iris['SepalWidthCm'])

ax[1,0].hist(iris['PetalLengthCm'])

ax[1,1].hist(iris['PetalWidthCm'])



ax[0,0].set(xlabel='Sepal Length')

ax[0,1].set(xlabel='Sepal Width')

ax[1,0].set(xlabel = 'Petal Length')

ax[1,1].set(xlabel = 'Petal Width')

iris.plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm')
#setosa flower

fig = iris[iris['Species'] == 'Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue',label='Setosa')



#Versicolor



iris[iris['Species'] == 'Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Versicolor', ax=fig)



#virginica



iris[iris['Species'] == 'Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Virginica', ax=fig)



fig.set_title('SepalLength vs Sepal Width')
#setosa flower

fig = iris[iris['Species'] == 'Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue',label='Setosa')



#Versicolor



iris[iris['Species'] == 'Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red',label='Versicolor', ax=fig)



#virginica



iris[iris['Species'] == 'Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green',label='Virginica', ax=fig)



fig.set_title('Petal Length vs Petal Width')
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)
#Correlation matrix



sns.heatmap(iris.corr(),cmap=sns.diverging_palette(20, 220, n=200),annot=True)
#split the train-test data



y = iris['Species']

X = iris.drop('Species',axis=1)



#Label Encoder



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
#standardise the features



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(C=100.0)

lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)



from sklearn.metrics import accuracy_score



print('Accuracy score : %.2f' % accuracy_score(y_test, y_pred))
from sklearn.svm import SVC



svm = SVC(kernel = 'rbf', random_state=1, gamma=0.2, C=100.0)

svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)



from sklearn.metrics import accuracy_score



print('Accuracy score : %.2f' % accuracy_score(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion = 'gini',

                             random_state=1,

                              max_depth=4

                             )



tree.fit(X_train_std, y_train)
y_pred = tree.predict(X_test_std)

print('Accuracy Score %0.2f' % accuracy_score(y_pred, y_test))
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(criterion='gini',

                               n_estimators=25,

                               random_state=1,

                               n_jobs=2)



forest.fit(X_train_std, y_train)

y_pred = forest.predict(X_test_std)

print('Accuracy score %.2f' %accuracy_score(y_pred,y_test))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5,

                          p=2,

                          metric = 'minkowski')

knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)

print('Accuracy score %.2f' %accuracy_score(y_pred,y_test))