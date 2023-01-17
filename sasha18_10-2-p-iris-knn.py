import warnings 

warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris

from sklearn.model_selection import cross_validate

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
#Data setup

df = pd.read_csv('../input/iris-data/iris.csv', skiprows=0)

df.sample(10)
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

imputer = imputer.fit(df.iloc[:,:-1])

imputed_data = imputer.transform(df.iloc[:,:-1].values)

df.iloc[:,:-1] = imputed_data



iris = df
iris.iloc[:,5].unique()
iris.head()
from sklearn.preprocessing import LabelEncoder

class_label_encoder = LabelEncoder()



iris.iloc[:,-1] = class_label_encoder.fit_transform(iris.iloc[:,-1])
iris.head()
iris.corr()
iris.var()
sns.pairplot(iris)

plt.show()
import numpy as np

from sklearn.model_selection import train_test_split



# Transform data into features and target

X = np.array(iris.ix[:, 1:5]) 

y = np.array(iris['Species'])



# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

print(X_train.shape)

print(y_train.shape)
print(X_test.shape)

print(y_test.shape)
# loading library

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# instantiate learning model (k = 3)

knn = KNeighborsClassifier(n_neighbors = 3)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, y_pred))



# instantiate learning model (k = 5)

knn = KNeighborsClassifier(n_neighbors=5)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, y_pred))



# instantiate learning model (k = 9)

knn = KNeighborsClassifier(n_neighbors=9)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, y_pred))
# creating odd list of K for KNN

myList = list(range(1,20))



# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold accuracy scores

ac_scores = []



# perform accuracy metrics for values from 1,3,5....19

for k in neighbors:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # predict the response

    y_pred = knn.predict(X_test)

    # evaluate accuracy

    scores = accuracy_score(y_test, y_pred)

    ac_scores.append(scores)



# changing to misclassification error

MSE = [1 - x for x in ac_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()