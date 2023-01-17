# data analysis and wrangling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualize

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

data.head()

#data.tail()
data.info()

data.isna().sum()
y = data.iloc[:,-1].values

X = data.iloc[:,:-1].values
# Split the data into training and test sets with a ratio of 7:3. 

# Set a random seed in order to get fixed results for the purpose of exploring.



from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 123)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Using linear regression to predict model



from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix, accuracy_score



regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test).round(0).astype(int)



accuracy_score(y_pred,y_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_pred,y_test)

# Use K-Neighbor classifier to predict model.



from sklearn.neighbors import KNeighborsClassifier as knn

ks = []

for i in range(1,300):

    knn_regressor = knn(n_neighbors = i,weights = 'distance')

    knn_regressor.fit(X_train,y_train)

    y_pred=knn_regressor.predict(X_test).round(0).astype(int)



    ks.append(accuracy_score(y_test, y_pred))

plt.plot(ks)



max_percent = max(ks)

index = ks.index(max_percent)+1

print(max_percent,index)

# With optimal neighbors



classifier = knn(n_neighbors = 23,weights='distance')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_pred,y_test)
from sklearn.ensemble import RandomForestClassifier

rfc = []

for i in range(1,100):

    classifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 123)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    rfc.append(accuracy_score(y_test, y_pred))



max_percent = max(rfc)

index = rfc.index(max_percent)+1

print(max_percent,index)

plt.plot(rfc)
classifier = RandomForestClassifier(n_estimators = 17, criterion = 'entropy', random_state = 123)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 123)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)