import numpy as np 

import pandas as pd



df = pd.read_csv("../input/mushrooms.csv")

df.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



predictors = list(df.columns.values)

ftrs = list(predictors)

ftrs.remove('class')



print(ftrs)
for i in ftrs:

    df[i] = le.fit_transform(df[i])

    

X = df[ftrs]

y = df['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1000)

print("There are {} samples in the training set and {} in the test set".format(X_train.shape[0], X_test.shape[0]))

df.head()
# SVC model

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train, y_train)



print('The accuracy of the svm classifier on the training data is {:.2f} out of 1'.format(svm.score(X_train, y_train)))

print('The accuracy of the svm classifier on the test data is {:.2f} out of 1'.format(svm.score(X_test, y_test)))
# K-nn model

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

knn.fit(X_train, y_train)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train, y_train)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test, y_test)))