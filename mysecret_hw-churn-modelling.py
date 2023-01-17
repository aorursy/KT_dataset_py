import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df.head()
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

df = df.dropna()
df.head()
from  sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories='auto')

feature_arr = ohe.fit_transform(df[['Geography','Gender']]).toarray()

feature_labels = ohe.categories_



feature_labels = np.array(feature_labels).ravel()

feature_labels =  np.concatenate((feature_labels), axis=None)

features = pd.DataFrame(feature_arr, columns=feature_labels)

print(features)
df = df.drop(columns=['Exited', 'Geography', 'Gender'])
df.head()
df = pd.concat([features,df], axis=1, sort=False)
df.head()
y = df.iloc[:,10].values

X = df.iloc[:,:].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
from sklearn.preprocessing import StandardScaler

sts = StandardScaler()

X_train = sts.fit_transform(X_train)

X_test = sts.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1)) 
from sklearn import svm



clf_svm = svm.SVC(kernel='rbf', tol=0.01)



clf_svm.fit(X_train,y_train)

y_pred = clf_svm.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1)) 
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print("Performance:")

print(" >accuracy = " + str(accuracy))

print(" >precision = " + str(precision))

print(" >recall = " + str(recall))

print(" >f1 = " + str(f1)) 