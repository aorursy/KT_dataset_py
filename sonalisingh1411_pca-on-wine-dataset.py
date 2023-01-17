#Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import os

print(os.listdir("../input"))

#Reading the dataset..

df=pd.read_csv("../input/wineQualityReds.csv")
#Checking the starting "5" values.

df.head()
#Checking if there is any existing null value or not

df.isnull().sum()
#Checking the unique values from "quality column"

df["quality"].unique()
#Count the unique values in "quality column"

df["quality"].value_counts()
#Plot for quality

df["quality"].value_counts().plot.bar(color='Yellow')

plt.xlabel("Quality score")

plt.legend()
#Checking the dimensions

df.shape


#Separating dependent and independent variable.

X = df.iloc[:, 1:12].values

y = df.iloc[:, 12].values





print(X)
print(y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)





# Applying PCA

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)

X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_



print(explained_variance)
#Fitting Logistic regression into dataset

from sklearn.linear_model import LogisticRegression

lr_c=LogisticRegression(random_state=0)

lr_c.fit(X_train,y_train)

lr_pred=lr_c.predict(X_test)

lr_cm=confusion_matrix(y_test,lr_pred)

print("The accuracy of  LogisticRegression is:",accuracy_score(y_test, lr_pred))
print(lr_cm)
#Fitting Randomforest into dataset

from sklearn.ensemble import RandomForestClassifier

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(X_train,y_train)

rdf_pred=rdf_c.predict(X_test)

rdf_cm=confusion_matrix(y_test,rdf_pred)

print("The accuracy of RandomForestClassifier is:",accuracy_score(rdf_pred,y_test))
print(rdf_cm)
#Fitting KNN into dataset

from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

knn_pred=knn.predict(X_test)

knn_cm=confusion_matrix(y_test,knn_pred)

print("The accuracy of KNeighborsClassifier is:",accuracy_score(knn_pred,y_test))
print(knn_cm)
#Fitting Naive bayes into dataset

from sklearn.naive_bayes import GaussianNB







gaussian=GaussianNB()

gaussian.fit(X_train,y_train)

bayes_pred=gaussian.predict(X_test)

bayes_cm=confusion_matrix(y_test,bayes_pred)

print("The accuracy of naives bayes is:",accuracy_score(bayes_pred,y_test))
print(bayes_cm)