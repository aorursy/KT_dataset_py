#importing required libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#reading csv file into data frame

df = pd.read_csv('../input/KNN_Project_Data')

df.head()
#performing exploratory data analysis

sns.set_style('darkgrid')

sns.pairplot(df, hue='TARGET CLASS', palette='colorblind').fig.set_size_inches(15,15)
#standardizing numerical variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_scaled = pd.DataFrame(scaled_features, columns=df.columns[:-1])

df_scaled.head()
#splitting dataframe into train and test

X = df.drop('TARGET CLASS', axis=1)

y = df['TARGET CLASS']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101)
#Using KNN for this prediction

from sklearn.neighbors import KNeighborsClassifier

#trying with k=1

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

#evaluating model performance

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, prediction))

confusion_matrix(y_test, prediction)
#Using elbow method to find optimal k

error_rate=[]

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_train, y_train)

    prediction_i = knn.predict(X_test)

    error_rate.append(np.mean(prediction_i != y_test))

sns.set_style('darkgrid')

plt.figure(figsize=(15,8))

plt.plot(range(1,40), error_rate, marker='o', markerfacecolor='red', linestyle='dashed', color='green', markersize=10)

plt.title('Error Rate VS K')

plt.xlabel('K')

plt.ylabel('Error Rate')
#Good values for k appear to be 24, 32, 36

#trying k = 24

knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

print(classification_report(y_test, prediction))

confusion_matrix(y_test, prediction)

#The score improves significantly from 74 to 82%.
#trying k = 32

knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

print(classification_report(y_test, prediction))

confusion_matrix(y_test, prediction)
#trying k = 36

knn = KNeighborsClassifier(n_neighbors=36)

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)

print(classification_report(y_test, prediction))

confusion_matrix(y_test, prediction)