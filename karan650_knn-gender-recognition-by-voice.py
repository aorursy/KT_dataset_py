# K-Nearest Neighbors (K-NN)



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

# Importing the dataset

df = pd.read_csv('../input/voice.csv')

df.head()
df.isnull().any()
g = sns.pairplot(df, kind='scatter', hue = 'label');
df.columns
# selecting only the main features

X = df[['IQR','meanfun','Q25','kurt','maxfun','skew']].values

y = df.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting K-NN to the Training set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2  )

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix\n" ,cm)
from sklearn.metrics import accuracy_score

print("Accuracy is",(accuracy_score(y_test, y_pred)*(100)).round(2))