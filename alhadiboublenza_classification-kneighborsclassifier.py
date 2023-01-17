import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        filename = os.path.join(dirname, filename)

        

#read in the data using pandas

df = pd.read_csv(filename)

#check data has been read in properly

df.head()
df.shape
df.describe
X = df.drop(columns=['Outcome'])
X.head()
y = df['Outcome'].values
y[0:5]
from sklearn.model_selection import train_test_split

#split dataset into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
len(X_train)
len(X_test)
y_train
y_test
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier

knn = KNeighborsClassifier(n_neighbors = 10)

# Fit the classifier to the data

knn.fit(X_train,y_train)
#show first 5 model predictions on the test data

knn.predict(X_test)[0:5]
y_test
knn.score(X_test, y_test)
from sklearn.model_selection import cross_val_score

import numpy as np

#create a new KNN model

knn_cv = KNeighborsClassifier(n_neighbors=3)

#train model with cv of 5 

cv_scores = cross_val_score(knn_cv, X, y, cv=5)

#print each cv score (accuracy) and average them

print(cv_scores)

print('cv_scores mean:{}'.format(np.mean(cv_scores)))
error_rate = []

pred_i = 0

# Might take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    error_rate.append(knn.score(X_test, y_test))
import matplotlib.pyplot as plt



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Score vs. K Value')

plt.xlabel('K')

plt.ylabel('Score')