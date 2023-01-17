import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import preprocessing, neighbors

df = pd.read_csv("../input/breast-cancer-wisconsin.data.csv")



#some missing values are present. denoted by ?

df.replace('?',-99999,inplace=True)

df.drop(['id'],1,inplace=True)
# Creating input and output data for training

X = np.array(df.drop(['class'],1))

y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Fitting the data into the model

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train,y_train)
#Finding the accuracy

accuracy = clf.score(X_test,y_test)

print(accuracy)
#Predictions

##Creating some example data for predictions



example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,1,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures),-1)

predicition = clf.predict(example_measures)

print(predicition)