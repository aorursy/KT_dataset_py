import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



df = pd.read_csv("../input/Iris.csv")



# visualize SepalLengthCm vs SepalWidthCm

plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"])

plt.xlabel('SepalLengthCm')

plt.ylabel('SepalWidthCm')

plt.show()



# visualize SepalLengthCm vs PetalLengthCm

plt.scatter(df["SepalLengthCm"], df["PetalLengthCm"])

plt.xlabel('SepalLengthCm')

plt.ylabel('PetalLengthCm')

plt.show()



# visualize SepalLengthCm vs PetalWidthCm

plt.scatter(df["SepalLengthCm"], df["PetalWidthCm"])

plt.xlabel('SepalLengthCm')

plt.ylabel('PetalWidthCm')

plt.show()



# visualize SepalWidthCm vs PetalLengthCm

plt.scatter(df["SepalWidthCm"], df["PetalLengthCm"])

plt.xlabel('SepalWidthCm')

plt.ylabel('PetalLengthCm')

plt.show()



# visualize SepalWidthCm vs PetalWidthCm

plt.scatter(df["SepalWidthCm"], df["PetalWidthCm"])

plt.xlabel('SepalLengthCm')

plt.ylabel('PetalWidthCm')

plt.show()



X = np.array(df.loc[:,"SepalLengthCm":"PetalWidthCm"])

y = np.array(df["Species"])





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print (accuracy_score(y_test, pred))



# empty list that will hold cv scores

cv_scores = []

myList = list(range(1,50))



# perform 10-fold cross validation

for k in myList:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

    

# changing to misclassification error

MSE = [1 - x for x in cv_scores]



# determining best k

optimal_k = myList[MSE.index(min(MSE))]

print ("The optimal number of neighbors is %d" % optimal_k)

plt.plot(myList, MSE)

plt.xlabel('Neighbors')

plt.ylabel('Error')

plt.show()