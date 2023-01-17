from sklearn.neighbors import KNeighborsClassifier



from sklearn import datasets

from skimage import exposure

import numpy as np



import matplotlib.pyplot as plt







digits = datasets.load_digits()

X = digits.data

y = digits.target
X
X.shape
y.shape
plt.gray() 

#matshow - Display an array as a matrix in a new figure window

plt.matshow(digits.images[3])  

plt.matshow(digits.images[666])  
plt.figure(figsize=(16, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X[i,:].reshape([8,8]), cmap='gray');
from sklearn.model_selection import train_test_split

from sklearn import datasets, neighbors, linear_model



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





knn = neighbors.KNeighborsClassifier()

logistic = linear_model.LogisticRegression()



knn_model = knn.fit(X_train, y_train)

logistic_model =  logistic.fit(X_train, y_train)
print(knn_model.score(X_test, y_test))
print(logistic_model.score(X_test, y_test))
#prediction

print(knn_model.predict(X_test[0].reshape(1,-1)))
y_test[0]