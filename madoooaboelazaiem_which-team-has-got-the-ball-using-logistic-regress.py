import numpy as np

import matplotlib.pyplot as plt



# Import the linear regression model from sklearn

from sklearn.linear_model import LogisticRegression

from sklearn import datasets



# import some data to play with

iris = datasets.load_iris()

X = iris.data[:100, :2]  # we only take the first two features (x,y).



Y = iris.target[:100]



print(X)

print(Y)

print(len(Y))
plt.scatter(X[:, 0], X[:, 1],c=Y[:100])
model = LogisticRegression()  # creating a new class of the Model to learn



model.fit(X,Y)        #the function used to fit/train our model
newPoint = [[5.5,3.1]]

predictions = model.predict(newPoint) # Get the model's predictions

print("my prediction is that the point belongs to class :",predictions[0]) #printing the prediction

#plotting the decision boundry " learnt function"

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5

y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

h = .02  # step size in the mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(4, 3))

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X[:, 0], X[:, 1],c=Y[:100] ,edgecolors='k', cmap=plt.cm.Paired)

plt.scatter(newPoint[0][0], newPoint[0][1], edgecolors='w')