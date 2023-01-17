# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt

import numpy as np

from itertools import product



#Create a variable named data that is a list that contains the four possible inputs to an AND gate.





data = [[0, 0], [0, 1], [1, 0], [1, 1]]

labels = [0, 0, 0, 1]



plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels) 

#The third parameter "c = labels" will make the points with label 1 a different color than points with label 0.

plt.show()



#Why is this data linearly separable?



#Let’s build a perceptron to learn AND.



classifier = Perceptron(max_iter = 40)

classifier.fit(data, labels)

print(classifier.score(data, labels))



#Note that it is pretty unusual to train and test on the same dataset. In this case, since there are only four possible inputs to AND, we’re stuck training on every possible input and testing on those same points.





#Your perceptron should have 100% accuracy! You just taught it an AND gate!
#Let’s change the labels so your data now represents an XOR gate. The label should be a 1 only if one of the inputs is a 1. What is the accuracy of the perceptron now? Is the data linearly separable?







data = [[0, 0], [0, 1], [1, 0], [1, 1]]

labels = [0, 1, 1, 0]



plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)

plt.show()



classifier = Perceptron(max_iter = 40)

classifier.fit(data, labels)

print(classifier.score(data, labels))

#Try changing the data to represent an OR gate. 





data = [[0, 0], [0, 1], [1, 0], [1, 1]]

labels = [0, 1, 1, 1]



plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)

plt.show()



classifier = Perceptron(max_iter = 40)

classifier.fit(data, labels)

print(classifier.score(data, labels))

#We know the perceptron has been trained correctly, but let’s try to visualize what decision boundary it is making. Reset your labels to be representing an AND gate.



data = [[0, 0], [0, 1], [1, 0], [1, 1]]

labels = [0, 0, 0, 1]



plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)

plt.show()



classifier = Perceptron(max_iter = 40)

classifier.fit(data, labels)

print(classifier.score(data, labels))



#Let’s first investigate the classifier’s .decision_function() method. Given a list of points, this method returns the distance those points are from the decision boundary. The closer the number is to 0, the closer that point is to the decision boundary.
#A decision boundary is the line that determines whether the output should be a 1 or a 0. Points that fall on one side of the line will be a 0 and points on the other side will be a 1.



print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))



#If your data is representing an AND gate, you should see that the point [1, 1] is closer to the decision boundary than [0, 0]. The point at [0.5, 0.5] is pretty close to the decision boundary as well.





#Even though an input like [0.5, 0.5] isn’t a real input to an AND logic gate, we can still check to see how far it is from the decision boundary.



#We could also do this to the point [0, 0.1], [0, 0.2] and so on. If we do this for a grid of points, we can make a heat map that reveals the decision boundary.
plt.figure(figsize = (12,7))

x_values = np.linspace(0, 1, 100)

y_values = np.linspace(0, 1, 100)

point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)

abs_distances = [abs(i) for i in distances]

distance_matrix = np.reshape(abs_distances, (100,100))

heatmap = plt.pcolormesh(x_values, y_values, distance_matrix)

plt.colorbar(heatmap)

plt.show()

#You should see a purple line where the distances are 0. That’s the decision boundary!
#Try other logic gates, e.g., OR, XOR.