#Imports
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
#Loading train-data and test-data
traindf = pd.read_csv("../input/train.csv")
testdf = pd.read_csv("../input/test.csv")

X_Train = traindf[["X","Y"]].values
Y_Train = traindf[["class"]].values

X_Test = testdf[["X","Y"]].values
Y_Test = testdf[["class"]].values
#Plotting train-data
colors = {0:'red',1:'blue'}
plt.figure(figsize=(5,5))
plt.scatter(X_Train[:,0],X_Train[:,1],c=traindf["class"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#Calculating average distance from center-point (0,0)
def get_average_distance(train, label, encoding):
    distance = 0
    j = 0
    for i in range(len(train[:,0])):
            if label[i] == encoding:    
                j+=1
                x = train[i,0]
                y = train[i,1]
                distance += math.sqrt(x**2+y**2)
    return distance/j
#Training
distance_red = get_average_distance(X_Train,Y_Train,0)
distance_blue = get_average_distance(X_Train,Y_Train,1)
#Prediction
def classify(x,y):
    encoding = 0
    distance = math.sqrt(x**2+y**2)
    #50/50 Weights
    if distance < distance_blue + (distance_red - distance_blue) / 2:
        encoding = 1
    return encoding 
#Testing
test_result = list()
for i in range(len(X_Test[:,0])):
    x = X_Test[i,0]
    y = X_Test[i,1]
    test_result.append(classify(x,y))

#Calculating test-data accuracy
error = 0
for i in range(len(Y_Test)):
    if test_result[i] != Y_Test[i]:
        error+=1

accuracy = 100 - 100.0 / len(Y_Test) * error   
print("Accuracy: ",accuracy)
#Plotting test-data classification and boundary
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
X = X_Test
y = test_result 
h = .02  # step size in the mesh
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

points = np.c_[xx.ravel(),yy.ravel()]

Z = np.zeros(len(points))
for i in range(len(points)):
    temp_x = points[i][0]
    temp_y = points[i][1]
    Z[i] = classify(temp_x,temp_y)
    

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(5,5))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=10)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()