import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

print(os.listdir("../input"))
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train = pd.DataFrame(train_data)
points = train.iloc[:,:2].values
category = train.iloc[:,2].values

test = pd.DataFrame(test_data)
test_points = test.iloc[:,:2].values
test_category = test.iloc[:,2].values
def plotPoints(points, category):
    for i in range(len(points)):
        c = "r"
        if category[i] == 0:
             c = "b"
        plt.scatter(points[i][0],points[i][1], s = 10, c = c)
plotPoints(points,category)
plotPoints(test_points, test_category)
#Source: Competition --> Discussion
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

# Compare
# http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA']) 
cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000']) 

def plot_decision_boundary(model,X,y):
    h = .02  # step size in the mesh
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
              edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(model.__class__.__name__)

    plt.show()
x_train, x_test, y_train, y_test = train_test_split(points, category, test_size = 0.2, random_state = 0)
from sklearn.neural_network import MLPClassifier

# Neural Network (Multi-layer Perceptron)

model_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(64, 2), random_state=1)
model_mlp.fit(x_train, y_train)
plot_decision_boundary(model_mlp, x_train, y_train)

print('MLP Classifier')
print ('train accuracy: {}'.format(model_mlp.score(x_train, y_train)))
print ('validation accuracy: {}'.format(model_mlp.score(x_test, y_test)))
predictions_mlp = model_mlp.predict(test_points)
plotPoints(test_points, predictions_mlp)

print ('test accuracy: {}'.format(model_mlp.score(test_points, test_category)))
