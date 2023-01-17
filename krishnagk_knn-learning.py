import numpy as np 
import pandas as pd 
from pandas.tools.plotting import scatter_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("../input/voice.csv")
data.info()
data.head(n=10)
data.hist()
plt.subplots_adjust(bottom=.051, right=1.9, top=1.5)
plt.show()
import seaborn as sns
sns.distplot(data['modindx'])
data.label = [1 if each == "male" else 0 for each in data.label]

data.head() # check if binary conversion worked
# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'label'], data['label'], stratify=data['label'], random_state=66)



# split data into X and y
#X = data[:,0:20]
#y = data[:,20]
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# plot feature importance
plot_importance(model)
pyplot.show()
import seaborn as sns
sns.distplot(data['meanfun'])
import seaborn as sns
sns.distplot(data['IQR'])
import seaborn as sns
sns.distplot(data['sfm'])
sns.distplot(data['Q25'])

import seaborn as sns
sns.distplot(data['minfun'])
#sns.distplot(data['maxdom'])
x = data.meanfun
y = data.IQR
plt.scatter(x, y ,s = 10*x,data = data,cmap = "plasma",c = data.meanfun)
plt.xlabel("meanfun")
plt.ylabel("IQR")
plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5)
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches

feature_names = [ 'meanfun', 'IQR','sfm']
X =data[feature_names]
y=data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
def plot_fruit_knn(X, y, n_neighbors):
    X_mat = X[['meanfun','IQR']].as_matrix()
    y_mat = y.as_matrix()
# Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00'])
    clf = KNeighborsClassifier()
    clf.fit(X_mat, y_mat)       
    
# Plot the decision boundary by assigning a color in the color map
    # to each mesh point.
    
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 30
    
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1],s=plot_symbol_size, c=y,  cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(0,0.3)
    plt.ylim(-0.1, 0.3)
    patch0 = mpatches.Patch(color='#FF0000', label='Female')
    patch1 = mpatches.Patch(color='#00FF00', label='Male')
    
    plt.legend(handles=[patch0, patch1])
    plt.xlabel('meanfun')
    plt.ylabel('IQR')
    plt.title("2-Class classification ")
    plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5) 
    plt.show()

plot_fruit_knn(X_train, y_train, n_neighbors=8)
    