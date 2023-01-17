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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/Iris.csv')
print(data.columns)
feature_names = [ 'PetalLengthCm', 'PetalWidthCm','SepalLengthCm', 'SepalWidthCm']
X =data[feature_names]
y=data['Species']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
df = pd.DataFrame(data, columns = ['PetalLengthCm', 'PetalWidthCm','SepalLengthCm', 'SepalWidthCm','Species'])
def score_to_numeric(x):
    if x=='Iris-virginica':
        return 3
    if x=='Iris-setosa':
        return 2
    if x=='Iris-versicolor':
        return 1
df['score_num'] = df['Species'].apply(score_to_numeric)   
#df['score_num']

feature_names = [ 'PetalLengthCm', 'PetalWidthCm','SepalLengthCm', 'SepalWidthCm']
X =data[feature_names]
y=df['score_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
def plot_fruit_knn(X, y, n_neighbors):
    X_mat = X[['PetalLengthCm', 'PetalWidthCm']].as_matrix()
    y_mat = y.as_matrix()
# Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
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
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FF0000', label='Iris-versicolor')
    patch1 = mpatches.Patch(color='#00FF00', label='Iris-setosa')
    patch2 = mpatches.Patch(color='#0000FF', label='Iris-verginica')
    plt.legend(handles=[patch0, patch1, patch2])
    plt.xlabel('PetalLengthCm')
    plt.ylabel('PetalWidthCm')
    plt.title("3-Class classification ")
    plt.subplots_adjust(bottom=0.1, right=1.9, top=1.5) 
    plt.show()

plot_fruit_knn(X_train, y_train, n_neighbors=9)
    
    
    