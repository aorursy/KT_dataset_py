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
df = pd.read_csv("../input/ml-zero-datasets-master/ml-zero-datasets-master/p4p5.csv", header=None)
d = df[[0,1,2,3]]
d.columns = ["a","b","c","y"]

d.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(d, d.y, test_size=0.2)
from sklearn import svm
from matplotlib import pyplot as plt

C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
clf.fit(X_train, y_train)
neg = d[d.y==-1]
pos = d[d.y==1]
fig = plt.figure()


plt.scatter(neg.a,neg.c, marker='+')
plt.scatter(pos.a,pos.c, c= 'green', marker='o')

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

plt.plot(xx, yy, 'k-')
plt.scatter(neg.b,neg.c, marker='+')
plt.scatter(pos.b,pos.c, c= 'green', marker='o')

from sklearn.preprocessing import StandardScaler
features = ['a','b','c']
# Separating out the features
x = d.loc[:, features].values
# Separating out the target
y = d.loc[:,['y']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#from sklearn.manifold import TSNE
#X_tsne = TSNE(learning_rate=100).fit_transform(x[:10,])
x[:10,]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, d[['y']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [-1,1]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['y'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
pca.explained_variance_ratio_
from sklearn import svm
from matplotlib import pyplot as plt

C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel = 'linear',  gamma=0.7, C=C )
svc.fit(X_train, y_train)

z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

tmp = np.linspace(-2,2,51)
x,y = np.meshgrid(tmp,tmp)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
# Plot stuff
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(X_train[X_train.y==-1].a, X_train[X_train.y==-1].b, X_train[X_train.y==-1].c,'ob')
ax.plot3D(X_train[X_train.y==1].a, X_train[X_train.y==1].b, X_train[X_train.y==1].c,'sr')
ax.plot_surface(x, y, z(x,y))
plt.show()
X_train[X_train.y==-1]
import numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.datasets.base import load_iris
from sklearn.manifold.t_sne import TSNE
from sklearn.linear_model.logistic import LogisticRegression

# replace the below by your data and model
iris = load_iris()
X,y = iris.data, iris.target
X_Train_embedded = TSNE(n_components=2).fit_transform(X)
print(X_Train_embedded.shape)
model = LogisticRegression().fit(X,y)
y_predicted = model.predict(X)
# replace the above by your data and model

# create meshgrid
resolution = 100 # 100x100 background pixels
X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:,0]), np.max(X_Train_embedded[:,0])
X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:,1]), np.max(X_Train_embedded[:,1])
xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

# approximate Voronoi tesselation on resolution x resolution grid using 1-NN
background_model = KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted) 
voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
voronoiBackground = voronoiBackground.reshape((resolution, resolution))

#plot
plt.contourf(xx, yy, voronoiBackground)
plt.scatter(X_Train_embedded[:,0], X_Train_embedded[:,1], c=y)
plt.show()
