# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pylab as plt

from sklearn import metrics

metrics.homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))

print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))

print (metrics.completeness_score([0, 0, 1, 1], [1, 1, 0, 0]))

print(metrics.completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))

print(metrics.completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))

print(metrics.completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))

print(metrics.completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
print (metrics.v_measure_score([0, 0, 1, 1], [0, 0, 1, 1]))

print (metrics.v_measure_score([0, 0, 1, 1], [1, 1, 0, 0]))
print("%.3f" % metrics.completeness_score([0, 1, 2, 3], [0, 0, 0, 0]))

print("%.3f" % metrics.homogeneity_score([0, 1, 2, 3], [0, 0, 0, 0]))

print("%.3f" % metrics.v_measure_score([0, 1, 2, 3], [0, 0, 0, 0]))

print("%.3f" % metrics.v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))

print("%.3f" % metrics.v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
print("%.3f" % metrics.v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))

print("%.3f" % metrics.v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
print("%.3f" % metrics.v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
print("%.3f" % metrics.v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
import numpy as np



#Create some data

MAXN=40

X = np.concatenate([1.25*np.random.randn(MAXN,2), 5+1.5*np.random.randn(MAXN,2)])

X = np.concatenate([X,[8,3]+1.2*np.random.randn(MAXN,2)])

X.shape
#Just for visualization purposes, create the labels of the 3 distributions

y = np.concatenate([np.ones((MAXN,1)),2*np.ones((MAXN,1))])

y = np.concatenate([y,3*np.ones((MAXN,1))])



plt.subplot(1,2,1)

plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')

plt.title('Data as were generated')



plt.subplot(1,2,2)

plt.scatter(X[:,0],X[:,1],color='r')

plt.title('Data as the algorithm sees them')



plt.savefig("/kaggle/working/sample.png",dpi=300, bbox_inches='tight')



from sklearn import cluster



K=3 # Assuming to be 3 clusters!



clf = cluster.KMeans(init='random', n_clusters=K)

clf.fit(X)
print (clf.labels_) # or

print (clf.predict(X)) # equivalent
print (X[(y==1).ravel(),0]) #numpy.ravel() returns a flattened array

print (X[(y==1).ravel(),1])
plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')



fig = plt.gcf()

fig.set_size_inches((6,5))
x = np.linspace(-5,15,200)

XX,YY = np.meshgrid(x,x)

sz=XX.shape

data=np.c_[XX.ravel(),YY.ravel()]

# c_ translates slice objects to concatenation along the second axis.
Z=clf.predict(data) # returns the labels of the data

print (Z)
# Visualize space partition

plt.imshow(Z.reshape(sz), interpolation='bilinear', origin='lower',

extent=(-5,15,-5,15),alpha=0.3, vmin=0, vmax=K-1)

plt.title('Space partitions', size=14)

plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')



fig = plt.gcf()

fig.set_size_inches((6,5))



plt.savefig("/kaggle/working/samples3.png",dpi=300, bbox_inches='tight')