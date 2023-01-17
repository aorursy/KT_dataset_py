# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('../input/Iris.csv')

#print(iris)



print(iris.head(2))          # take a look at it

print(iris.isnull().sum())   # nulls?
# Let's just take a look at what we're dealing with here by plotting it all up



import matplotlib.pyplot as plt



fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(2,3,1)

ax2 = fig.add_subplot(2,3,2)

ax3 = fig.add_subplot(2,3,3)

ax4 = fig.add_subplot(2,3,4)

ax5 = fig.add_subplot(2,3,5)

ax6 = fig.add_subplot(2,3,6)



ax1.plot(iris['SepalLengthCm'], iris['SepalWidthCm'], 'bo')

ax1.set_xlabel('Sepal Length (cm)')

ax1.set_ylabel('Sepal Width (cm)')

ax2.plot(iris['SepalLengthCm'], iris['PetalLengthCm'], 'bo')

ax2.set_xlabel('Sepal Length (cm)')

ax2.set_ylabel('Petal Length (cm)')

ax3.plot(iris['SepalLengthCm'], iris['PetalWidthCm'], 'bo')

ax3.set_xlabel('Sepal Length (cm)')

ax3.set_ylabel('Petal Width (cm)')

ax4.plot(iris['PetalLengthCm'], iris['SepalWidthCm'], 'bo')

ax4.set_xlabel('Petal Length (cm)')

ax4.set_ylabel('Sepal Width (cm)')

ax5.plot(iris['PetalLengthCm'], iris['PetalWidthCm'], 'bo')

ax5.set_xlabel('Petal Length (cm)')

ax5.set_ylabel('Petal Width (cm)')

ax6.plot(iris['SepalWidthCm'], iris['PetalWidthCm'], 'bo')

ax6.set_xlabel('Sepal Width (cm)')

ax6.set_ylabel('Petal Width (cm)')

plt.show()
# More poking around, not plotting in the most efficient way but whatever



from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(221, projection='3d')

ax2 = fig.add_subplot(222, projection='3d')

ax3 = fig.add_subplot(223, projection='3d')

ax4 = fig.add_subplot(224, projection='3d')

ax1.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'], iris['PetalWidthCm'])

ax1.set_xlabel('Sepal Length (cm)')

ax1.set_ylabel('Sepal Width (cm)')

ax1.set_zlabel('Petal Width (cm)')

ax2.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'], iris['PetalLengthCm'])

ax2.set_xlabel('Sepal Length (cm)')

ax2.set_ylabel('Sepal Width (cm)')

ax2.set_zlabel('Petal Length (cm)')

ax3.scatter(iris['SepalLengthCm'], iris['PetalWidthCm'], iris['PetalLengthCm'])

ax3.set_xlabel('Sepal Length (cm)')

ax3.set_ylabel('Petal Width (cm)')

ax3.set_zlabel('Petal Width (cm)')

ax4.scatter(iris['SepalWidthCm'], iris['PetalWidthCm'], iris['PetalLengthCm'])

ax4.set_xlabel('Sepal Width (cm)')

ax4.set_ylabel('Sepal Width (cm)')

ax4.set_zlabel('Petal Width (cm)')

plt.show()
# Okay, looks like we need some kind of clustering algorithm. I'll try a 

# k-neighbors algorithm from scikitlearn I think. N.B. sckikit learn has

# two possible classifiers for labeled data - KNeighborsClassifier and 

# RadiusNeighborsClassifier. I'll use K neighbors because these data are 

# pretty regularly sampled. From the plots I can see the petal length and 

# width seem to be drivers of the separation, but I'll go ahead and run the 

# classifier on all pairs for kicks and giggles.



from sklearn.neighbors import KNeighborsClassifier



y = iris['Species'].astype('category')

y.cat.categories = [0,1,2]



knn = KNeighborsClassifier()  # use default parameters, nothing fancy needed



knn.fit(iris[['PetalWidthCm', 'PetalLengthCm']], y)

x_min, x_max = iris['PetalWidthCm'].min() - 1, iris['PetalWidthCm'].max() + 1

y_min, y_max = iris['PetalLengthCm'].min() - 1, iris['PetalLengthCm'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='cool')

plt.scatter(iris['PetalWidthCm'], iris['PetalLengthCm'], c = 'k')

plt.xlabel('Petal Width (cm)')

plt.ylabel('Petal Length (cm)')

plt.show()

knn.fit(iris[['PetalWidthCm', 'SepalWidthCm']], y)

x_min, x_max = iris['PetalWidthCm'].min() - 1, iris['PetalWidthCm'].max() + 1

y_min, y_max = iris['SepalWidthCm'].min() - 1, iris['SepalWidthCm'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='cool')

plt.scatter(iris['PetalWidthCm'], iris['SepalWidthCm'], c = 'k')

plt.xlabel('Petal Width (cm)')

plt.ylabel('Sepal Width (cm)')

plt.show()

knn.fit(iris[['PetalWidthCm', 'SepalLengthCm']], y)

x_min, x_max = iris['PetalWidthCm'].min() - 1, iris['PetalWidthCm'].max() + 1

y_min, y_max = iris['SepalLengthCm'].min() - 1, iris['SepalLengthCm'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='cool')

plt.scatter(iris['PetalWidthCm'], iris['SepalLengthCm'], c = 'k')

plt.xlabel('Petal Width (cm)')

plt.ylabel('Sepal Length (cm)')

plt.show()

knn.fit(iris[['SepalWidthCm', 'SepalLengthCm']], y)

x_min, x_max = iris['SepalWidthCm'].min() - 1, iris['SepalWidthCm'].max() + 1

y_min, y_max = iris['SepalLengthCm'].min() - 1, iris['SepalLengthCm'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='cool')

plt.scatter(iris['SepalWidthCm'], iris['SepalLengthCm'], c = 'k')

plt.xlabel('Sepal Width (cm)')

plt.ylabel('Sepal Length (cm)')

plt.show()

knn.fit(iris[['PetalLengthCm', 'SepalLengthCm']], y)

x_min, x_max = iris['PetalLengthCm'].min() - 1, iris['PetalLengthCm'].max() + 1

y_min, y_max = iris['SepalLengthCm'].min() - 1, iris['SepalLengthCm'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='cool')

plt.scatter(iris['PetalLengthCm'], iris['SepalLengthCm'], c = 'k')

plt.xlabel('Petal Length (cm)')

plt.ylabel('Sepal Length (cm)')

plt.show()

knn.fit(iris[['PetalLengthCm', 'SepalWidthCm']], y)

x_min, x_max = iris['PetalLengthCm'].min() - 1, iris['PetalLengthCm'].max() + 1

y_min, y_max = iris['SepalWidthCm'].min() - 1, iris['SepalWidthCm'].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap='cool')

plt.scatter(iris['PetalLengthCm'], iris['SepalWidthCm'], c = 'k')

plt.xlabel('Petal Length (cm)')

plt.ylabel('Sepal Width (cm)')

plt.show()

# Pretty good job.  Obviously, the messiest one is the combination of sepal

# length and width, which makes sense given our hypothesis from the 3D scatter 

# plots that petal length and widths were the drivers of separation.