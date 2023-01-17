# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("/kaggle/input/iris/Iris.csv")
del iris['Id']
iris.head()
import plotly_express as px
px.histogram(iris, x = 'SepalLengthCm',color = 'Species',nbins=20)
p_setosa = len(iris[iris['Species']=='Iris-setosa'])/len(iris)

p_versicolor = len(iris[iris['Species']=='Iris-versicolor'])/len(iris)

p_virginica = len(iris[iris['Species']=='Iris-virginica'])/len(iris)

print(p_setosa,p_versicolor,p_virginica)
import numpy as np                                                              

import seaborn as sns                                                           

from scipy import stats                                                         

import matplotlib.pyplot as plt                                                 



sns.set(style="ticks")



# calculate the pdf over a range of values

xx = np.arange(min(iris['SepalLengthCm']), max(iris['SepalLengthCm']),0.001)



x = iris[iris['Species']=='Iris-setosa']['SepalLengthCm']

sns.distplot(x, kde = False, norm_hist=True,color='skyblue',label = 'Setosa')

yy = stats.norm.pdf(xx,loc=np.mean(x),scale=np.std(x))

plt.plot(xx, yy, 'skyblue', lw=2) 



x = iris[iris['Species']=='Iris-versicolor']['SepalLengthCm']

sns.distplot(x, kde = False, norm_hist=True,color='green',label = 'Versicolor')

yy = stats.norm.pdf(xx,loc=np.mean(x),scale=np.std(x))

plt.plot(xx, yy, 'green', lw=2) 



x = iris[iris['Species']=='Iris-virginica']['SepalLengthCm']

g = sns.distplot(x, kde = False, norm_hist=True,color='red',label = 'Virginica')

yy = stats.norm.pdf(xx,loc=np.mean(x),scale=np.std(x))

plt.plot(xx, yy, 'red', lw=2) 

sns.despine()

g.figure.set_size_inches(20,10)

g.legend()


x = iris[iris['Species']=='Iris-setosa']['SepalLengthCm']

print(np.mean(x),np.std(x))



x = iris[iris['Species']=='Iris-versicolor']['SepalLengthCm']

print(np.mean(x),np.std(x))



x = iris[iris['Species']=='Iris-virginica']['SepalLengthCm']

print(np.mean(x),np.std(x))


x = iris[iris['Species']=='Iris-setosa']['SepalLengthCm']

print("Setosa",stats.norm.pdf(7,loc=np.mean(x),scale=np.std(x))*.33)



x = iris[iris['Species']=='Iris-versicolor']['SepalLengthCm']

print("Versicolor",stats.norm.pdf(7,loc=np.mean(x),scale=np.std(x))*.33)



x = iris[iris['Species']=='Iris-virginica']['SepalLengthCm']

print("Virginica",stats.norm.pdf(7,loc=np.mean(x),scale=np.std(x))*.33)
px.scatter(iris, 'SepalLengthCm', 'PetalLengthCm',color = 'Species')
import numpy as np                                                              

import seaborn as sns                                                           

from scipy import stats                                                         

import matplotlib.pyplot as plt                                                 



sns.set(style="ticks")



# calculate the pdf over a range of values

xx = np.arange(min(iris['SepalLengthCm']), max(iris['SepalLengthCm']),0.001)



x1 = iris[iris['Species']=='Iris-setosa']['SepalLengthCm']

x2 = iris[iris['Species']=='Iris-setosa']['PetalLengthCm']

sns.scatterplot(x1,x2, color='skyblue',label = 'Setosa')

#yy = stats.norm.pdf(xx,loc=np.mean(x),scale=np.std(x))

#plt.plot(xx, yy, 'skyblue', lw=2) 



x1 = iris[iris['Species']=='Iris-versicolor']['SepalLengthCm']

x2 = iris[iris['Species']=='Iris-versicolor']['PetalLengthCm']

sns.scatterplot(x1,x2,color='green',label = 'Versicolor')

#yy = stats.norm.pdf(xx,loc=np.mean(x),scale=np.std(x))

#plt.plot(xx, yy, 'green', lw=2) 



x1 = iris[iris['Species']=='Iris-virginica']['SepalLengthCm']

x2 = iris[iris['Species']=='Iris-virginica']['PetalLengthCm']



g = sns.scatterplot(x1, x2, color='red',label = 'Virginica')

#yy = stats.norm.pdf(xx,loc=np.mean(x),scale=np.std(x))

#plt.plot(xx, yy, 'red', lw=2) 

sns.despine()

g.figure.set_size_inches(20,10)

g.legend()
import numpy as np                                                              

import seaborn as sns                                                           

from scipy import stats                                                         

import matplotlib.pyplot as plt    

from matplotlib.mlab import bivariate_normal

sns.set(style="ticks")



# SETOSA

x1 = iris[iris['Species']=='Iris-setosa']['SepalLengthCm']

x2 = iris[iris['Species']=='Iris-setosa']['PetalLengthCm']

sns.scatterplot(x1,x2, color='skyblue',label = 'Setosa')



mu_x1=np.mean(x1)

mu_x2=np.mean(x2)

sigma_x1=np.std(x1)**2

sigma_x2=np.std(x2)**2

xx = np.arange(min(x1), max(x1),0.001)

yy = np.arange(min(x2), max(x2),0.001)



X, Y = np.meshgrid(xx, yy)

Z = bivariate_normal(X,Y, sigma_x1, sigma_x2, mu_x1, mu_x2)

plt.contour(X,Y,Z,colors='skyblue')



# VERSICOLOR

x1 = iris[iris['Species']=='Iris-versicolor']['SepalLengthCm']

x2 = iris[iris['Species']=='Iris-versicolor']['PetalLengthCm']

sns.scatterplot(x1,x2,color='green',label = 'Versicolor')



mu_x1=np.mean(x1)

mu_x2=np.mean(x2)

sigma_x1=np.std(x1)**2

sigma_x2=np.std(x2)**2

xx = np.arange(min(x1), max(x1),0.001)

yy = np.arange(min(x2), max(x2),0.001)



X, Y = np.meshgrid(xx, yy)

Z = bivariate_normal(X,Y, sigma_x1, sigma_x2, mu_x1, mu_x2)

plt.contour(X,Y,Z,colors='green')



# VIRGINICA

x1 = iris[iris['Species']=='Iris-virginica']['SepalLengthCm']

x2 = iris[iris['Species']=='Iris-virginica']['PetalLengthCm']

g = sns.scatterplot(x1, x2, color='red',label = 'Virginica')



mu_x1=np.mean(x1)

mu_x2=np.mean(x2)

sigma_x1=np.std(x1)**2

sigma_x2=np.std(x2)**2

xx = np.arange(min(x1), max(x1),0.001)

yy = np.arange(min(x2), max(x2),0.001)



X, Y = np.meshgrid(xx, yy)

Z = bivariate_normal(X,Y, sigma_x1, sigma_x2, mu_x1, mu_x2)

plt.contour(X,Y,Z,colors='red')





sns.despine()

g.figure.set_size_inches(20,10)

g.legend()
