import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
iris = pd.read_csv("/kaggle/input/irisdata/iris.csv")
print(iris)
print(iris.columns)
iris['species'].value_counts()
iris.plot(kind='scatter',x= 'sepal_length' , y = 'sepal_width')
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(iris , hue='species',height=7)\
.map(plt.scatter , 'sepal_length','sepal_width') \
.add_legend()
plt.show()
plt.close()
sns.set_style('whitegrid')
sns.pairplot(iris , hue='species', height = 7)
plt.show()
iris_setosa = iris.loc[iris['species']=='setosa']
iris_virginica = iris.loc[iris['species']=='virginica']
iris_versicolor = iris.loc[iris['species']=='versicolor']

plt.plot(iris_setosa['petal_length'])
plt.plot(iris_virginica['petal_length'])
plt.plot(iris_versicolor['petal_length'])
plt.show()
sns.FacetGrid(iris , hue= "species", height = 7)\
   .map(sns.distplot , 'petal_length')\
   .add_legend()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
iris = pd.read_csv('/kaggle/input/irisdata/iris.csv')
sns.FacetGrid(iris , hue= "species", height = 7)\
   .map(sns.distplot , 'petal_width')\
   .add_legend()
plt.show()
sns.FacetGrid(iris , hue= "species", height = 7)\
   .map(sns.distplot , 'sepal_width')\
   .add_legend()
plt.show()
sns.FacetGrid(iris , hue= "species", height = 7)\
.map(sns.distplot , 'sepal_width')\
   .add_legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
counts , bin_edges = np.histogram(iris_setosa['petal_length'],bins = 10,density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()
iris_setosa = iris.loc[iris['species']=='setosa']
iris_virginica = iris.loc[iris['species']=='virginica']
iris_versicolor = iris.loc[iris['species']=='versicolor']

plt.plot(iris_setosa['petal_width'])
plt.plot(iris_virginica['petal_width'])
plt.plot(iris_versicolor['petal_width'])
plt.grid()

plt.show()

print('means:')
print(np.mean(iris_setosa['petal_length']))
print(np.mean(np.append(iris_setosa['petal_length'],50)))
print(np.mean(iris_virginica['petal_length']))
print(np.mean(iris_versicolor['petal_length']))
print(np.std(iris_setosa['petal_length']))
print(np.std(iris_versicolor['petal_length']))
print(np.std(iris_virginica['petal_length']))
print('median:')
print(np.median(iris_setosa['petal_length']))
print(np.median(np.append(iris_setosa['petal_length'],50)))
print(np.median(iris_virginica['petal_length']))
print(np.median(iris_versicolor['petal_length']))
print(np.percentile(iris_setosa['petal_length'],np.arange(0,100,25)))
print(np.percentile(iris_virginica['petal_length'],np.arange(0,100,25)))
print(np.percentile(iris_versicolor['petal_length'],np.arange(0,100,25)))
from statsmodels import robust

print(robust.mad(iris_virginica['petal_length']))
print(robust.mad(iris_versicolor['petal_length']))
print(robust.mad(iris_setosa['petal_length']))
sns.boxplot(x='species' , y = 'petal_length' , data = iris)
plt.show()
sns.violinplot(x='species' , y = 'petal_length' , data = iris)
plt.show()
sns.jointplot(x='petal_length' , y = 'petal_width' , data = iris , kind = 'kde')
plt.show()
import numpy as np 
import pylab
import scipy.stats as stats

std_normal = np.random.normal(loc = 0, scale = 1 , size = 5000)
for i in range(0,101):
    print(i,np.percentile(std_normal,i))
m = np.random.normal(loc = 20 ,scale = 5, size = 100000)
stats.probplot(m,dist = 'norm',plot = pylab )
pylab.show()
m = np.random.uniform(low = -1 ,high = 5, size = 10000)
stats.probplot(m,dist = 'norm',plot = pylab )
pylab.show()
import random
print (random.random())
import numpy
from pandas import read_csv
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
x = numpy.array([180,13,133,123,353,34,134,134,445,344])
n_iterations = 100
n_size = int(len(x))

medians = list()
for i in range(n_iterations):
    s  = resample(x , n_samples=n_size)
    m = numpy.median(s)
    medians.append(m)
    
pyplot.hist(medians)
pyplot.show()

alpha = 0.95
p = ((1.0-alpha)/2.0)*100
lower = numpy.percentile(medians,p)
p = (alpha+(1.0-alpha)/2.0)*100
upper = numpy.percentile(medians,p)
print('%.if confidence interval %.if and %.if'%(alpha*100,lower,upper))
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
x = stats.norm.rvs(size= 1000 )
sns.set_style('darkgrid')
sns.kdeplot(np.array(x), bw= 0.5)
plt.show()
stats.kstest(x,'norm')
y = np.random.uniform(0,1,1000)
sns.kdeplot(np.array(y), bw  = 0.1 )
plt.show()
stats.kstest(y,'norm')