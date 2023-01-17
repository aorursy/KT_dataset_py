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
import numpy as np

import numpy.random as nprnd

import matplotlib.pyplot as plt

import pandas as pd
a = [ [1,1],[1,2],[3,3]]

b = [ [4,1],[5,2],[4,1],[6,6]]



aa = np.array(a)

bb = np.array(b)

xyz=np.array(np.random.random((100,3)))

plt.scatter(xyz[:,0], xyz[:,1])

plt.show()
x2 = np.random.randint(10, size=(3, 2))  # Two-dimensional array

x22 = np.random.randint(10, size=(4, 2))  # Two-dimensional array
x22
plt.scatter(x22[:,0], x22[:,1])

plt.show()
arrayA = np.array([[1,1],[1,2],[3,3]])
arrayA
plt.scatter(arrayA[:,0], arrayA[:,1])

plt.show()
x = [1,1,3]

y = [1,2,3]



plt.plot(x, y,'-ok')
meanA


np.mean(arrayA)
np.mean(a, axis=0)
np.mean(a, axis=1)
np.mean(b, axis=0)
plt.scatter(bb[:,0], bb[:,1])

plt.show()
# a line crossing mean points 

plt.plot(np.mean(a, axis=0), np.mean(b, axis=0),'-ok')
rng = np.random.RandomState(0)

x = rng.randn(100)

y = rng.randn(100)

colors = rng.rand(100)

sizes = 1000 * rng.rand(100)



plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,

            cmap='viridis')

plt.colorbar();  # show color scale
plt.scatter(x, y, marker='o');
pd.DataFrame({'X': [1, 1, 3], 'Y': [1, 2, 3]}) # A = { {1,1},{1,2},{3,3}}
pd.DataFrame({'X': [1, 1, 3], 

              'Y': [1, 2, 3]},

             index=['Point A', 'Point B', 'Point C'])
pd.Series([[4,1],[5,2],[4,1],[6,6]]) # 2D Series is a sequence of data values
my_dataset = pd.Series([[4,1],[5,2],[4,1],[6,6]], index=['Point A', 'Point B', 'Point C', 'Point D'], name='Coordinates')
my_dataset
#We can use the shape attribute to check how large the resulting DataFrame is:

my_dataset.shape