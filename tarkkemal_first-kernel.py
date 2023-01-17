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
t = (1,2,"kemal")

t2 = (3,4,5)

t3 = t + t2



l = list(t3)

l[0] = "murat"

t3 = tuple(l)

t3
s = "ankara istanbul"

if len(s.split()) > 1:

    print("hello")
print("hello {}".format(3) + " hello")
a = lambda x: x + 3 / 2

int(a(2))

dir(int)
a = "kemal"

if not type(a) is int:

    print("not integer")

    

try:

    print(e)

except:

    raise TypeError("nope")

finally:

    print("done")





# START OF NUMPY



arr = np.array([[1,2,3,4], [1,2,3,4]])



arr.size

arr.ndim



zeros = np.zeros((3,3))

zeros[1][2] = 1

zeros



np.arange(3,10, 5)

np.linspace(3,10,5)
arr2 = arr

np.sin(arr)

arr3 = arr + arr2

arr2 ** 2

arr2 * arr



arr.sum()

arr.min()

arr.max()



arr.sum(axis = 0)



np.square(arr)



randarr = np.random.random(5)

randarr



array = np.array([[1,2,3,4], [5,6,7,8]])



array[:,1:3]



array.resize(4,2)

array[:,:] = 0

array.T

b = np.array([[1,2,6], [5,6,7]])

c = np.array([[3,4,5], [2,4,7]])



np.hstack((b,c)) 

np.vstack((b,c))
liste = [[1,2,3], [4,5,6]]



array = np.array(liste)

array



# END OF NUMPY