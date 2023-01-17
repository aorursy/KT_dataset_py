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
import numpy as np



#1.Create 1D, 2D, and boolean array using numpy.



#1D Array

a = np.array([1,2,3,4,5])



print (a)

print("\n")



#2D Array



b = np.array([[1,2,3],[4,5,6]])



print(b)

print("\n")



# Boolean Array



bool_ar = np.array([1,0,'a',0.5,0],dtype=bool)



print(bool_ar)

print("\n")





#2.Extract the odd numbers from a 2D array using numpy package.



arr = np.array([[1,2,3],[4,5,6]])



for i in arr:

    for j in i:

        if j%2 != 0:

            print(j, end=" ")

print("\n")

        

        

#3. How to replace items that satisfy a condition with another value in numpy array?



arr = np.array([1,4,16,3,22,9,11,15,54])



arr = np.where(arr>=15,0,arr)



print(arr)

print("\n")





#4. How to get the common items between two python numpy arrays?



a = np.array([1,2,3,4,5,6])



b = np.array([2,4,6])



c = np.intersect1d(a,b)



print(c)

print("\n")







#5.How to remove from one array those items that exist in another?



a = np.array([1,2,3,4,5,6])



b = np.array([2,4,6])



for i in b:

    for j in a:

        if i == j:

            a = a[a!=j]

print(a)

print("\n")



#6.How to get the positions where elements of two arrays match?



a = np.array([1,2,3,4,5,6])



b = np.array([2,4,6])



for i in b:

    for j in a:

        if i == j:

            index = np.argwhere(a==j)

            print(index)