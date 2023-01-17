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
#random matrix with in 0 to 1

random_matrix = np.random.random((2,2))

print(random_matrix)





#2D array with random integers

random_mat = np.random.randint(1,145,size=(20,2))

print(random_mat)



#copy

mat_A = random_mat.copy()   

#print(mat_A)



#min, max, mean, median, standar deviation

print("min:",np.min(random_mat))

print("max:",np.max(random_mat))

print("mean:",np.mean(random_mat))

print("median:",np.median(random_mat))

print("std deviation:",np.std(random_mat))





#linspace, creating a matrix where the 100 nums equally distributed in between 2,120



A = np.linspace(2,10,20)

print(A)

print("=================")

print(A.shape)

print("=================")

reshape_A = np.reshape(A,(20,1))

print(reshape_A[2,0])

print(A[2])

print("=================")





#index slicing

sliced_A = A[0:3]        

print(sliced_A)           # last one will not included

print("=================") 

sliced_A_Lats = A[:-2]               

print(sliced_A_Lats)

print("=================")# except last 2 num, remaining will print

sliced_A_4 = A[-4:]               

print(sliced_A_4)      # prints last 4 

print("=================")
#array



Identity_ar = np.array([[1,0,0],[0,1,0],[0,0,1]])

print(Identity_ar)
#Transpose and inverse



m = np.random.randint(1,9,size=(3,3))

print(m)

print("=================")

mt = np.transpose(m)

print(mt)

print("=================")

mi = np.linalg.inv(m)

mul = mi.dot(m)

print(mul)

#algebra



deg = np.random.randint(1,9,size=(3,3))



sin = np.sin(deg)

cos = np.cos(deg)

tan = np.tan(deg)

print(sin)

print(cos)

print(tan)