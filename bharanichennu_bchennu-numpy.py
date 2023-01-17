#Numpy



import numpy as np # linear algebra

import pandas as pd 



#######################################



arr = np.arange(10)

odds = arr[arr%2 != 0]

odds



a = np.arange(10).reshape(2,-1)

b = np.repeat(1, 10).reshape(2,-1)

print("Horizontally stacked arrays")

np.hstack((a, b))



a = np.array([2, 6, 1, 9, 10, 3, 27])

a=a[(a > 5) & (a < 10)]

a

####################################



def maxy(x, y):

    if x >= y:

        return x

    else:

        return y



a = np.array([5, 7, 9, 8, 6, 4, 5])

b = np.array([6, 3, 4, 8, 9, 7, 1])



def pair_max(a, b):

    return np.array([maxy(x, y) for x, y in zip(a, b)])  



pair_max(a, b)

#################################