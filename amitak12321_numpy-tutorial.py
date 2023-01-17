## initially Lets import numpy

import numpy as np
import pandas as pd
my_lst=[1,2,3,4,5]

arr=np.array(my_lst)
print(arr)
type(arr)
## Multinested array
my_lst1=[1,2,3,4,5]
my_lst2=[2,3,4,5,6]
my_lst3=[9,7,6,8,9]

arr=np.array([my_lst1,my_lst2,my_lst3])
arr
type(arr)
## check the shape of the array

arr.shape
## Accessing the array elements

arr
arr[2,2]
arr
arr[1:,:2]
arr[:,3:]
arr
### Some conditions very useful in Exploratory Data Analysis 

val=2

arr[arr<3]
## Create arrays and reshape

np.arange(0,10).reshape(5,2)
arr1=np.arange(0,10).reshape(2,5)
arr2=np.arange(0,10).reshape(2,5)
arr1*arr2
np.ones((2,5),dtype=int)
## random distribution
np.random.rand(3,3)
arr_ex=np.random.randn(4,4)
arr_ex
import seaborn as sns
sns.distplot(pd.DataFrame(arr_ex.reshape(16,1)))
np.random.randint(0,100,8).reshape(4,2)
np.random.random_sample((1,5))
