import numpy as np

#create numpy array 
a = np.array([1,2,3])
print(a)
import numpy as np

#create a numpy 2d array
x = np.array([[2,4,5],[6,8,10]],np. int32)
print(type(x))
print(x.shape)
print(x.dtype)

import numpy as np

#creating a boolean array
bool_arr = np.array([1, 0.5, 0, None, 'a', '', True, False], dtype=bool)
print(bool_arr)
import numpy as np
#creating the array
a = np.array([[1,2,3], [4,5,6]], np.int32)
answer = (a[a%2!=0])
print (answer)

import numpy as np
a=np.random.randint(0,5, size=(5,4))
print("a\n")
print(a)
b=(a<3).astype(int)
print("\nb")
print(b)
import numpy as np
a = np.arrary([0, 1, 2, 4, 6])
print("Array1: ",a)
a2 = [1, 3, 4]
print("Array2: ",a2)
print("Common vaues between two arrays:")
print(np.intersect1d(a, a2))
import numpy as np
a = np.array([0, 1, 2, 4, 6])
print("Array to remove: ",a)
a2 = [1, 2, 3, 4, 5, 6, 7]
print("Array: ",a2)
for i in a2:
    if i in a2:
        a2.remove(1)
print("Array:" ,a2)        
        
import numpy as np

a = np.array([0, 1, 2, 3, 4, 5, 6])
b = np.array([6, 5, 4, 3, 2, 1, 6])
np.where(a==b)