import numpy as np

a = np.ones((5,3))

b = np.array([[2,4],[5,6],[1,3]])

np.dot(a,b)
import numpy as np



arr = np.arange(11)



arr[4:8] = np.multiply(arr[4:8],-1)



print(arr)
import numpy as np

x = np.ones((5,5))

print("Original array:")

print(x)

print("After keping 1 on the border and 0 inside in the array")

x[1:-1,1:-1] = 0

print(x)
import numpy as np

x = np.random.random(30)

print("Array:")

print (x)

print("Mean value:")

mean = x.mean()

print(mean)
import numpy as np

x = np.random.random((10,10))

print("Array:")

print(x) 

minimum = x.min()

maximum = x.max()

print("Minimum Values:", minimum)

print("Maximum Values:", maximum)
import numpy as np

x = np.random.random((3,3,3))

print(x)
import numpy as np

x=np.identity(3)

print('3x3 matrix:')

print(x)
import numpy as np 

x = np.array([1,2,0,0,4,0]) 

print ("Original Array :")

print(x)

y = np.nonzero(x) 

print ("Indices of non zero elements : ", y)
import numpy as np

x = np.arange(0, 11)

print("Original array:")

print(x)

print("Array after reversing:")

x = x[::-1]

print(x)
import numpy as np

x = np.zeros(10)

print(x)

print("After Updating 5th value to 1: ")

x[5] = 1

print(x)
import numpy as np

print(np.__version__)

print(np.show_config())
import numpy as np

x = np.linspace(0,1,12,endpoint=True)[1:-1]

print(x)
import numpy as np

x = np.random.randint(0,2,6)

print("1st array:")

print(x)

y = np.random.randint(0,2,6)

print("2nd array:")

print(y)

print("Are the 2 arrays equal? ")

equal = np.allclose(x, y)

print(equal)