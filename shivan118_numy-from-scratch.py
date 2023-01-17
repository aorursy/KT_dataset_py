import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#sns.set() #To set style



import warnings

warnings.filterwarnings('ignore')
x = np.float32(1.0)

print(x)
x = np.float64(946464646464646464646464646464466.333131313113)

print(x)
x = np.int_([1,2,4])

print(x)
x = np.arange(3, dtype=np.uint8)

print(x)
x = np.arange(2, 10, 2)

x
x = np.array([1, 2, 3], dtype='f')

print(x)
x = np.int8(34)

print(x)
import numpy as np
np.zeros((7, 8))
np.ones((5, 6))
np.arange(10)
np.arange(2, 10)
np.arange(3.5, 9.8, 0.2)
np.arange(2, 10, dtype=float)
np.arange(2, 10, 0.2)
np.linspace(5, 10, 6)
np.linspace(1, 10, 20)
np.indices((2, 3, 1))
x = np.array([1, 2, 3])

print(x)
x = np.array([[0, 0],[1, 1]])

print(x)

x = np.arange(10)

x
print(x[5])
# let's create an array

x = np.arange(10)



# let's make the array 2 dimensional

x.shape = (2, 5)

x

# let's try to access array elements

print(x[1, 2]) #7

print(x[0, 2])  #2

print(x[1, 3])  #8
# let's try to get the rows of the array



print(x[0])

print(x[1])
# let's try anther way to access the elements



print(x[0][2])
# let's try other indexing options



x = np.arange(20)

print(x[5:11]) #5 6 7 8 9 10
x = [1, 2, 3, 4, 5]

x= x[::-1]

x

x = np.array([1, 2, 3, 4, 5])

print(x[:-3])

print(x[:3])

print(x[-3:])

print(x[3:])
x = np.array([2, -4, 5, -6, 7, 0])

x = x[x > 0]

x
# indexing through values



print(x[x > 2])

print(x[x == 3])

print(x[x >= 2])
x = [1, 2, 3, 4, 5]

z = (a**2 for a in x)



q = []

for i in z:

    q.append(i)

print(q)
x = [1, 2, 3, 4, 5]

z = [a/3 for a in x]

z

x = np.array([('Sita', 39, 51.8), ('Gita', 23, 47.5)],

            dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
# let's print x(structured array)



print(x)
# let's do some indexing



print(x[1])

print(x[0])

# let's try to access according to the fields



print(x['age'])

print(x['name'])

print(x['weight'])
# let's try to modify the values using the fields



x['age'] = [19, 74]

print(x['age'])
# lets check the documentation of np.ones()



help(np.ones)

np.zeros(4)
np.zeros((3, 4))
np.zeros((2, 3), dtype = 'int')
np.random.random(3)
np.random.random(11)
np.arange(2, 50)
np.arange(2, 200, 10)
np.linspace(1, 10)
np.linspace(2, 7, 10)
import numpy as np

np.full((2,5), 6)

np.full(5, 9)
np.tile([0, 1], 5)
np.tile([0, 1], [3, 4])
np.eye(2)
np.eye(4)
np.random.randint(1, 100)
np.random.randint(30)
np.random.randint(-40, 40)
a = np.array([2, 3, 45, 6])

print(a)
a.shape

a.dtype

a.ndim

a.itemsize
import numpy as np



x= np.zeros((2, 4))

print(x)
# lets take an array a and take 12 elements inside it



a = np.arange(0, 12)

print(a)

# lets try to resize this array



print('Array with 3 Rows and 4 Columns')

b = a.reshape(3, 4)

print(b,'\n')



print('Array with 4 Rows and 3 Columns')

c = a.reshape(4, 3)

print(c, '\n')



print('Array with 6 Rows and 2 Columns')

d = a.reshape(6, 2)

print(d, '\n')



print('Array with 2 Rows and 6 Columns')

e = a.reshape(2, 6)

print(e, '\n')

# If you specify -1 as a dimension, the dimensions are automatically calculated



x = a.reshape(4, -1)

y = a.reshape(3, -1)

z = a.reshape(6, -1)

q = a.reshape(2, -1)



print(x, '\n')

print(y, '\n')

print(z, '\n')

print(a, '\n')
import numpy as np

x = np.full((2, 6), 4)

x.T
# lets make an array and transpose the array



# lets take an array

array = np.array([[3, 4, 4],[7, 7,8]])



print('Before Transposition of Array b')

print(array)



# lets transpose the array

transpose = array.T



print('After Transposition of Array a')

print(transpose)
# lets create two arrays for performing the operation of vertical stacking



# for vertical stacking the number of columns must be same

arr1 = np.arange(28).reshape(7, 4)

arr2 = np.arange(20).reshape(5, 4)



print('First Array')

print(arr1, '\n')

print('Second Array')

print(arr2)

# Note that np.vstack(a, b) throws an error - you need to pass the arrays as a list



np.vstack((arr1, arr2))

# lets create two arrays for performing the operation of vertical stacking



# for horizontal stakcing we must have same numbers of rows

arr1 = np.arange(21).reshape(3, 7)

arr2 = np.arange(6).reshape(3, 2)



print('First Array')

print(arr1, '\n')

print('Second Array')

print(arr2)

# Note that np.hstack(a, b) throws an error - you need to pass the arrays as a list



np.hstack((arr1, arr2))