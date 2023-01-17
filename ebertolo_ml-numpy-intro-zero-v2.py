# Libraries
import numpy as np
import matplotlib.pyplot as plt
# Python list allows different data types
p_list = [False, "Text", 7.5, 42, np]

[type(item) for item in p_list]
# Numpy array allows only data type
np_list = np.array([100, 5, 6.7, 8, 5] , dtype=int)

print(type(np_list))
[type(item) for item in np_list]

print(np_list.dtype)
print(np_list.shape)
print(np_list.ndim)
print(len(np_list))


np1 = np.zeros((2,4))
np1
np2 = np.ones((4,6), dtype='int16')    
np2
np3 = np.eye(3)
np3
# Creating a Numpy Array
print('np.arange( 10) = ', np.arange(10))
print('np.arange( 3, 8) = ', np.arange(3,8))
print('np.arange( 0, 2, 0.5) = ', np.arange(0, 2, 0.5))
print('np.linspace( 0, 2, 5 ) = ', np.linspace(0, 10, 5))

# vector[start:stop:step]
a = np.arange(20)
print(a[1:17:2])
print()
# Negative indexes
list_a = np.arange(20)
print('Negative indexes counts backwords from last position')
print(a[0:-1])
print()
# nested lists result in multi-dimensional arrays, using List Comprehensions
np_vet = np.array([range(i, i + 4) for i in [10, 20, 30]])
np_vet
np_vet[:,:]
np.savez_compressed('arq' , v1=np_vet, v2=list_a)
load_vet = np.load('arq.npz')
print(load_vet['v1'], '\n')
print(load_vet['v2'], '\n')
# Only can reshape compatible number of elements
grid = np.arange(1, 22).reshape((3, 7 ))
print(grid)
x = np.array([1, 2, 3, 5 , 6])
y = np.array([3, 2, 1])
np.concatenate([x, y])

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])

# concatenate along the first axis
np.concatenate([grid, grid])

# concatenate along the second axis (zero-indexed)
np.concatenate([grid, grid], axis=1)

x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])

# vertically stack the arrays
np.vstack([x, grid])

# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])

x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)
tempC = np.random.normal(25, 2.5, 10000000)
tempC[0:50]
#len(tempC)
%%time
for temp in tempC:
    temp = (temp * 1.8) + 32 # Celsius to Fahrenheit
%%time
tempF = (tempC * 1.8) + 32
#tempF =  np.add(np.multiply(tempC,1.8), 32) 
tempF[:20]
%%time
#average tempeture
np.mean(tempC)
%%time
print('sum: ',tempC.sum())
print('max: ',tempC.max())
print('min: ',tempC.min())
print('mean:',tempC.mean())
print('median:',np.median(tempC))
print('std: ',tempC.std())
print('var:',np.var(tempC), '\n')
%%time
np.add.reduce(tempC)
x = [1, 2, 3, 4]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))
x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

#import math
#print('\n', math.log(2**4, 2))
M = np.random.random((3, 4))
print(M)
M.min(axis=0)
M.max(axis=1)
import pandas as pd
df = pd.DataFrame(grid)
df
np.array(df)
from IPython.display import display
%matplotlib inline

# Read mminist dataset - this is a very popular dataset with hand written digits from 0 to 9
train = np.loadtxt("../input/train.csv", delimiter=',', skiprows=1)
#  42000 numbers with 32 x 32 pixels,
#  first coluns has the value of the number
print(train.shape)
train[0:1,:]
#Now we separate the target class (y) from the features x vector
X_train = train[:, 1:] # gets all rows, all columns except the first one (index 0) 
X_train[0:5, :] # print first 5 rows
y_train = train[:, 0:1] # gets all rows, only first column (index 0) 
y_train[0:5, :] # print first 5 rows
# for a simple test let us print the first number of the train list
plt.imshow(np.reshape(X_train[0],(28,28)),cmap = 'gray') 
print('class:',y_train[0])
for n, digit in enumerate(X_train[:4]):
#for digit in X_train[:4]:
    filtered = np.reshape(digit,(28,28)) + np.random.normal(127, 20.0, (28,28))
    plt.imshow(filtered,cmap = 'gray')
    plt.xlabel(y_train[n])
    plt.show()