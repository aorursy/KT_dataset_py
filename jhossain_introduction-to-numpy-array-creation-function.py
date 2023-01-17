# import numpy 
import numpy as np 
# 1D array of length 3 with all values 0 
Z1 = np.zeros(3)
print(Z1)
# 2D array of 3x4 with all values 0 
Z2 = np.zeros((3,4))
print(Z2)
# 1D array of length 3 with all values 1
A1 = np.ones(3)  
print(A1) 
# 2D array of 3x4 with all values 1
A2 = np.ones((3,4))
A2
print(A2) 
# not specify start and step 
A1 = np.arange(10)
print(A1)
# specifying start and step 
A2 = np.arange(start=1, stop=10, step=2)
print(A2)
# another way 
A3 = np.arange(10, 25, 2)
print(A3)
# array of evenly spaced values 0 to 2, here sample size = 9
L1 = np.linspace(0,2,9)
print(L1)
# Array of 6 evenly divided values from 0 to 100
L2 = np.linspace(0, 100, 6)
print(L2) 
# Array of 1 to 5
L3 = np.linspace(start=1, stop=5, endpoint=True, retstep=False)
print(L3) 
# Array of 1 to 5
L4 = np.linspace(start=1, stop=5, endpoint=True, retstep=True)
print(L4) 
# generate an array with 5 samples with base 10.0 
np.logspace(1, 10, num=5, endpoint=True)
# generate an array with 5 samples with base 2.0
np.logspace(1, 10, num=5, endpoint=True, base=2.0)
# generate 2x2 constant array, constant = 7
C = np.full((2, 2), 7)
print(C)
# generate 2x2 identity matrix 
I = np.eye(2)
print(I) 
# create an array with randomly generated 5 values 
R = np.random.rand(5)
print(R)
# generate 2x2 array of random values 
R1 = np.random.random((2, 2))
print(R1)
# generate 4x5 array of random floats between 0-1
R2 = np.random.rand(4,5)
print(R2)
# generate 6x7 array of random floats between 0-100
R3 = np.random.rand(6,7)*100
print(R3)
# generate 2x3 array of random ints between 0-4
R4 = np.random.randint(5, size=(2,3))
print(R4)
# generate an empty array 
E1 = np.empty(2) 
print(E1)
# 2x2 empty array
E2 = np.empty((2, 2)) 
print(E2)
# generate an array of floats 
D = np.ones((2, 3, 4), dtype=np.float16)
D