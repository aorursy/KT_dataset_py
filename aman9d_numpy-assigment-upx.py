import numpy as np
vector1 = np.arange(10,49,1)

vector1
vector1_rev = vector1[::-1]

vector1_rev
matrix1 = np.arange(0,9,1).reshape(3,3)

matrix1
a = [1,2,0,0,4,0]

np.nonzero(a)
np.random.rand(3,3,3)
b = np.random.rand(10,10)

print('min =', np.min(b))

print('max =', np.max(b))
c = np.zeros((5,5))

#c+= np.arange(5)

c+=[0,1,2,3,4]

c
A = np.random.rand(2,2)

B = np.random.rand(2,2)

#A==B

np.array_equal(A,B)
#using vector1

scalar1 = np.random.uniform(0,40)

print(scalar1)

index = (np.abs(vector1 - scalar1)).argmin()

print(vector1[index])
vector2 =  [1, 2, 3, 4, 5]

new_vector =np.zeros(len(vector2)+(len(vector2)-1)*3)

new_vector[::3+1]=vector2

print(new_vector)
np.bincount(b).argmax()