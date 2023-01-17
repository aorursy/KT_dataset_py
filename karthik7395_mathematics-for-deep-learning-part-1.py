import numpy as np
import pandas as pd 
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# Getting to know the type of scalar
a = 5
b = 7.5
print(type(a))
print(type(b))
# External type casting
a=float(a)
print(type(a))
# Is Scalar Function
def isscalar(num):
    if isinstance(num, generic):
        return True
    else:
        return False

print(np.isscalar(3.1))
print(np.isscalar([3.1,3.2]))
print(np.isscalar(False))
import numpy as np

# Declaring Vectors

x = [1, 2, 3]
y = [4, 5, 6]

print(type(x))

# This does not give the vector addition.
print(x + y)

# Vector addition using Numpy

z = np.add(x, y)
print(z)

# Vector Cross Product
mul = np.cross(x, y)
print(mul)
x = np.matrix([[1,3],[4,5]])
x
a = x.mean(0)
a    
z = x.mean(1)
z
# Broadcasting
a - z
# Matrix Addition

x = np.matrix([[1, 2], [4, 3]])
sum = x.sum() #axis=1
print(sum)
x = np.matrix([[1, 2], [4, 3]])
y = np.matrix([[3, 4], [3, 10]])
m_sum = np.add(x, y)
print(m_sum)
print(m_sum.shape)
# Matrix-Scalar Addition

x = np.matrix([[1, 2], [4, 3]])
s_sum = x + 1
print(s_sum)
# Matrix-Scalar Multiplication

x = np.matrix([[1, 2], [4, 3]])
s_mul = x * 3
print(s_mul)
# Matrix multiplication
a = [[1, 0], [0, 1]]
b = [1, 2]
np.matmul(a, b)
# Matrix Transpose
a = np.array([[1, 2], [3, 4]])
a.transpose()
# Tensor Creation
import torch
a = torch.Tensor(5,3,2,1)
print(a.tolist())
print(type(a))
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'age': [42, 52, 36, 24, 73], 
        'preTestScore': [4, 24, 31, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70]}
df = pd.DataFrame(data, columns = ['name', 'age', 'preTestScore', 'postTestScore'])
df
plt.hist(df['preTestScore'],bins=3)
plt.show()
plt.hist(df['postTestScore'],bins=2)
plt.show()
df['preTestScore'].count()
df['preTestScore'].min()
df['preTestScore'].median()
df['preTestScore'].var()
df['preTestScore'].std()
#Summary Statistics
df['preTestScore'].describe()
