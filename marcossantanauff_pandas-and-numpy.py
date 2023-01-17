%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
def add_two(x):

    '''This function adds 2 to every element on a list'''

    results = [] # Create empty list

    for i in x:

        a = i + 2

        results.append(a)

    return results
results = add_two(list(range(1,5)))
print(results)
import pandas as pd
df = pd.read_csv('../input/sample.csv',sep=',')
df.head(20)
list(df['Target Name'].unique())
list(df['Molecule ChEMBL ID'].unique())
df.groupby('Target Name')['Molecule ChEMBL ID'].count()
df.hist('Molecular Weight')

plt.show()
df.hist('pChEMBL Value')

plt.show()
sns.pairplot(df)
import numpy as np
myarray = np.asarray([1,2,3,4,5]) # 1D
myarray
myarray2 = np.asarray([[1,2,3,4,5]]) # 2D
myarray3 = np.asarray([[[1,2,3,4,5]]]) # 3D
myarray2
print(myarray.shape,myarray2.shape,myarray3.shape)
bigger_array = np.random.rand(4,3)
bigger_array
bigger_array[0,0]
bigger_array[0]
bigger_array[0,:]
bigger_array[0,:bigger_array.shape[1]]
bigger_array[1,1]
mylist = [1,2,3,4,5]

for i in mylist:

    print(i+2)
myarray
add_two_array = myarray + 2

print(add_two_array)
mult_two_array = myarray * 2

print(mult_two_array)
a = np.random.rand(3,5)

b = np.random.rand(5,10)
a
b
c = a@b
df.head()
df.shape
df.values