# I import the numpy package (mathematics), matplotlib (visualization), pandas (data)

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# importing data from the MNIST dataset, contaning digits written by hand. 

df=pd.read_csv('../input/mnist.csv',header=None)



# showing the first rows of the dataset

df.head()
print(df.shape)



# the dataset containts 2001 entries, characterized by 784 (28 x 28) values 

# that characterize the images plus 1 value that corresponds to the number represented 
# Let us reshape a single row in a matrix 28 x 28 in order to visualize it
# coverting data frame in array



# this is the true number, corresponding to the first column

ycheck=np.array(df.iloc[:,0])



# this is the representation of the number under the form of a vector

nn=df.shape[1]

X=np.array(df.iloc[:,1:nn])



# the vector of dimension 784 is transformed in a matrix of dimension 28x28

newmatrix=np.random.rand(28,28)   #creating a random 28x28 matrix



def numimg(t):

    for i in range(28):

        for j in range(28):

            newmatrix[i,j]=X[t,j+28*i]

    return newmatrix
dispnum=100    #This is the entry that will be displayed (there are 2001 entries)



print('The true number is '+str(ycheck[dispnum]))



plt.imshow(numimg(dispnum),cmap='gray')

plt.colorbar()

plt.show()