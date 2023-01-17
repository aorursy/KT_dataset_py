import numpy as np
#Creating a matrix with all zeros
np.zeros((3,3))
#Creating 3 such matrices
np.zeros((3,3,3))
#First number indicates the number of such matrices
np.zeros((4,3,3))
#This produces four matrices of 3X3
#Matrix with all one's
np.ones((2,2))
#For integer values
np.ones((2,2), dtype= int)
#Create a matrix with custom value and custon length
np.full((2,2),"hellottwyyw", dtype='<U7')
np.full((3,2),"hellottwyyw", dtype='<U8')
#For random values
np.random.rand(10)
#Random Integers
data = np.random.randint(5,10,(10,10))
data[:10]
#Custom Rows and Columns (Zero Indexing)
data[1:3] #First Row to Third Row(Excluded)
#First Row to Third Row(Excluded)
#First Column to Third Column(Excluded)
data[1:3,1:3] 
#Combine data 
data1 = np.random.randint(5,15,(5,3))
data2 = np.random.randint(5,15,(5,3))
#Horizontal Combine
np.hstack([data1,data2])
#Vertical Combine
np.vstack([data1,data2])
#Linear Concatenation
np.concatenate([data1, data2], axis=1)
#Dimensional Concatenation
data1 = np.random.randint(5,15,size=(5,3,2))
data2 = np.random.randint(5,15,size=(5,3,2))
#Horizontal
np.concatenate([data1,data2],axis = 2)
#Vertical
np.concatenate([data1,data2],axis=1)
#Splittting the data

a1,a2 = np.hsplit(data,[5])
b1,b2,b3 = np.vsplit(data,[5,7])
b2
data.shape
data.reshape(4,25)
data.reshape(-1,20)
#Sum of the matrix
np.sum(data)
#sum of individual rows
np.sum(data,axis=1)
data
#Transpose of Matrix
data.T
data.mean()
#Mean of individual rows
data.mean(axis=1)
#Standard Deviation
data.std(axis=1)