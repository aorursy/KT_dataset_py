import numpy as np
myData = np.array([[1,3,5,7], [2,4,6,8], [0,10,11,12]])  
print(myData.shape)                                  
myData2 = myData[:2, 1:3]   

print(myData[0, 1])                                    
myData2[0, 0] = 20                                    
print(myData[0, 1])                                 

row1 = myData[2, :]                                   
row2 = myData[1, :]                                
print(row1, row1.shape)                            
print(row2, row2.shape)                             

import numpy as np

myData = np.array([0, 2, 4, 6, 8])
print(myData[-1])                                
print(myData[-4])                                   
print(myData[-3:])                                

import numpy as np
myData = np.array([[0, 2, 4], [6, 8, 10], [12, 14, 16]])
X, y = myData[:, :-1], myData[:, -1]
print(X)
print(y)

import numpy as np
print (np.arange(4) )             
print(np.arange(4.0))           
print(np.arange(2, 5))           
print(np.arange(3, 11, 2))        
import numpy as np

myData1 = np.array([[3,4],[5,6]])
myData2 = np.array([[7,8],[0,1]])

print(myData1 + myData2)
print(np.add(myData1, myData2))

print(np.sqrt(myData1))

import numpy as np

myData1 = np.array([2,4])
myData2 = np.array([1,3])
myData3 = np.array([[1,3], [2, 5]])
myData4 = np.array([[0,1], [2, 3]])
print(np.dot(myData1, myData2))                                 
print(np.dot(myData1, myData3))                                
print(np.dot(myData3, myData4))

import numpy as np
myData = np.array([1, 3, 5, 7, 0, 2, 4, 6])
myData = myData.reshape(2, 4)
print (myData)

import numpy as np
myData = np.array([2, 3, 4, 5, 6])
print(myData.shape)
print (myData.shape[0])                
myData = myData.reshape((myData.shape[0], 1))      
print(myData.shape)
print (myData)
