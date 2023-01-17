import numpy as np
np.arange(4)             
np.arange(4.0)           
np.arange(2, 5)           
np.arange(3, 9, 2)        
import numpy as np

myData1 = np.array([[3,4],[5,6]])
myData2 = np.array([[7,8],[0,1]])

print(myData1 + myData2)
print(np.add(myData1, myData2))

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
myData = myData.reshape((2, 4))
print (myData)

import numpy as np
myData = np.array([2, 3, 4, 5, 6])
print(myData.shape)
print (myData.shape[0])                 
myData = myData.reshape((myData.shape[0], 1))      
print(myData.shape)
print (myData)
