count = 0
while count < 8:
   print (count)
   count += 2
T1 = (10, 20, 30) 
T2= (7, )
print (T1)
print (T1[1])
print (T2[0])

list1 = [10, 30, 50]
print(list1[0])
print(list1[-2])    
list1.append(70)
print(list1)
print(len(list1))

import numpy as np

myData = np.array([10, 15, 20])                                     
print(myData.shape)                                                       
print(myData[0], myData[1], myData[2])                   
myData[0] = 5                                                                 
print(myData)                                                                 

myData = np.array([[10, 20, 30],[40, 50, 60]])             
print(myData.shape)                                                    
print(myData.shape[0])                                               
print(myData.shape[1])                                               
print(myData[0, 0], myData[0, 1], myData[1, 0])        

import numpy as np
myData = np.array([[0, 2, 4],[6, 8, 10],[12, 14, 16]])
 
X, y = myData[:, :-1], myData[:, -1]
print(X)
print(y)
