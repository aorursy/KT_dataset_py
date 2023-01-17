import numpy as np
mydata=np.array([10,15,20])
print(mydata[0],mydata[1],mydata[2])

mydata[0]=5
print(mydata)
mydata=np.array([[10,20,30],[40,50,60]])
print(mydata.shape)
print(mydata.shape[0])
print(mydata.shape[1])
print(mydata[0,0],mydata[0,1],mydata[1,0])