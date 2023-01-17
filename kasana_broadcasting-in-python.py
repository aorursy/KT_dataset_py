import numpy as np

A = np.array([[56.00,0.00,4.44, 68.0],
             [1.2,104, 52.0,8.0],
             [1.8,135.00,0.99,0.9]])

print(A)
cal = A.sum(axis=0)
##Axis 0 means to sum verticaly
##Axis 1 means to sum horizontly
print(cal)

#per = 100*A/cal.reshape(1,4)  ###reshape - Broadcasting 
per = 100*A/cal
print(per)