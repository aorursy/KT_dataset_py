import numpy as np

a=[[1,2,3],[4,5,6],[7,8,9]]

print (a)

print (type(a))



# """List to array conversion"""

b=np.array(a)

print (b)

print (type(b))

b.shape

print (a[1][1])

print (a[1])

print (a[1][:])

print (b[1][1])

print (b[1])

print (b[1][:])
print (b[0:3,1:])

