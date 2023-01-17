import numpy as np
x = np.array([1,2,3,4,5])

y = np.array([1,3,20,30,40,50])
print(np.intersect1d(x, y))
x = np.setdiff1d(x,y)
# Final Output

print(x)
print(y)