import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])

print("Common values between two arrays:")
print(np.intersect1d(a, b))

# From 'a' remove all of 'b'
a=np.setdiff1d(a,b)
print("a, b:",a,b)


