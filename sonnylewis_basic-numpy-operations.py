import numpy as np

arr = np.arange(0,10)
arr + arr
arr * arr
arr - arr
# Warning on division by zero, but not an error!

# Just replaced with nan

arr/arr
# Also warning, but not an error instead infinity

1/arr
arr**3
#Taking Square Roots

np.sqrt(arr)
#Calcualting exponential (e^)

np.exp(arr)
np.max(arr) #same as arr.max()
np.sin(arr)
np.log(arr)