!pip install jovian --upgrade -q
import jovian
jovian.commit(project='5-numpy functions for Statistics')
import numpy as np
# List of functions explained 
function1 = np.mean
function2 = np.median
function3 = np.var
function4 = np.std
function5 = np.amin
# Example 1 - Calculating arithmetic mean of a one dimensional array

# Initializing an array
arr1 = [1,2,3,4]

# Calculating mean
print(np.mean(arr1))

# Alternatively, we can also specify axis
print(np.mean(arr1 , axis = 0))
# Example 2 - Calculating arithmetic mean of a two dimensional array

# Initializing an array
arr2 = [[1,2,3] , [4,5,6]]

# Calculating mean
print(np.mean(arr2))

# we can also specify axis
print(np.mean(arr2 , axis = 0))
print(np.mean(arr2 , axis = 1))
# Example 3 - breaking : NaN's
arr3 = [1,2,np.nan,4]

print(np.mean(arr3))
print(np.nanmean(arr3))
jovian.commit(project='5-numpy functions for Statistics')
# Example 1 - Calculating median of a one dimensional array

# Initializing an array
arr4 = [1,2,3,4]

# Calculating median
print(np.median(arr4))

# Alternatively, we can also specify axis
print(np.median(arr4 , axis = 0))
# Example 2 - Calculating median of a two dimensional array

# Initializing an array
arr5 = [[1,2,3] , [4,5,6]]

# Calculating mean
print(np.median(arr5))

# we can also specify axis
print(np.median(arr5 , axis = 0))
print(np.median(arr5 , axis = 1))
# Example 3 - breaking : NaN's
arr6 = [1,2,np.nan,4]

print(np.median(arr6))
print(np.nanmedian(arr6))
jovian.commit(project='5-numpy functions for Statistics')
# Example 1 - Calculating variance of a one dimensional data set

# Initializing an array
arr7 = [1,20,15,5,7,8,9]

# Calculating Variance
print(np.var(arr7))

# Alternatively, we can also specify axis
print(np.var(arr7 , axis = 0))
# Example 2 - Calculating variance of a two dimensional data set

# Initializing an array
arr8 = [[1,20,15] ,[5,7,8]]

# Calculating Variance
print(np.var(arr8))
print(np.var(arr8 , axis = 0))
print(np.var(arr8 , axis = 1))
# Example 3 - breaking : NaN's
# Initializing an array
arr9 = [1,20,15,np.nan,7,8,9]

# Calculating Variance
print(np.var(arr9))
print(np.nanvar(arr9))
jovian.commit(project='5-numpy functions for Statistics')
# Example 1 - Calculating standard deviation of a one dimensional data set

# Initializing an array
arr10 = [1,20,15,5,7,8,9]

# Calculating standard deviation
print(np.std(arr10))

# Alternatively, we can also specify axis
print(np.std(arr10 , axis = 0))
# Example 2 - Calculating standard deviation of a two dimensional data set

# Initializing an array
arr11 = [[1,20,15] ,[5,7,8]]

# Calculating standard deviation
print(np.std(arr11))
print(np.std(arr11 , axis = 0))
print(np.std(arr11 , axis = 1))
# Example 3 - breaking : NaN's
# Initializing an array
arr12 = [1,20,15,np.nan,7,8,9]

# Calculating Standard deviation
print(np.std(arr12))
print(np.nanstd(arr12))
jovian.commit(project='5-numpy functions for Statistics')
# Example 1 - Determining minimum value of a one dimensional array
arr13 = [1,2,3,4,5,6]

#Determining minimum value
print(np.amin(arr13))

# Alternatively, we can also specify axis
print(np.amin(arr13 , axis = 0))
# Example 2 - Determining minimum value of a two dimensional array
arr14 = [[1,2,3],[4,5,6]]

#Determining minimum value
print(np.amin(arr14))
print(np.amin(arr14 , axis = 0))
print(np.amin(arr14 , axis = 1))
# Example 3 - breaking : NaN's
arr15 = [1,2,np.nan,4,5,6]

#Determining minimum value
print(np.amin(arr15))
print(np.nanmin(arr15))
jovian.commit(project='5-numpy functions for Statistics')
jovian.commit(project='5-numpy functions for Statistics')