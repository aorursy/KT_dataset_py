import numpy as np
data = [1, 2, 3, 4]
arr = np.array(data)
arr
data2 = [[10,20,30], [40,50,60], [70,80,90]]
arr2 = np.array(data2)
arr2
arr2[0]
arr2[0][1]
range_data = np.arange(10,20)
range_data
np.arange(0,100,3)
# The reshape function is used to convert the array to a matrix.
range_data.reshape(2,5)
np.zeros(10)
np.ones(10)
np.zeros((2,2)) # To create a 2x2 matrix, we define a tuple of the matrix dimensions into the zeros function.
np.ones((3,5))
np.linspace(0,100,5) # The linspace function is used to divide a range of numbers into equal parts.
np.eye(3) # Creates a 3x3 unit matrix.
np.random.choice(data, 2) # It selects 2 samples from data series with replacement.
np.random.choice(data, 2, replace=False) # It selects 2 samples from data series without replacement.
np.random.randint(0,10) # Returns one random value in the 0-10 range.
np.random.randint(0,10, 3) # Returns 3 values as array in the 0-10 range.
np.random.rand(3) # Returns 3 values as array in the 0-1 range.
# Random Number Generating With Standart Normal Distribution (Mean=0, Standart deviation=1)

np.random.randn(5)
# Random Number Generating With Standart Normal Distribution 2 (Mean=0, Standart deviation=1)

np.random.standard_normal(5)
# Random Number Generating With Normal Distribution

np.random.normal(1,3,10) # It generates 10 numbers with an average of 1 and a standard deviation of 3.
# Random Number Generating With Binomial Distribution (n: number of results (1 or 0 etc.), p: success rate)

np.random.binomial(6, 0.17, 5)
# Random Number Generating With Negative Binomial Distribution

np.random.binomial(5, 0.2, 5)
# Random Number Generating With Uniform Distribution

np.random.uniform(1, 10, 15)
# Random Number Generating With Poisson Distribution

np.random.poisson(5, 10)
# Random Number Generating With Geometric Distribution

np.random.geometric(0.5, 10)
# Random Number Generating With Hypergeometric Distribution

np.random.hypergeometric(3, 7, 3, 10)
# Random Number Generating With Exponential Distribution

np.random.exponential(0.25, 5)
# Random Number Generating With Lognormal Distribution

np.random.lognormal(5, 0.45, 4)
# Random Number Generating With Weibull Distribution

np.random.weibull(0.10, 6)
# Random Number Generating With Beta Distribution (a: alpha, b: beta)

np.random.beta(2, 3, 5)
# Random Number Generating With Gamma Distribution

np.random.gamma(5, 5, 5)
# Random Number Generating With Chi-Square Distribution (df: degrees of freedom)

np.random.chisquare(3, 5)
my_array = np.random.randint(0, 100, 20)
my_array
my_array.size
my_array.shape
my_array.dtype
my_array.max()
my_array.min()
my_array.mean()
my_array.std()
my_array.var()
my_array.sum()
my_array.argmax() # Returns the index of the max value.
my_array.argmin() # Returns the index of the min value.
my_array.sort()
my_array
detArray = np.random.randint(0,100,25)
detArray = detArray.reshape(5,5)
detArray
np.linalg.det(detArray)
arr = np.arange(1,10)
# To copy an array independently:
arr2 = arr.copy()
arr[:3] = 10 # Replaces the first 3 values ​​of the arr variable with 10.
arr2 > 5
# To bring values greater than 5 in array:
arr2[arr2>5]
mat = np.arange(1,21).reshape(5,4)
# To see the first 2 columns of the mat matrix:
mat[:,:2]
mat[:3, :2] # first 2 columns of the first 3 rows
mat[:2] # all columns of the first 2 rows
arr1 = np.array([1,2,3,4,5])
arr2 = np.array([6,7,8,9,0])
arr1 + arr2
arr2 - arr1
arr1 * arr2
arr2 / arr1
arr1 * 3
arr2 + 10
np.sqrt(arr2)