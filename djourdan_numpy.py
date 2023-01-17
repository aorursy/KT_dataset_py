import numpy as np
import pandas as pd
array = np.array([1, 4, 5, 8], float)
print(array)
print()
array = np.array([[1, 2, 3], [4, 5, 6]], float)  # a 2D array/Matrix
print(array)
print(type(array))
a = np.array([[1,2], [2, 4], [6, 8]])
b = np.array([[6, 3, 9], [3, 2, 3]])
print(a.shape)
print(b.shape)
ten_zeros = np.zeros(10)
print(ten_zeros)
ten_ones = np.ones(10)
print(ten_ones)
squared_ten_zeros = np.zeros((10,10))
print(squared_ten_zeros)
## Random nunbers
squared_five_randoms = np.random.random((5, 5))
print(squared_five_randoms)
## Gaussian distribution numbers
G_squared_five = np.random.randn(5,5)
print(G_squared_five)
print()
print(G_squared_five.mean())
print(G_squared_five.var())
print(np.array([0, 1, 2, 3]).dtype) #int64 means a 64 bit integer
print(np.array([1.0, 1.5, 2.0, 2.5]).dtype)
print(np.array([True, False, True]).dtype)
array = np.array([1, 4, 5, 8], float)
print(array)
print(array[1])
print(array[:2])
print(array[:])

array[1] = 5.0
print(array[1])
two_D_array = np.array([[1, 2, 3], [4, 5, 6]], float)
print(two_D_array)
print(two_D_array[1][1])
print(two_D_array[1, :]) ## a[0, :] would select the first row from all columns of a.
print(two_D_array[:, 2]) ## a[:, 0] would select the first column from all rows of a.
print(two_D_array[:, :]) ## prints the whole elements, but not creating a new array
two_D_array = np.array([[1, 2, 3], [4, 5, 6]], float)
copy = two_D_array[:, :]
print(copy[1][1])
copy[1][1] = 0 ## makes change to the copied array also
print(copy)
print(two_D_array)
a = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]])
print(a[1:3, 1:3])
a = [[2,4], [6, 3]]
b = np.array([[8, 1], [2, 9]])
print(a[0][0] == b[1, 0])
a = np.array([1, 2, 3, 4])
b = np.array([True, True, False, False])
    
print(a[b])
print(a[np.array([True, False, True, False])])
array_1 = np.array([1, 2, 3], float)
array_2 = np.array([5, 2, 6], float)
print(array_1 + array_2)
print(array_1 - array_2)
print(array_1 * array_2)
print(array_1 ** 2)
print(array_1 ** array_2)
array_1 = np.array([[1, 2], [3, 4]], float)
array_2 = np.array([[5, 6], [7, 8]], float)
print(array_1 + array_2)
print(array_1 - array_2)
print(array_1 * array_2)
print()
a = np.array([[1, 2],[5, 7], [4, 3]])
b = a[2]
print(b * a[0, 1])
a = np.array([1, 2, 3, 2, 1])
b = (a >= 2)
    
print(a[b])
print(a[a >= 2])
a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 2, 1])
    
print(b == 2)
print(a[b == 2])
# Time spent in the classroom in the first week for 20 students
time_spent = np.array([
       12.89697233,    0.        ,   64.55043217,    0.        ,
       24.2315615 ,   39.991625  ,    0.        ,    0.        ,
      147.20683783,    0.        ,    0.        ,    0.        ,
       45.18261617,  157.60454283,  133.2434615 ,   52.85000767,
        0.        ,   54.9204785 ,   26.78142417,    0.
])

# Days to cancel for 20 students
days_to_cancel = np.array([
      4,   5,  37,   3,  12,   4,  35,  38,   5,  37,   3,   3,  68,
     38,  98,   2, 249,   2, 127,  35
])

def mean_time_for_paid_students(time_spent, days_to_cancel):
    '''
    Fill in this function to calculate the mean time spent in the classroom
    for students who stayed enrolled at least (greater than or equal to) 7 days.
    Unlike in Lesson 1, you can assume that days_to_cancel will contain only
    integers (there are no students who have not canceled yet).
    '''
    return time_spent[days_to_cancel >= 7].mean()
a = np.array([1, 2, 3, 4])
b = a
a += np.array([1, 1, 1, 1])
print(b)
a = np.array([1, 2, 3, 4])
b = a
a = a + np.array([1, 1, 1, 1])
print(b)
a = np.array([1, 2, 3, 4])
slice = a[:3]
slice[0] = 100
print(a)
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])
    
print(a & b)
print(a | b)
print(~a)
    
print(a & True)
print(a & False)
    
print(a | True)
print(a | False)
array_1 = np.array([1, 2, 3], float)
array_2 = np.array([[6], [7], [8]], float)
array_3 = np.array([1, 2, 6], float)
# sum
print(array_1.sum())
# mean
print(np.mean(array_1))
print(np.mean(array_2))
print(np.mean(array_1) == array_1.mean())
# dot product
print(np.dot(array_1, array_2))
# median
print(np.median(array_1))
# return the position of the maximum values of an array using argmax()
print(array_1.argmax()) 
# standard deviation
print(np.std(array_1))
print(np.std(array_1) == array_1.std())
# correlation
print(np.corrcoef(array_1, array_3))
a = np.array([1, 2])
b = np.array([2, 1])
# dot product using for loop
def dot(a,b):
    dot = 0
    for e, f in zip(a, b):
        dot = dot + e*f
    return dot

print(dot(a,b))
a = np.array([1, 2])
b = np.array([2, 1])
# other ways in calculating the dot product
print(np.sum(a*b))
print(a.dot(b))
print(b.dot(a))
print(np.dot(a, b))
a_mag = np.sqrt(np.sum(a*a))
print(a_mag)
# finding the normal value of a vector using linalg
a_mag = np.linalg.norm(a)
b_mag = np.linalg.norm(b)
print(a_mag)

# angle between two vectors
cosangle = np.dot(a, b) / (a_mag * b_mag)
print(cosangle)
angle = np.arccos(cosangle)
print(angle)
# transposing a matrix
print(a)
print(a.T)

matrix_2D = np.matrix([[1, 2], [3, 4]])
print(matrix_2D)
print(matrix_2D.T)
print()

matrix_2D = np.array([[1, 2], [3, 4]])
print(matrix_2D.T)
A = np.array([[1, 2], [3, 4]])

## inverse matrix
A_inv = np.linalg.inv(A)
print(A_inv)

## identity matrix
I = A_inv.dot(A)
print(I)
## determinant
Det_A = np.linalg.det(A)
print(Det_A)

## diagonal position
print(np.diag(A))
## diagonal matrix
A_diag = np.diag([1, 2, 3])
print(A_diag)

# Summing the values in diagonal positions
print(np.trace(A_diag))
print(A_diag.sum())
a = np.array([1, 2])
b = np.array([3, 4])

# Outer product : C(i,j) = A(i)B(j)
print(np.outer(a,b))
# Inner product : C = sum over i{A(i)B(i)} == dot product
print(np.inner(a,b))
## Finding covariance
A = np.random.randn(100, 3)
cov = np.cov(A.T) # transpose the matrikx first
print(cov)
print(np.linalg.eigh(cov))
print(np.linalg.eig(cov))
A = np.array([[1, 2], [3, 4]])
B = np.array([3, 4])

# AX = B
X = np.linalg.inv(A).dot(B)
print(X)
X = np.linalg.solve(A, B) # more efficient and more accurate
print(X)

array_1 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
])
print(array_1.sum())
print(array_1.sum(axis=0)) # vertical axis
print(array_1.sum(axis=1)) # horizontal axis
print(array_1.mean(axis=1))
countries = np.array([
    'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina'
])

employment = np.array([
    55.70000076,  51.40000153,  50.5       ,  75.69999695, 58.40000153
])

for country in countries:
    print('Examining country {}'.format(country))
    
for i in range(len(countries)):
    country = countries[i]
    country_employment = employment[i]
    print('Country {} has employment {}'.format(country, country_employment))
    
def max_employment1(countries, employment):
    max_country = None
    max_employment = 0
    
    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]
        if country_employment > max_employment:
            max_country = country
            max_employment = country_employment
    return (max_country, max_employment)

def max_employment(countries, employment):
    #.argmax() returns the index of the first maximally-valued element
    i = employment.argmax() 
    return (countries[i], employment[i])

print(max_employment(countries, employment))
# Subway ridership for 5 stations on 10 different days
ridership = np.array([
    [   0,    0,    2,    5,    0],
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691],
    [1560, 3392, 3826, 4787, 2613],
    [1608, 4802, 3932, 4477, 2705],
    [1576, 3933, 3909, 4979, 2685],
    [  95,  229,  255,  496,  201],
    [   2,    0,    1,   27,    0],
    [1438, 3785, 3589, 4174, 2215],
    [1342, 4043, 4009, 4665, 3033]
])

def mean_riders_for_max_station(ridership):
    '''
    Fill in this function to find the station with the maximum riders on the
    first day, then return the mean riders per day for that station. Also
    return the mean ridership overall for comparsion.
    
    Hint: NumPy's argmax() function might be useful:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
    '''
    max_station = ridership[0, :].argmax()
    overall_mean = np.mean(ridership) # Replace this with your code
    mean_for_max = np.mean(ridership[:, max_station])# Replace this with your code
    
    return (overall_mean, mean_for_max)

def min_and_max_riders_per_day(ridership):
    '''
    First, for each subway station, calculate the mean ridership per day. 
    Then, out of all the subway stations, return the 
    maximum and minimum of these values. That is, find the maximum
    mean-ridership-per-day and the minimum mean-ridership-per-day for any
    subway station.
    '''
    station_riders = ridership.mean(axis=0)
    max_daily_ridership = station_riders.max()
    min_daily_ridership = station_riders.min()
    
    return (max_daily_ridership, min_daily_ridership)
