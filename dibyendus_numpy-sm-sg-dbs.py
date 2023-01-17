import numpy as np
a = np.array([1,2,3])

a
b = np.array([[1,2,3],[4,5,6]])

print(b)



print('-------------------------------------------')



# Print out the shape of array

print('Shape:' + str(b.shape))



print('-------------------------------------------')



# Shape is a tuple , first is no of rows and secords element is no of columns



# Print out the data type of array

print(b.dtype)
my_list = [1,2,3]



#In order to multiply each element of a list with a scalar, we need to write a loop

emp_list = []

for element in my_list:

    emp_list.append(element*2)

print(emp_list)  



#We cannot do straight way elementwise multiplication in list 

my_list*2 #This is doing concatenation
a = np.array([1,2,3])

print(a*2)            #Much faster way to do
a = np.array([1,2,3])

b = np.array([4,5,6])

print(a+b)             #same as np.add(a,b)

print(a-b)             #same as np.substract(a,b)

print (np.multiply(a,b))  

print (np.divide(a,b)) 

print(np.sqrt(a))       #Elementwise square root
x = np.array([[1,2],[3,4]])

y = np.array([[5,6],[7,8]])

print(np.dot(x,y))  
x = np.array([[1,2],[3,4]])

print(np.transpose(x))
a = np.zeros((2,2))   #Create a 2*2 array of zeros

print(a)
b = np.ones((1,2))    #array of all ones

print(b)
c = np.eye(2)         #Creates an 2*2 Identity matrix

print(c)
d = np.random.random((2,2))  #Creates an array of random numbers

print(d)
np.arange(1,10)

np.arange(1, 10, 2)
# Create an 2-D array of 3 rows and 4 columns

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]) 

print(a)
print (a[ : , : ]) #Prints all rows and Columns
print(a[1 :  ,  : ]) # From 2nd row to all rows & all the columns
print(a[0,3]) # Print the 4th element of 1st row

print(a[2,0]) # Print the first element of 3rd row
print(a[ :  , 1 : 3]) # All rows along with 2nd & 3rd columns
# Suppose we want all the records of the array which are greater than 3

print(a>2)

print(a[a>2])
# Create an 2-D array of 3 rows and 4 columns

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]) 

print(a)



print('-------------------------------------------')



# Suppose we want to reshape the array with 2 rows and 6 columns

a = a.reshape(2,6)

print('reshaped array:')

a
a = np.array([[1,2,3],[4,5,6],[10,15,100]])

print(np.median(a)) # Median 

print(np.mean(a)) # Mean

print(np.var(a)) # Variance - average of squared deviations from mean 

print(np.std(a)) # Standard Deviation - square root of variance