## Declare Numpy array from a list





# Alternative code using list comprehensions: https://www.pythonforbeginners.com/basics/list-comprehensions-in-python

# test_list = [i**2 for i in range(10)]

 



print(test_array)
# Demo: Simply analyze the code and run the code cell



# sum 

print("sum function: ", sum(test_array))



# len

print("\nlen function:", len(test_array))



# max  

print("\nmax function:", max(test_array))



# min

print("\nmin function:", min(test_array))



# sort

print("\nsorted function:", sorted(test_array, reverse=True))



# Join method on strings 

letter_array = np.array(["The", "Coolest", "Kids", "Out", "There"])

letter_string = " ".join(letter_array)

print("\nJoin method on strings: {}".format(letter_string))
## Test array and list creation to see speed differences

%time new_list = [i**2 for i in time_list]
%time new_array = (time_array**2).copy()
test_array.dtype
# Demo: Simply analyze the code and run the code cell



def data_type_example(np_array):

    print("The array:", np_array)

    print("Data type:", np_array.dtype)

    print("\n")

    return



# modify the following line: 

odd_array = np.array(["a", 1, 1.0, False]) 



# see results 

data_type_example(odd_array) 
## Create array





## Add and subtract scalars

print("adding scalar:",                            )

print('subtracting scalar:',                       )

print('dividing scalar:',                          ) # multiplication works too

print('raise to power of scalar:',                 )



## Add and operate on other vectors 

print('\nadding vector:',                          )

print('dividing vector:',                          )

print('multiplying vector:',                       )

print('raising to the power of a vector:\n',       )

# Demo: Simply analyze the code and run the code cell



# Lists behave differently!  

test = [1,2,3,4]

test2 = [5,6,7,8]



# Adding two lists concatenates them

print("\nAdding two lists", test + test2)



# Multiplying them by a scalar simply creates a longer list with the original list repeated

print("\nMultiplying lists", test*2)



# You cannot add, subtract, divide, or raise lists to a power though



#print(test+2)

#print(test-2)

#print(test/2)

#print(test**2)
## Create list and array

%time five_list = [i+5 for i in test_array]
%time five_array = (test_array + 5)
# Demo: Simply analyze the code and run the code cell



test_array = np.array([

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

print("\nWhole array\n", test_array)



# Slicing numpy 2D array



# Accessing elements

print("\nAccessing 2nd row, 4th column\n", test_array[1, 3])

print("\nAccessing 2nd to 3rd row, 4th through 5th column\n", test_array[1:3, 3:5])

print("\nAccesing the entire 2nd column\n", test_array[1, :])

    

# Vectorized operations on rows or columns

print("\nVectorized operations on two columns\n", test_array[0, :] + test_array[1, :])

print("\nVectorized operations on two rows\n", test_array[:, 0] + test_array[:, 1])

    

# Vectorized operations on entire arrays

array_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

array_2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

print("\nVectorized operations on entire arrays\n", array_1 + array_2)



# Functions with 2D arrays (axis=None, axis=0 vs. axis=1)

print("\nSum across whole array\n",test_array.sum())

print("\nSum across rows per column\n",test_array.sum(axis=0))

print("\nSum across columns per row\n",test_array.sum(axis=1))



# argmax with 2D arrays (if you want argmax of a column or row, subset by that row or column)

print("\nargmax of whole array\n",test_array.argmax())

print("\nargmax of just the first row\n", test_array[:, 0].argmax()) 

# if not, it'll just give you argmaxes in a given axis  

print("\nargmax of each column\n", test_array.argmax(axis=0))
# Setup pandas and data







students = [

    "Brent", "Eric", "Ré", "Mohit", "Steven", "Riyaz", "Charu", "Poonam", "Olena", 

    "Alivelu", "Min", "Laura", "Alexia", "Giovanni", "Jon", "Matt", "Manasi" 

]



height_values = [72, 68, 61, 65, 64, 71, 63, 64, 67, 66, 64, 60, 65, 68, 67, 69, 63] 



## Student Data from classroom





## Three methods to access a person's height at given position in Series 









# Recommend to use iloc or loc in the future due to speed and guarantee that the value you're operating on is

# the value from the actual Series and not a temporary copy of the Series.



# Reference: http://pandas.pydata.org/pandas-docs/version/0.20/indexing.html#returning-a-view-versus-a-copy





## Access multiple people's heights





## Subset based on conditions, use the loc function  



# Demo: Simply analyze the code and run the code cell



print("\nAdding two series together with matching indices:\n")



double_height = heights + heights



print(double_height)



print('\nDividing series by a scalar value:\n', double_height//2)



print("\nHalf class Series\n")



half_class = heights.iloc[0::2].copy() 



print(half_class)



print("\nAdding together two Series with some non-overlapping indices\n", half_class + heights)
## Add 'A' to each item in a Series and turn each item to a string


