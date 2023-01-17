import numpy as np   #alias
import pandas as pd  #alias
import os 
# A series is a one dimensional array ( you can think of this as a powerful version of python list)
# Generating a Series object !


py_data_stack  = ['Numpy', 'Pandas', 'Scipy', 'Matplotlib', 'StatsModels']
py_data_stack
# the dot is gives a list of options.

pd.Series( py_data_stack )

# We can see that we have a dtype as Object , Object is a synonym for 'String' type in Python
# In the left you can see a list of numbers starting from 0 to 4 , This is the Index.
# By default The index starts from 0

# Difference between Python lists and Pandas Series is ,Series Index need not be numbers.
comments = [ 5, 10, 15, 65, 78]

pd.Series ( data = comments )      # Data is comments and index is automatically generated from 0 till length of data
hotel_open = [ True, False, True, True]
pd.Series( data = hotel_open)   # Here we can See that the dtype is bool short for Boolean
machine_learning_algorithms = { 'KNN' : 'Distance based algorithm',
                                'Logistic Regression': 'A regression algorithm with a Logit Link(  classificaton )',
                                'Support Vector Machines' : 'Think twice before you run this on large Datasets with rbf kernel :)',
                                'Boosting Algorithms' : 'Learn from your past Mistakes :) '
                                }

print('The Dictionary is :: ', machine_learning_algorithms)
print()
print('This looks a bit different doesn\'t it ?' )
print( 'Here please note that the INDEX is not Numerical as discussed before :) ')
print( 'Here the INDEX are the KEYS from the DICTIONARY machine_learnin_algorithms :: \n')
print( 'The DATA of SERIES is the VALUES from DICTIONARY ::\n ')
print()
print( pd.Series( machine_learning_algorithms) )
### ATTRIBUTE deliver and give information about the Object, on the otherhand METHOD  can modify the object.

py_data_stack  = ['Numpy', 'Pandas', 'Scipy', 'Matplotlib', 'StatsModels']

s = pd.Series ( data = py_data_stack )
s
## Press TAB after the '.' to get attributes and methods

print('s.values returns values of Series as an Array ::',  s.values )

# The index starts at 0, stops at 5-1 = 4 and increments by 1
print('s.Index returns Range Index (Default) of Series as an Array ::',  s.index ) 

print('s.dtype Gives the Datatype of values Stored in Series :: ', s.dtype)
natural_numbers = [ 1, 2, 3, 4]

s = pd.Series( natural_numbers )
print( s )
print( 's.sum adds all the values in the Series ::', s.sum() )
print('s.product multiplies all the values in Series ::', s.product() )
print('s.mean gives the Mean of all values in Series ::', s.mean() )
# Read csv as the name suggests reads the CSV FILE
# IF you do not know what a CSV file is please check this out :: https://en.wikipedia.org/wiki/Comma-separated_values

print( 'We are Loading the Indian Foods csv file for learning more about Pandas :: \n')


pd.read_csv( filepath_or_buffer =   '../input/indian-food-101/indian_food.csv' )  # Filepath is the path where csv file is present

# What you see below is called a Dataframe which is another Pandas object which we will explore later.
food_name = pd.read_csv( filepath_or_buffer = '../input/indian-food-101/indian_food.csv',
                    usecols = ['name'])   # HERE We are only using the Name of the Food .

# Its still a DataFrame but we want a series object !!
food_name   
food_name = pd.read_csv( filepath_or_buffer = '../input/indian-food-101/indian_food.csv',
                    usecols = ['name'], squeeze = True )   # HERE We are only using the Name of the Food .
                                                           # Squeeze = True Squeezes the DataFrame into an Series object

print('Now we get a Series Object :: \n')
print( food_name  )
cook_time = pd.read_csv( filepath_or_buffer = '../input/indian-food-101/indian_food.csv',
                    usecols = ['cook_time'], squeeze = True )   # HERE We are only using the Name of the Food .
                                                           # Squeeze = True Squeezes the DataFrame into an Series object

print( cook_time  )
# Sometimes we just need to Take a look at the SAMPLE Data , we dont need  to Take a look at the Entire data,
# The Head and tail method  helps us do that.
# The head and Tail method gives us the First and last rows of a Series object
# It returns the Brand New series

print('****'*20)
print( 'Head method (default first 5 rows):: \n',food_name.head() )  # By default it gives first 5 Rows
print('****'*20)
print( 'tail method ( default last 5 rows) :: \n',food_name.tail() )   # By default it gives last 5 Rows
print('****'*20)
print( 'Head method :: \n',food_name.head( 3 ) )  #  Give first 3 Rows
print('****'*20)
print( 'tail method :: \n',food_name.tail( 3 ) )   #  Give last 3 Rows
print('****'*20)
print('Length of food_name :: ', len(food_name))
print('Length of cook_time :: ', len(cook_time))
print()
print('Sorted cook_time values :: ', sorted(cook_time) )  # Sorted from Least to Highest

print()  
print('Max gives the Maximum value in Series :: ', max(cook_time) ) # Similarly Min  !!
print()
print('Get the dictionary from Series Object :: \n', dict(food_name))
print('food.values gives the values of Series ::\n\n ', food_name.values )
print('name attribute gives the Name of the Series object :: ', food_name.name )  # HERE our value column name is Name 
print('is_unique returns True if every Value in the Series is UNIQUE :: ', food_name.is_unique ) # So all food are unique :)
print('shape attribute gives the Shape of Series :: ', food_name.shape ) # Here is 225 values so a tuple ( 225, )
# We can change the Name of the Series

food_name.name = 'Indian Foods'
print('name attribute gives the Name of the Series object :: ', food_name.name )  # We can see that its changed :) 
print('Our original food series looks like ::\n', food_name)
# Here we can see that he Values are Sorted in ascending Alphabetical order
food_name.sort_values(ascending = True)  # Ascending = True orders from smallest to greatest ( A - Z in case of Names)

# Here we can see that he Values are Sorted in decending Alphabetical order 
food_name.sort_values(ascending = False)  # Ascending = True orders from smallest to greatest ( A - Z in case of Names)
# Each time we call a method , we Get a brand new series, on which we can call other methods such as head and tail
food_name.sort_values(ascending = False).head( 10 )  # Calling head on sort_values method

# The values are not permanently changed , its temporary , To make it permanent we need to set inplace parameter = True
food_name.sort_values(ascending = False, inplace = True)
food_name # Now  its in decending order
### Sort Index method
print('****'*20)
print('After doing Sort Values, we can see on the left that our INDEX is Completely Shuffled up:: ')
print()
food_name.sort_values( ascending = True, inplace = True)
print( food_name  )
print('****'*20)
print('How can we get this back into its original Index i.e from 0 till length of Series -1 ??')
print('****'*20)
print()
print('For this , we use the Sort_index() Method !!!! ')
food_name.sort_index( ascending = True, inplace = True)
print(food_name)
print('****'*20)
# Checking if a value exists in the Series
'Dosa'  in food_name   # By default, the 'in' Keyword checks in the INDEX, not the VALUES.
## So lets check in values
'Dosa' in food_name.values  # It returns True !!
### Getting the value at index position 10 , which is Laddu ( For those of you who do not know what laddu is , its a Dessert )
print('***'*20)
print('To get the value at Index position 10, we do this ::\n', food_name [ 10 ] )  
print('***'*20)
print('Get all foods beginning from index position 10 till 15 ::\n', food_name [10 : 16 ])
print('***'*20)
print('Get foods at indices 10, 35 and 99 :: ', food_name[ [15, 35, 99] ] )
print('Original food Series ::\n\n', food_name )
print('***'*20)
print('Now lets change the Index ::\n')
food_name.index = [ 'Food_'+ str(i) for i in range( len( food_name ) ) ]
print('After Changing Index ::\n', food_name.index )
print('***'*20)
print('After Changes, the Series looks like :: ')
print( food_name )
print('***'*20)
## We can pull out rows based on Index Label !! 

print('***'*20)
print('Picking out Foods ::\n', food_name[ [ 'Food_0', 'Food_252']]  ) 
print('***'*20)
print('Slicing as in  Lists ::\n', food_name[ 'Food_0' : 'Food_10'])
print('***'*20)
## Here we use the cook_time Series
cook_time
print('***'*20)
print('count() gives the total number of NON NULL(NaN) records in Series :: ' , cook_time.count() )
print('mean() gives the mean of the Series values :: ', cook_time.mean() )
print('***'*20)
print('median() gives the median of the Series values :: ', cook_time.median() )
print('mode() gives the median of the Series values :: \n', cook_time.mode() )   # 30 is the Mode and Median
print('***'*20)
print('Standard Deviation of the Series :: ', cook_time.std() )
print('Index max returns the index of the largest value in Series :: ', cook_time.idxmax() )
print('Index min returns the index of the smallest value in Series :: ', cook_time.idxmin() )
print('***'*20)
## If we want to get a count of how many times each of them occurs we use value_counts ()

print( cook_time.value_counts() ) # We can see That the Value30 occurs 59 Times in the cook_time Series
### Applying a custom function on Series !!

# Suppose I want to categorize all the foods that take more than 25 units of TIME as 'TAKES MORE TIME TO PREPARE'
# And all thos less than 25 units as 'CAN BE PREPARED QUICKLY'

# To Do this we can use the Apply method
# The apply method() applies a function on the Series !!

def categorize_foods ( time_taken_to_cook ):
    if time_taken_to_cook > 25 :
        return 'TAKES MORE TIME TO PREPARE'
    else :
        return 'CAN BE PREPARED QUICKLY'
# now  Lets apply this on our Series object !!

hold_categorized_foods = cook_time.apply( categorize_foods )
hold_categorized_foods
### Lets check the Value counts !!

hold_categorized_foods.value_counts()  # We can see that there are roughly 100 Food Varieties which could be prepared quickly :)