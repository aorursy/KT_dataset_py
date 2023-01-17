## Importing Libraries !!

## References Boris Pashvaker



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import os
# Lets read the NBA csv file 



nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  





nba.head()   # You would remember the Head method from the Series Note that was discussed on Day 2 



# We can Also see an NaN in the Salary Column !!
nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  



nba.head( 3 )  # Get 3 rows
## The Tail method



nba.tail( 3 )   # Oooh we have a row full of Missing values, which we will revisit later
### Index attribute

nba.index
### Values attribute  !!

# Stores all the information in numpy array format

nba.values 
### Dtype ( Datatype attribute)

## Gives the Datatype  of each column



nba.dtypes
print('column attribute returns all the columns in Dataframe :: \n', nba.columns )
print( 'Axes attribute ( has 2 axes Index and columns therefore dataframe is 2d ) ::\n')



print( nba.axes)
### Quick information on all columns / Features 



nba.info() 



# Memory usage is how much memory it takes on disk 

# We can see that there are few  missing in College and Salary columns
nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  

nba.head( 5 )
### We can use the dot < column > format to extract any column from a dataframe



print('Get the "Name" Column from Dataframe :: \n', nba.Name )

print('****'*20)

print('What type is a column ?? ', type(nba.Name))

print('Yes, its a Series object !!')
### College column

print('Similarly the College Column ::\n', nba.College )
### The above format of extracting columns fail when there are spaces in column names
### Another way to extract



### The format is  Dataframe_name [ Column you want to extract ]



print('****'*20)

print('Extracting Name column :: \n', nba['Name'] )

print('****'*20)

print('Extracting College column:: \n', nba['College'] )

print('****'*20)

print('Extracting Team column:: \n', nba['Team'] )

print('****'*20)
print( 'Lets check the Name , college and Salary columns :: \n')

nba [ ['Name','College', 'Salary'] ]
### Since we can get the columns from  dataframe.columns

### What if we do this ?



nba[ nba.columns ]  ## This gives the whole data !
 # [ : 3 ] => Grab the first 3 columns (Name,Team and Number)

nba[ nba.columns[ : 3 ] ] 
# eg :



# You can clearly see that you have a key error !!

nba[ 'Phone Number' ]  
nba [ 'Location'] = 'United States'

nba.head( 7 )

## We can see that on extreme right that our newly created Column is present
# Loc is the location where we want to insert the column

# Column is the name of column to be inserted

# Value is the values of the column which is to be inserted 





# Lets read our csv again

nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  



# We inserted Location between Name and  Team

# Name has the value 0, location ha the value 1, Team has the value 2 and so on ...

nba.insert( loc = 1, column = 'Location' , value = 'United States'   )

nba.head(4)
### Reading again

nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  

nba.head()  
### Adding 5 years to age column



nba['Age'].add( 5 )
# short form

# We can use the '+' to add values

# Similarly we can do the same for Subtraction , multiplication etc ..

nba['Age'] + 5
### Lets now do an apply type of operation as we did in Series for our Dataframe



### We begin by extracting the Age column

nba[ 'Age' ] 
print( 'If a Player\'s Age is less than 23 he is Young else he is old:: \n')





# We store this in a new column called Age Group !!

nba['Age_group'] = nba['Age'].apply( lambda x : 'Young' if x <24 else 'Old' )

nba.head( 4 )
# Converting Salary in Dollars to Salary in Million Dollar units



# Here we can clearly see that each Salary got divided by 1 million

nba['Salary_in_Dollor_Millions'] = nba['Salary'] / 1000000  

nba.head( 3 )
### Now lets see how many Old and Young players are there !!



# If you Remember, we can use value_counts() method as we had done earlier on Series.



print('We can see that there are 351 Old and 107 Young players !!! ')

nba['Age_group'].value_counts()
nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  

nba.head( 3 )
print( 'Here we can see that there is a Row Full of Missing values ( for all columns ) :: \n')

nba.tail( 5 )
print('The Drop Na Method allows us to drop the Null values !! ')

print()

print('Dropna when the  "how" parameteris enabled drops, the rows where ')

print('all columns have Null/ Missing Values !! ')

nba.dropna( how = 'all')



# We can see that we do not have the row with all Null values
# We can pass in a list of columns for dropna to check

# If and only if all these column values are Null/ NaN / Missing, the row is dropped



print('Lets say that we only want to look at players wtih valid College value ::\n')



print('You can clearly observe that we have reduced row count from 458 to 373 :: \n')

nba.dropna( subset = [ 'College'] )



nba = pd.read_csv(filepath_or_buffer = '../input/old-nba-data/nba.csv' )  

nba.head( 3 )
print( 'Lets take a look at the College column :: \n')

print('We can see that there are plenty of missing values !!! ')

nba[ 'College' ]
print('Okay we got the value counts :: \n')

nba['College'].value_counts()
cllg_counts = pd.DataFrame(nba['College'].value_counts() )

cllg_counts.head(3)
# For now dont worry how to plot things, just observe that a very large number of players are from Kentucky and distribution is skewed

# The below plot is called a horizontal Barplot !

cllg_counts.plot.barh(figsize = ( 20, 20)) 

plt.xlabel( 'Player count')                      # For naming X axis

plt.ylabel('College Name')                        # For Naming Y axis

plt.title( 'Player count from Colleges')          # Adding a Title to our plot

plt.show()


# We know from Previous Series notebook, that we can do method chaining i.e applying one method after another !!

# The isnull() method results in a boolean True where null values are present else False

# sum() method sums up all the Trues in the College column resulting in the value 85



print('There are 85 Missing values in the College column :: ', nba['College'].isnull().sum() )
# The fillna method has a value parameter

# The value parameter takes a value and replaces all the Misssing values 

# with the value provided



print('We can clearly observe that at \nROWS 4, 5 We have NaNs which got replaced  by Kentucky!!')

nba['College_modified']  = nba['College'].fillna( value = 'Kentucky')

nba.head( 10 )