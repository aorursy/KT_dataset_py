#Load parks data from Biodiversity in National Parks dataset

import pandas as pd

parks_data = pd.read_csv('../input/park-biodiversity/parks.csv', index_col=['Park Code'])
parks_data.head()
#Indexing: Single Rows

#The simplest way to access a row is to pass the row number to the .iloc method. Note that first row is zero, just like list indexes.

parks_data.iloc[2] #Row with index no.2
#The other main approach is to pass a value from your dataframe's index to the .loc method:

parks_data.loc['BADL']  #Row with index name BADL
#Indexing: Multiple Rows

#If we need multiple rows, we can pass in multiple index values. Note that this changes the order of the results

parks_data.loc[['BADL', 'ARCH', 'ACAD']]  #display multiple rows with index names BADL ARCH ACAD
parks_data.iloc[[2, 1, 0]]
#List selected columns (0 to 30)

parks_data[:3]
parks_data[3:6]
#Indexing: Columns

#We can access a subset of the columns in a dataframe by placing the list of columns in brackets like so:

parks_data['State'].head(3)  #Display State column (data frame format)
#You can also access a single column as if it were an attribute of the dataframe, but only if the name has no spaces, uses only basic characters, and doesn't share a name with a dataframe method. 

parks_data.State.head(3) #Display State column (list dot format)
#Replace space in columns with underscore(_)

parks_data.columns = [col.replace(' ', '_').lower() for col in parks_data.columns]

print(parks_data.columns)
#Indexing: Columns and Rows

#If we need to subset by both columns and rows, you can stack the commands we've already learned.

parks_data[['state', 'acres']][:5]  #Display Columns state & acres and rows 0 to 4
#Indexing: Scalar Values

#As you may have noticed, everything we've tried so far returns a small dataframe or series. 

#If you need a single value, simply pass in a single column and index value.

parks_data.state.iloc[2]
#Note that you will get a different return type if you pass a single value in a list.

parks_data.state.iloc[[2]]
#Selecting a Subset of the Data

#The main method for subsetting data in Pandas is called boolean indexing. 

#First, let's take a look at what pandas does when we ask it to evaluate a boolean:

(parks_data.state == 'UT').head(3)
#We get a series of the results of the boolean. 

#Passing that series into a dataframe gives us the subset of the dataframe where the boolean evaluates to True.

parks_data[parks_data.state == 'UT']
#Some of the logical operators are different:

#~ replaces not

#| replaces or

#& replaces and

#If you have multiple arguments they'll need to be wrapped in parentheses. For example:

parks_data[(parks_data.latitude > 60) | (parks_data.acres > 10**6)].head(3)
#You can also use more complicated expressions, including lambda

parks_data[parks_data['park_name'].str.split().apply(lambda x: len(x) == 3)].head(3)
#Import Melbourne Housing Data

import pandas as pd

melbourne_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv') 
#Display first 5 rows of Melbourne Housing data

melbourne_data.head()
#You can display columns of the data

melbourne_data.columns
#Selecting a Single Column

#You can pull out any variable (or column) with dot-notation. This single column is stored in a Series, which is broadly like a DataFrame with only a single column of data. Here's an example:

melbourne_data.Price.head()
#Selecting Multiple Columns: 

#You can select multiple columns from a DataFrame by providing a list of column names inside brackets. Remember, each item in that list should be a string (with quotes).

melbourne_data[['Landsize', 'BuildingArea']].head()
#We can verify that we got the columns we need with the describe command.

columns_of_interest = ['Landsize', 'BuildingArea']

two_columns_of_data = melbourne_data[columns_of_interest]

two_columns_of_data.describe()
parks_data[parks_data.state.isin(['WA', 'OR', 'CA'])].head()
#Import wine reveiw data (use column 0 as the index)

import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)
reviews['province'].head() #province column in reviews table
reviews['province'].value_counts().head() #count of province column in reviews table
#Bar charts and categorical data (count of province item)

reviews['province'].value_counts().head(10).plot.bar()
#Bar charts and categorical data (count of province item) (Normalize: 0 to 1)

(reviews['province'].value_counts().head(10)/len(reviews)).plot.bar()
#Bar charts and categorical data (count of points item and sort by index)

reviews['points'].value_counts().sort_index().plot.bar()
#Line charts and categorical data (count of points item and sort by index)

reviews['points'].value_counts().sort_index().plot.line()
#Area charts and categorical data (count of points item and sort by index)

reviews['points'].value_counts().sort_index().plot.area()
#Display price column where price < 200 and plot histogram

reviews[reviews['price'] < 200]['price'].plot.hist()
#Display price column and plot histogram

reviews['price'].plot.hist()
#Display the rows with price greater than 1500

reviews[reviews['price'] > 1500]
#Display price column and plot histogram

reviews['points'].plot.hist()
pd.set_option('max_columns', None)

pokemon_data = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon_data.columns
#Bar plot of categories in column 'Type 1' (be mindful of space in the column)

pokemon_data['Type 1'].value_counts().plot.bar()
#The frequency of Pokemon by HP stat total:

pokemon_data['HP'].value_counts().sort_index().plot.line()
#The frequency of Pokemon by Speed

pokemon_data['Speed'].plot.hist()
reviews['province'].value_counts().head(10).plot.pie()

# Unsquish the pie.

import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
#Scatter plot of price of all the reviews where price is greater than 100

reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')
#Scatter plot of price for 100 reviews where price is greater than 100

reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')
#A hex plot aggregates points in space into hexagons, and then colors those hexagons based on the values within them

#The data in this plot is directly comparable with that in the scatter plot from earlier, but the story it tells us is very different. From this hexplot we can see that the bottles of wine reviewed by Wine Magazine cluster around 87.5 points and around $20.

#We did not see this effect by looking at the scatter plot, because too many similarly-priced, similarly-scoring wines were overplotted. By doing away with this problem, this hexplot presents us a much more useful view of the dataset.

#Hexplots and scatter plots can by applied to combinations of interval variables and/or ordinal categorical variables.

reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
#A hex plot aggregates points in space into hexagons, and then colors those hexagons based on the values within them

reviews[reviews['price'] < 100].sample(100).plot.hexbin(x='price', y='points', gridsize=15)