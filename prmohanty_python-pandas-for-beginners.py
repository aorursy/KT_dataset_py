import pandas as pd
# read a CSV file into Python and return a DataFrame as output

Movie_DF = pd.read_csv('../input/movie.csv')
# Check the Data type of the output returned by pd.read_csv()



print(type(Movie_DF))
# Show the content of first 5 observations in a Dataframe



Movie_DF.head()
# Show the content of last 5 observations in a Dataframe



Movie_DF.tail()
# Show the content of first 3 observations in a Dataframe



Movie_DF.head(3)
# Fetch all the Index values from the Dataframe.



Movie_index = Movie_DF.index



print("The values in the index column is : " , Movie_index)



print('\n')



print("The Data Type of the Index is     : " , type(Movie_index))
# Fetch all the Column Names from the Dataframe.



Movie_Col_Names = Movie_DF.columns



print("The Column Names are  : " , Movie_Col_Names)



print('\n')



print("The Data Type of the Column Names is     : " , type(Movie_Col_Names))
# Fetch all the Values from the Dataframe.



Movie_Values = Movie_DF.values



print("The Values in the Dataframe are  : " , Movie_Values)



print('\n')



print("The Data Type of the Values is     : " , type(Movie_Values))
# Display the structure of a Dataframe



Movie_DF.info()
# Just display the Data types of the columns of a Dataframe



Movie_DF.dtypes
# Display the cummulative Data Type Counts in a Dataframe



Movie_DF.get_dtype_counts()
# Extract the values in the column 'director_name' from the Movies DataFrame



Movie_DF['director_name']
# Extract the values in the column 'director_name' from the Movies DataFrame usinf dot approach

# Note - This approach is not encouraged as it will not work when a column name has blank space



Movie_DF.director_name
# Display the Data type of a Column extracted from a DataFrame



print(type(Movie_DF['director_name']))
# Create 2 Series from the Movies Dataframe.



director = Movie_DF['director_name']

actor_1_fb_likes = Movie_DF['actor_1_facebook_likes']



print(type(director))

print(type(actor_1_fb_likes))
print(director.head())



print('\n')  # Print an Empty Line 



print(actor_1_fb_likes.head(3))
# Display the Count of Unique values in a SERIES 



print(director.value_counts())
# Display the Statistical Summary of a Series of Numerical Data Type



print(actor_1_fb_likes.describe())
# Display the Summary of a Series of Categorical Data Type



print(director.describe())
# Check if any value in a Series is Missing 



director.isnull()
# Replace the Missing Values with 0 



director_flled = director.fillna(0)



print("Count of Non Missing Values before applying fillna() :" ,director.count())

print("Count of Non Missing Values after applying fillna()  :" ,director_flled.count())
# Replace the Missing Values with 0 



director_NA_Dropped = director.dropna()



print("Count of elements before applying dropna() :" ,director.size)

print("Count of elements after applying dropna()  :" ,director_NA_Dropped.size)
# Applying multiplaction operator on Series containing String data type 

# Resulting in each Value being concatenated to itself



director * 2
# Applying Division operator with the Series containing Numeric Data 

# Resulting in each value being divided by 100



actor_1_fb_likes / 100
# Example of Method Chaining 



director.value_counts().head(3)
# Find the total number of Missing values in a SERIES 



actor_1_fb_likes.isnull().sum()
# Specify the Index Column during the read_csv() step 



Movie_DF2 = pd.read_csv('../input/movie.csv', 

                        index_col='movie_title')



print(Movie_DF2.head())
# Change the Index Column after the read_csv() step 



Movie_DF3 = Movie_DF.set_index('movie_title')



print(Movie_DF3.head())
Movie_DF4 = Movie_DF3.reset_index()



Movie_DF4.head()
# Count & Display the number of Columns in a Dataframe



Movie_DF.columns.size
# Add a New column to a DataFrame



Movie_DF['Tot_FB_Actors_Likes'] = Movie_DF.actor_1_facebook_likes + Movie_DF.actor_2_facebook_likes + Movie_DF.actor_3_facebook_likes
# Count & Display the number of Columns in a Dataframe after adding a new column



Movie_DF.columns.size
# Drop the new column from the Dataframe



Movie_DF.drop('Tot_FB_Actors_Likes',axis='columns' , inplace=True)
# Count & Display the number of Columns in a Dataframe after dropping the new column



Movie_DF.columns.size
# Creata a DataFrame which is consisting of subset of the columns from the Original dataFrame



Movie_actor_director = Movie_DF[['actor_1_name', 'actor_2_name','actor_3_name', 'director_name']]



Movie_actor_director.head()




Movie_DF.get_dtype_counts()
# Extract only those columns which are of Integer Data Type



Movie_DF.select_dtypes(include=['int']).head()
# Extract only those columns which are of Numeric Data Type



Movie_DF.select_dtypes(include=['number']).head()
# Extract only those columns which has 'facebook' in the Column name



Movie_DF.filter(like='facebook').head()
# get the shape & size of a DataFrame



print(Movie_DF.shape)



print(Movie_DF.size)
# count method is used to find the number of non-missing values for each column.



print(Movie_DF.count())
# Find the Statistical Summary of each column in the DataFrame



print(Movie_DF.describe())