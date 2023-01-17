import pandas as pd #importing pandas for dataframe manipulation, data processing, CSV file I/O (e.g. pd.read_csv)
#read file and save to a pandas dataframe, a two-dimensional, size-mutatable, potentially heterogeneous data structure 
df =  pd.read_csv('../input/Pokemon.csv') 
#Show the columns contained within the dataframe. 
df.columns
#Show the shape of the dataframe, number of rows by number of columns
df.shape
#Show the data type for each column in the dataframe
df.dtypes
#Show the indexes of the dataframe, you can use the index to filter or select data 
df.index
#Perhaps we want to change the index?
#set the index to the named attribute in the dataframe
df = df.set_index('Name')

#Show the index, which will now be NAME
df.index
#sort values by the index, and append .head() to return only our head records
df.sort_index().head()
#Show the index data type and memory information
df.info()
#df.head(n) returns a DataFrame holding the first n rows of df. Useful to briefly browse our data. df.tail returns a DataFrame holding the bottom n rows of df. Useful to briefly browse our data

df.head(n=5)
#Using df.sort_values() to sort our data by a specific attribute. Sort values by Attack attribute, default is ascending
#Appending .head(n=5) filters the result set to only the first 5 results
df.sort_values(by='Attack').head(n=5)
#Sort values by TYPE2 attribute, using some of the function's parameters. This time we want descending, and show the NaN last
df.sort_values(by='Type 2', ascending=False, na_position='last').tail(n=5)
#Select a single column to return a Series
df['Type 1'].head(n=5)
#Slice rows starting at row 10 up to row 15
df[10:15]
#Select data based on the index - retrieves all data for index label Bulbasaur
df.loc['Bulbasaur'] 
#Same return, but using index position key value
df.iloc[0] 
#Selecting on a multi-axis using label and attribute name
df.loc['Bulbasaur':'Venusaur',['Type 1','Type 2']]
#Alternatively, Selecting on a multi-axis using integers
df.iloc[0:3,1:3]
#Showing how to return all attributes for a multi-axis subset
df.iloc[0:3,:]
#Return data by filtering on a columns integer value   
df[df.HP > 150]

#Give us a distinct list of TYPE1 values
df['Type 1'].unique()
#Show us the results where we've filtered for a specific value in an attribute
df[df['Type 1']=='Dragon'].head(5) 
#Return data by filtering on multiple columns
df[((df['Type 1']=='Fire')  & (df['Type 2']=='Dragon'))]
#Alternatively via boolean indexing we can use isin() method to filter 
df[df['Type 1'].isin(['Dragon'])].head(n=5)
#Return index value with highest value
df['Defense'].idxmax()
#Return index value with lowest value
df['Attack'].idxmin()
df.max()
#Take a look at some of the averages of each attribute
df.mean(axis=0)
#Get mean value of a specific column
df['HP'].mean()
#Histogramming, getting the counts of values in our data. Now we can start to see the distribution of Pokémon over their types
df['Type 1'].value_counts()
#Filter out dataframe for only Legendary Pokémon, and get the mean Attack as per our stakeholders request
df[(df['Legendary']==True)].Attack.mean()
#Show a brief statistical summary of the data frame
df.describe()
#Before we maniuplate, it might be prudent to create a copy of our data so we can return to it if needed
df2 = df.copy()
#drop the columns with axis=1; By the way, axis=0 is for rows
df2=df.drop(['#'],axis=1) 

#Showing that # Column is now dropped
df2.head(n=5)         
#pandas primarily uses the np.nan to represent missing data. It is by default not included in computations
#Get the boolean mask where values are nan            
pd.isna(df2).head(10)
#Again, take a copy. This time genuinley we will revert as the next step is just a demonstration
df3 = df2.copy()

#Drop any rows containing NaN values. However, sometimes you may not want to do this as it might start to affect your analysis
df3.dropna(how='any')    
pd.isna(df3).head(10)
# The new index of Names contains erroneous text. We want to remove all the text before "Mega"
df2.index = df.index.str.replace(".*(?=Mega)", "")

df2.head(5)