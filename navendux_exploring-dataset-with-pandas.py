import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
ign_reviews_dataframe = pd.read_csv("../input/ign.csv")
ign_reviews_dataframe.describe()
# check shape of dataframe
ign_reviews_dataframe.shape
# review top 5 records
ign_reviews_dataframe.head()
# retrieving specific rows and columns using pandas.DataFrame.iloc method [rows, colums]
ign_reviews_dataframe.iloc[0:5, :] # replicates .head()
# removing first useless column
ign_reviews_dataframe = ign_reviews_dataframe.iloc[:,1:]
ign_reviews_dataframe.head()
# We can work with labels using the pandas.DataFrame.loc method, which allows us to index using labels instead of positions.
ign_reviews_dataframe.loc[0:5,:]
some_reviews = ign_reviews_dataframe.loc[10:20,:]
some_reviews.head() # check row labels
some_reviews.loc[10:20,:]
some_reviews.iloc[:5,:]
some_reviews.loc[:5,:] # since loc goes for specific label this won't display any result
ign_reviews_dataframe.loc[:5,'score'] # can be used to fetch specific column using labels
ign_reviews_dataframe.loc[:5, ['score', 'genre', 'release_year']] # to fetch multiple columns using labels
# we can also directly retrieve specific column
ign_reviews_dataframe['score']
# or multiple columns
ign_reviews_dataframe[['score', 'genre']]
type(ign_reviews_dataframe)
type(ign_reviews_dataframe['score']) # single column/row in dataframe is called Series
type(ign_reviews_dataframe[['score', 'genre']])
prime = pd.Series([1,2,3,5]) # define example series of prime numbers
type(prime)
prime
cities = pd.Series(["jhansi", "mumbai", "indore", "bangalore"]) # string series
cities
mixed = pd.Series(["apple", 10, "orange", 100]) # mixed types
mixed
d1 = pd.DataFrame([prime,cities,mixed])
d1.head()
# specifiy row and column names
d2 = pd.DataFrame([["rlps","iitb"],['1999','2005']], columns = ["school", "year"], index=["row1","row2"]) 
d2
d2.loc["row1","school"]
d2.loc[:,"school"]
# another way to define column name using dictionary format
d3 = pd.DataFrame({"airport": ["london","houston"],"landed_first":['2006','2008']}) 

d3
# starting analysis on ign data
ign_reviews_dataframe.head()
titles = ign_reviews_dataframe['title']
titles.head()
# calculating mean of score
ign_reviews_dataframe['score'].mean()
#calculating mean of all numerical columns
ign_reviews_dataframe.mean()
# to calculate mean for all rows, change axis to 1,default is 0 - refering to column.
# Again this will calculate mean only of numerical values and ignore other types
ign_reviews_dataframe.mean(axis=1)
ign_reviews_dataframe.corr()
ign_reviews_dataframe.std()
# For example, we can divide every value in the score column by 2 to switch the scale from 0-10 to 0-5:
ign_reviews_dataframe['score']/2
score_filter = ign_reviews_dataframe["score"] > 7
score_filter
# now select only those rows that satisfy the defined filter
filtered_reviews = ign_reviews_dataframe[score_filter]
filtered_reviews.shape
# defining multiple coditions in a filter
xbox_one_high_reviews_filter = (ign_reviews_dataframe["score"]>7) & (ign_reviews_dataframe["platform"] == "Xbox One")
xbox_one_high_reviews = ign_reviews_dataframe[xbox_one_high_reviews_filter]
xbox_one_high_reviews.head()
# using lambda function
xbox_high_reviews_filter = (ign_reviews_dataframe["score"]>7) & (ign_reviews_dataframe["platform"].apply(lambda platform: platform.startswith('Xbox')))
xbox_high_reviews = ign_reviews_dataframe[xbox_one_high_reviews_filter]
xbox_high_reviews.shape
# Call %matplotlib inline to set up plotting inside a Jupyter notebook.
%matplotlib inline
ign_reviews_dataframe.hist('score')
ign_reviews_dataframe[ign_reviews_dataframe["platform"] == "Xbox One"]["score"].plot(kind="hist")
xbox_one_high_reviews['score'].hist()
phrase_filter = ign_reviews_dataframe["score_phrase"].apply(lambda phrase: phrase.startswith('Amazing'))
filtered_reviews = ign_reviews_dataframe[phrase_filter]
filtered_reviews['score'].hist()
phrase_filter = ign_reviews_dataframe["score_phrase"].apply(lambda phrase: phrase.startswith('Great'))
filtered_reviews = ign_reviews_dataframe[phrase_filter]
filtered_reviews['score'].hist()
# check for null values that may be important for analysis
null_columns=ign_reviews_dataframe.columns[ign_reviews_dataframe.isnull().any()]
print(ign_reviews_dataframe[ign_reviews_dataframe.isnull().any(axis=1)][null_columns].head())
#Group By
ign_reviews_dataframe.groupby('platform').mean()
# Pivot Table
recent_filter = ign_reviews_dataframe['release_year'] > 2010
ign_reviews_dataframe[recent_filter].pivot_table(values="score", index=['platform','genre'], columns=['release_year'])

