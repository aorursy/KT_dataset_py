# conventional way to import pandas
import pandas as pd

# allow plots to appear in the notebook
%matplotlib inline
# read_csv is used to read a comma separated file.
df_ufo = pd.read_csv('../input/ufo-sightings/ufo.csv')
type(df_ufo)
# examine the first 5 rows
df_ufo.head()
df_ufo.info()
# examine the column names
df_ufo.columns
# rename two of the columns by using the 'rename' method
df_ufo.rename(columns={'Colors Reported':'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace=True)
df_ufo.columns
# select the 'City' Series using bracket notation
# df_ufo['City']

# or equivalently, use dot notation
df_ufo.City
# type(ufo.City)
# create a new 'Location' Series (must use bracket notation to define the Series name)
df_ufo['Location'] = df_ufo.City + ', ' + df_ufo.State
df_ufo.head()
df_ufo.Location
# remove a single column (axis=1 refers to columns)
df_ufo.drop('Colors_Reported', axis=1, inplace=True)
df_ufo.head()
# remove multiple columns at once
df_ufo.drop(['City', 'State'], axis=1, inplace=True)
df_ufo.head()
# instead of dropping the columns with missing data, let's try something else.
# re-read the dataset of UFO reports into a DataFrame
df_ufo = pd.read_csv('../input/ufo-sightings/ufo.csv')
df_ufo.head()
# 'isnull' returns a DataFrame of booleans (True if missing, False if not missing)
df_ufo.isnull().head(20)
# count the number of missing values in each Series
df_ufo.isnull().sum()
# use the 'isnull' Series method to filter the DataFrame rows
df_ufo[df_ufo.City.isnull()].head(25)
# fill in missing values with a specified value
df_ufo['Shape Reported'].fillna(value='Unknown', inplace=True)
# confirm that the missing values were filled in
df_ufo['Shape Reported'].value_counts().head(10)
df_ufo.shape

# read a dataset of alcohol consumption into a DataFrame, and check the data types
df_drinks = pd.read_csv('../input/drinks/drinks.csv')
type(df_drinks)
df_drinks.info()
df_drinks.dtypes
# get the mean of all the numeric columns
df_drinks.mean()
# describe all of the numeric columns
df_drinks.describe()
# pass the string 'all' to describe all columns
df_drinks.describe(include='all')
df_drinks.head()
df_drinks.tail()
df_drinks[df_drinks.beer_servings > 300]
df_drinks[df_drinks.wine_servings > 275]
# calculate the mean beer servings just for countries in Africa
df_drinks[df_drinks.continent=='Africa'].beer_servings.mean()
# calculate the mean beer servings for each continent
df_drinks.groupby('continent').wine_servings.mean()
# multiple aggregation functions can be applied simultaneously
df_drinks.groupby('continent').beer_servings.agg(['count', 'mean', 'min', 'max'])
# What are the 23 countries in North America
df_drinks[df_drinks.continent == "North America"]
df_drinks.groupby('continent').mean()
# side-by-side bar plot of the DataFrame directly above
df_drinks.groupby('continent').mean().plot(kind='bar')
