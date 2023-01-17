# conventional way to import pandas
import pandas as pd

orders = pd.read_table('../input/chipotle.csv')
ufo = pd.read_csv('../input/ufo.csv')
movies = pd.read_csv('../input/imdb_1000.csv')
drinks = pd.read_csv('../input/drinks.csv')
train = pd.read_csv('../input/titanic_train.csv')
test = pd.read_csv('../input/titanic_test.csv')
# read a dataset of movie reviewers (modifying the default parameter values for read_table)

# Works with python 2.7
#user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
#users = pd.read_table('../input/movieusers', sep='|', header=None, names=user_cols)
# examine the first 5 rows
#users.head()
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_table('../input/uforeports', sep=',')
# read_csv is equivalent to read_table, except it assumes a comma separator
#ufo = pd.read_csv('http://bit.ly/uforeports')
# examine the first 5 rows
ufo.head()
# select the 'City' Series using bracket notation
ufo['City']

# or equivalently, use dot notation
ufo.City
# create a new 'Location' Series (must use bracket notation to define the Series name)
ufo['Location'] = ufo.City + ', ' + ufo.State
ufo.head()
# read a dataset of top-rated IMDb movies into a DataFrame
#movies = pd.read_csv('http://bit.ly/imdbratings')
# example method: show the first 5 rows
movies.head()
# example method: calculate summary statistics
movies.describe()
# example attribute: number of rows and columns
movies.shape
# example attribute: data type of each column
movies.dtypes
# use an optional parameter to the describe method to summarize only 'object' columns
movies.describe(include=['object'])
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo = pd.read_csv('../input/ufo.csv')
# examine the column names
ufo.columns
# rename two of the columns by using the 'rename' method
ufo.rename(columns={'Colors Reported':'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace=True)
ufo.columns
ufo.columns
# replace all of the column names by overwriting the 'columns' attribute
ufo_cols = ['City', 'Colors_Reported', 'Shape_Reported', 'State', 'Time']
ufo.columns = ufo_cols
ufo.columns
# replace the column names during the file reading process by using the 'names' parameter
ufo = pd.read_csv('../input/ufo.csv')
ufo.columns
# replace all spaces with underscores in the column names by using the 'str.replace' method
ufo.columns = ufo.columns.str.replace(' ', '_')
ufo.columns
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head()
# remove a single column (axis=1 refers to columns)
ufo.drop('Colors_Reported', axis=1, inplace=True)
ufo.head()
# remove multiple columns at once
ufo.drop(['City', 'State'], axis=1, inplace=True)
ufo.head()
# remove multiple rows at once (axis=0 refers to rows)
ufo.drop([0, 1], axis=0, inplace=True)
ufo.head()
# read a dataset of top-rated IMDb movies into a DataFrame
#movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()
# sort the 'title' Series in ascending order (returns a Series)
movies.title.sort_values().head()
# sort in descending order instead
movies.title.sort_values(ascending=False).head()
# sort the entire DataFrame by the 'title' Series (returns a DataFrame)
movies.sort_values('title').head()
# sort in descending order instead
movies.sort_values('title', ascending=False).head()
# sort the DataFrame first by 'content_rating', then by 'duration'
movies.sort_values(['content_rating', 'duration']).head()
# read a dataset of top-rated IMDb movies into a DataFrame
#movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()
# examine the number of rows and columns
movies.shape
# create a list in which each element refers to a DataFrame row: True if the row satisfies the condition, False otherwise
booleans = []
for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)
# confirm that the list has the same length as the DataFrame
len(booleans)
# examine the first five list elements
booleans[0:5]
# convert the list to a Series
is_long = pd.Series(booleans)
is_long.head()
# use bracket notation with the boolean Series to tell the DataFrame which rows to display
movies[is_long]
# simplify the steps above: no need to write a for loop to create 'is_long' since pandas will broadcast the comparison
is_long = movies.duration >= 200
movies[is_long]

# or equivalently, write it in one line (no need to create the 'is_long' object)
movies[movies.duration >= 200]
# select the 'genre' Series from the filtered DataFrame
movies[movies.duration >= 200].genre

# or equivalently, use the 'loc' method
movies.loc[movies.duration >= 200, 'genre']
# read a dataset of top-rated IMDb movies into a DataFrame
#movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()
# filter the DataFrame to only show movies with a 'duration' of at least 200 minutes
movies[movies.duration >= 200]
# demonstration of the 'and' operator
print(True and True)
print(True and False)
print(False and False)
# demonstration of the 'or' operator
print(True or True)
print(True or False)
print(False or False)
# CORRECT: use the '&' operator to specify that both conditions are required
movies[(movies.duration >=200) & (movies.genre == 'Drama')]
# INCORRECT: using the '|' operator would have shown movies that are either long or dramas (or both)
movies[(movies.duration >=200) | (movies.genre == 'Drama')].head()
# use the '|' operator to specify that a row can match any of the three criteria
movies[(movies.genre == 'Crime') | (movies.genre == 'Drama') | (movies.genre == 'Action')].head(10)

# or equivalently, use the 'isin' method
movies[movies.genre.isin(['Crime', 'Drama', 'Action'])].head(10)
# read a dataset of UFO reports into a DataFrame, and check the columns
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.columns
# specify which columns to include by name
ufo = pd.read_csv('../input/ufo.csv', usecols=['City', 'State'])

# or equivalently, specify columns by position
ufo = pd.read_csv('../input/ufo.csv', usecols=[0, 4])
ufo.columns
# specify how many rows to read
#ufo = pd.read_csv('http://bit.ly/uforeports', nrows=3)
ufo
# Series are directly iterable (like a list)
#for c in ufo.City:
#    print(c)
# various methods are available to iterate through a DataFrame
#for index, row in ufo.iterrows():
    #print(index, row.City, row.Time)
# read a dataset of alcohol consumption into a DataFrame, and check the data types
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.dtypes
# only include numeric columns in the DataFrame
import numpy as np
drinks.select_dtypes(include=[np.number]).dtypes
# describe all of the numeric columns
drinks.describe()
# pass the string 'all' to describe all columns
drinks.describe(include='all')
# pass a list of data types to only describe certain types
drinks.describe(include=['object', 'float64'])
# pass a list even if you only want to describe a single data type
drinks.describe(include=['object'])
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# drop a column (temporarily)
drinks.drop('continent', axis=1).head()
# drop a row (temporarily)
drinks.drop(2, axis=0).head()
# calculate the mean of each numeric column
drinks.mean()

# or equivalently, specify the axis explicitly
drinks.mean(axis=0)
# calculate the mean of each row
drinks.mean(axis=1).head()
# 'index' is an alias for axis 0
drinks.mean(axis='index')
# 'columns' is an alias for axis 1
drinks.mean(axis='columns').head()
# read a dataset of Chipotle orders into a DataFrame
orders = pd.read_csv('../input/chipotle.csv')
orders.head()
# normal way to access string methods in Python
'hello'.upper()
# string methods for pandas Series are accessed via 'str'
orders.item_name.str.upper().head()
# string method 'contains' checks for a substring and returns a boolean Series
orders.item_name.str.contains('Chicken').head()
# use the boolean Series to filter the DataFrame
orders[orders.item_name.str.contains('Chicken')].head()
# string methods can be chained together
orders.choice_description.str.replace('[', '').str.replace(']', '').head()
# many pandas string methods support regular expressions (regex)
orders.choice_description.str.replace('[\[\]]', '').head()
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# examine the data type of each Series
drinks.dtypes
# change the data type of an existing Series
drinks['beer_servings'] = drinks.beer_servings.astype(float)
drinks.dtypes
# alternatively, change the data type of a Series while reading in a file
#drinks = pd.read_csv('http://bit.ly/drinksbycountry', dtype={'beer_servings':float})
drinks.dtypes
# read a dataset of Chipotle orders into a DataFrame
#orders = pd.read_table('http://bit.ly/chiporders')
orders.head()
# examine the data type of each Series
orders.dtypes
# convert a string to a number in order to do math
orders.item_price.str.replace('$', '').astype(float).mean()
# string method 'contains' checks for a substring and returns a boolean Series
orders.item_name.str.contains('Chicken').head()
# convert a boolean Series to an integer (False = 0, True = 1)
orders.item_name.str.contains('Chicken').astype(int).head()
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# calculate the mean beer servings across the entire dataset
drinks.beer_servings.mean()
# calculate the mean beer servings just for countries in Africa
drinks[drinks.continent=='Africa'].beer_servings.mean()
# calculate the mean beer servings for each continent
drinks.groupby('continent').beer_servings.mean()
# other aggregation functions (such as 'max') can also be used with groupby
drinks.groupby('continent').beer_servings.max()
# multiple aggregation functions can be applied simultaneously
drinks.groupby('continent').beer_servings.agg(['count', 'mean', 'min', 'max'])
# specifying a column to which the aggregation function should be applied is not required
drinks.groupby('continent').mean()
# allow plots to appear in the notebook
%matplotlib inline
# side-by-side bar plot of the DataFrame directly above
drinks.groupby('continent').mean().plot(kind='bar')
# read a dataset of top-rated IMDb movies into a DataFrame
#movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()
# examine the data type of each Series
movies.dtypes
# count the non-null values, unique values, and frequency of the most common value
movies.genre.describe()
# count how many times each value in the Series occurs
movies.genre.value_counts()
# display percentages instead of raw counts
movies.genre.value_counts(normalize=True)
# 'value_counts' (like many pandas methods) outputs a Series
type(movies.genre.value_counts())
# thus, you can add another Series method on the end
movies.genre.value_counts().head()
# display the unique values in the Series
movies.genre.unique()
# count the number of unique values in the Series
movies.genre.nunique()
# compute a cross-tabulation of two Series
pd.crosstab(movies.genre, movies.content_rating)
# calculate various summary statistics
movies.duration.describe()
# many statistics are implemented as Series methods
movies.duration.mean()
# 'value_counts' is primarily useful for categorical data, not numerical data
movies.duration.value_counts().head()
# allow plots to appear in the notebook
%matplotlib inline
# histogram of the 'duration' Series (shows the distribution of a numerical variable)
movies.duration.plot(kind='hist')
# bar plot of the 'value_counts' for the 'genre' Series
movies.genre.value_counts().plot(kind='bar')
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.tail()
# 'isnull' returns a DataFrame of booleans (True if missing, False if not missing)
ufo.isnull().tail()
# 'nonnull' returns the opposite of 'isnull' (True if not missing, False if missing)
ufo.notnull().tail()
# count the number of missing values in each Series
ufo.isnull().sum()
# use the 'isnull' Series method to filter the DataFrame rows
ufo[ufo.City.isnull()].head()
# examine the number of rows and columns
ufo.shape
# if 'any' values are missing in a row, then drop that row
ufo.dropna(how='any').shape
# 'inplace' parameter for 'dropna' is False by default, thus rows were only dropped temporarily
ufo.shape
# if 'all' values are missing in a row, then drop that row (none are dropped in this case)
ufo.dropna(how='all').shape
# if 'any' values are missing in a row (considering only 'City' and 'Shape Reported'), then drop that row
ufo = pd.read_csv('../input/ufo.csv')
ufo.dropna(subset=['City', 'Shape Reported'], how='any').shape
# if 'all' values are missing in a row (considering only 'City' and 'Shape Reported'), then drop that row
ufo.dropna(subset=['City', 'Shape Reported'], how='all').shape
# 'value_counts' does not include missing values by default
ufo['Shape Reported'].value_counts().head()
# explicitly include missing values
ufo['Shape Reported'].value_counts(dropna=False).head()
# fill in missing values with a specified value
ufo['Shape Reported'].fillna(value='VARIOUS', inplace=True)
# confirm that the missing values were filled in
ufo['Shape Reported'].value_counts().head()
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# every DataFrame has an index (sometimes called the "row labels")
drinks.index
# column names are also stored in a special "index" object
drinks.columns
# neither the index nor the columns are included in the shape
drinks.shape
# index and columns both default to integers if you don't define them
#pd.read_table('http://bit.ly/movieusers', header=None, sep='|').head()
# identification: index remains with each row when filtering the DataFrame
drinks[drinks.continent=='South America']
# selection: select a portion of the DataFrame using the index
drinks.loc[23, 'beer_servings']
# set an existing column as the index
drinks.set_index('country', inplace=True)
drinks.head()
# 'country' is now the index
drinks.index
# 'country' is no longer a column
drinks.columns
# 'country' data is no longer part of the DataFrame contents
drinks.shape
# country name can now be used for selection
drinks.loc['Brazil', 'beer_servings']
# index name is optional
drinks.index.name = None
drinks.head()
# restore the index name, and move the index back to a column
drinks.index.name = 'country'
drinks.reset_index(inplace=True)
drinks.head()
# many DataFrame methods output a DataFrame
drinks.describe()
# you can interact with any DataFrame using its index and columns
drinks.describe().loc['25%', 'beer_servings']
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# every DataFrame has an index
drinks.index
# every Series also has an index (which carries over from the DataFrame)
drinks.continent.head()
# set 'country' as the index
drinks.set_index('country', inplace=True)
# Series index is on the left, values are on the right
drinks.continent.head()
# another example of a Series (output from the 'value_counts' method)
drinks.continent.value_counts()
# access the Series index
drinks.continent.value_counts().index
# access the Series values
drinks.continent.value_counts().values
# elements in a Series can be selected by index (using bracket notation)
drinks.continent.value_counts()['Africa']
# any Series can be sorted by its values
drinks.continent.value_counts().sort_values()
# any Series can also be sorted by its index
drinks.continent.value_counts().sort_index()
# 'beer_servings' Series contains the average annual beer servings per person
drinks.beer_servings.head()
# create a Series containing the population of two countries
people = pd.Series([3000000, 85000], index=['Albania', 'Andorra'], name='population')
people
# calculate the total annual beer servings for each country
(drinks.beer_servings * people).head()
# concatenate the 'drinks' DataFrame with the 'population' Series (aligns by the index)
pd.concat([drinks, people], axis=1).head()
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head(3)
# row 0, all columns
ufo.loc[0, :]
# rows 0 and 1 and 2, all columns
ufo.loc[[0, 1, 2], :]
# rows 0 through 2 (inclusive), all columns
ufo.loc[0:2, :]
# this implies "all columns", but explicitly stating "all columns" is better
ufo.loc[0:2]
# rows 0 through 2 (inclusive), column 'City'
ufo.loc[0:2, 'City']
# rows 0 through 2 (inclusive), columns 'City' and 'State'
ufo.loc[0:2, ['City', 'State']]
# accomplish the same thing using double brackets - but using 'loc' is preferred since it's more explicit
ufo[['City', 'State']].head(3)
# rows 0 through 2 (inclusive), columns 'City' through 'State' (inclusive)
ufo.loc[0:2, 'City':'State']
# accomplish the same thing using 'head' and 'drop'
ufo.head(3).drop('Time', axis=1)
# rows in which the 'City' is 'Oakland', column 'State'
ufo.loc[ufo.City=='Oakland', 'State']
# accomplish the same thing using "chained indexing" - but using 'loc' is preferred since chained indexing can cause problems
ufo[ufo.City=='Oakland'].State
# rows in positions 0 and 1, columns in positions 0 and 3
ufo.iloc[[0, 1], [0, 3]]
# rows in positions 0 through 2 (exclusive), columns in positions 0 through 4 (exclusive)
ufo.iloc[0:2, 0:4]
# rows in positions 0 through 2 (exclusive), all columns
ufo.iloc[0:2, :]
# accomplish the same thing - but using 'iloc' is preferred since it's more explicit
ufo[0:2]
# read a dataset of alcohol consumption into a DataFrame and set 'country' as the index
#drinks = pd.read_csv('http://bit.ly/drinksbycountry', index_col='country')
drinks.head()
# row with label 'Albania', column in position 0
drinks.ix['Albania', 0]
# row in position 1, column with label 'beer_servings'
drinks.ix[1, 'beer_servings']
# rows 'Albania' through 'Andorra' (inclusive), columns in positions 0 through 2 (exclusive)
drinks.ix['Albania':'Andorra', 0:2]
# rows 0 through 2 (inclusive), columns in positions 0 through 2 (exclusive)
ufo.ix[0:2, 0:2]
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head()
ufo.shape
# remove the 'City' column (doesn't affect the DataFrame since inplace=False)
ufo.drop('City', axis=1).head()
# confirm that the 'City' column was not actually removed
ufo.head()
# remove the 'City' column (does affect the DataFrame since inplace=True)
ufo.drop('City', axis=1, inplace=True)
# confirm that the 'City' column was actually removed
ufo.head()
# drop a row if any value is missing from that row (doesn't affect the DataFrame since inplace=False)
ufo.dropna(how='any').shape
# confirm that no rows were actually removed
ufo.shape
# use an assignment statement instead of the 'inplace' parameter
ufo = ufo.set_index('Time')
ufo.tail()
# fill missing values using "backward fill" strategy (doesn't affect the DataFrame since inplace=False)
ufo.fillna(method='bfill').tail()
# compare with "forward fill" strategy (doesn't affect the DataFrame since inplace=False)
ufo.fillna(method='ffill').tail()
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# exact memory usage is unknown because object columns are references elsewhere
drinks.info()
# force pandas to calculate the true memory usage
drinks.info(memory_usage='deep')
# calculate the memory usage for each Series (in bytes)
drinks.memory_usage(deep=True)
# use the 'category' data type (new in pandas 0.15) to store the 'continent' strings as integers
drinks['continent'] = drinks.continent.astype('category')
drinks.dtypes
# 'continent' Series appears to be unchanged
drinks.continent.head()
# strings are now encoded (0 means 'Africa', 1 means 'Asia', 2 means 'Europe', etc.)
drinks.continent.cat.codes.head()
# memory usage has been drastically reduced
drinks.memory_usage(deep=True)
# repeat this process for the 'country' Series
drinks['country'] = drinks.country.astype('category')
drinks.memory_usage(deep=True)
# memory usage increased because we created 193 categories
drinks.country.cat.categories
# create a small DataFrame from a dictionary
df = pd.DataFrame({'ID':[100, 101, 102, 103], 'quality':['good', 'very good', 'good', 'excellent']})
df
# sort the DataFrame by the 'quality' Series (alphabetical order)
df.sort_values('quality')
# define a logical ordering for the categories
df['quality'] = df.quality.astype('category', categories=['good', 'very good', 'excellent'], ordered=True)
df.quality
# sort the DataFrame by the 'quality' Series (logical order)
df.sort_values('quality')
# comparison operators work with ordered categories
df.loc[df.quality > 'good', :]
# read the training dataset from Kaggle's Titanic competition into a DataFrame
#train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()
# create a feature matrix 'X' by selecting two DataFrame columns
feature_cols = ['Pclass', 'Parch']
X = train.loc[:, feature_cols]
X.shape
# create a response vector 'y' by selecting a Series
y = train.Survived
y.shape
# fit a classification model to the training data
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, y)
# read the testing dataset from Kaggle's Titanic competition into a DataFrame
#test = pd.read_csv('http://bit.ly/kaggletest')
test.head()
# create a feature matrix from the testing data that matches the training data
X_new = test.loc[:, feature_cols]
X_new.shape
# use the fitted model to make predictions for the testing set observations
new_pred_class = logreg.predict(X_new)
# create a DataFrame of passenger IDs and testing set predictions
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).head()
# ensure that PassengerID is the first column by setting it as the index
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').head()
# write the DataFrame to a CSV file that can be submitted to Kaggle
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').to_csv('sub.csv')
# save a DataFrame to disk ("pickle it")
train.to_pickle('train.pkl')
# read a pickled object from disk ("unpickle it")
pd.read_pickle('train.pkl').head()
# read a dataset of UFO reports into a DataFrame
#ufo = pd.read_csv('http://bit.ly/uforeports')
ufo.head()
# use 'isnull' as a top-level function
pd.isnull(ufo).head()
# equivalent: use 'isnull' as a DataFrame method
ufo.isnull().head()
# position-based slicing is inclusive of the start and exclusive of the stop
ufo.iloc[0:4, :]
# 'iloc' is simply following NumPy's slicing convention...
ufo.values[0:4, :]
# ...and NumPy is simply following Python's slicing convention
'python'[0:4]
# sample 3 rows from the DataFrame without replacement (new in pandas 0.16.1)
ufo.sample(n=3)
# use the 'random_state' parameter for reproducibility
ufo.sample(n=3, random_state=42)
# sample 75% of the DataFrame's rows without replacement
train = ufo.sample(frac=0.75, random_state=99)
# store the remaining 25% of the rows in another DataFrame
test = ufo.loc[~ufo.index.isin(train.index), :]
# read the training dataset from Kaggle's Titanic competition
train = pd.read_csv('../input/titanic_train.csv')
train.head()
# create the 'Sex_male' dummy variable using the 'map' method
train['Sex_male'] = train.Sex.map({'female':0, 'male':1})
train.head()
# alternative: use 'get_dummies' to create one column for every possible value
pd.get_dummies(train.Sex).head()
# drop the first dummy variable ('female') using the 'iloc' method
pd.get_dummies(train.Sex).iloc[:, 1:].head()
# add a prefix to identify the source of the dummy variables
pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 1:].head()
# use 'get_dummies' with a feature that has 3 possible values
pd.get_dummies(train.Embarked, prefix='Embarked').head(10)
# drop the first dummy variable ('C')
pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:, 1:].head(10)
# save the DataFrame of dummy variables and concatenate them to the original DataFrame
embarked_dummies = pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:, 1:]
train = pd.concat([train, embarked_dummies], axis=1)
train.head()
# reset the DataFrame
#train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()
# pass the DataFrame to 'get_dummies' and specify which columns to dummy (it drops the original columns)
pd.get_dummies(train, columns=['Sex', 'Embarked']).head()
# use the 'drop_first' parameter (new in pandas 0.18) to drop the first dummy variable for each feature
pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True).head()
# read a dataset of UFO reports into a DataFrame
ufo = pd.read_csv('../input/ufo.csv')
ufo.head()
# 'Time' is currently stored as a string
ufo.dtypes
# hour could be accessed using string slicing, but this approach breaks too easily
ufo.Time.str.slice(-5, -3).astype(int).head()
# convert 'Time' to datetime format
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo.head()
ufo.dtypes
# convenient Series attributes are now available
ufo.Time.dt.hour.head()
ufo.Time.dt.weekday_name.head()
ufo.Time.dt.dayofyear.head()
# convert a single string to datetime format (outputs a timestamp object)
ts = pd.to_datetime('1/1/1999')
ts
# compare a datetime Series with a timestamp
ufo.loc[ufo.Time >= ts, :].head()
# perform mathematical operations with timestamps (outputs a timedelta object)
ufo.Time.max() - ufo.Time.min()
# timedelta objects also have attributes you can access
(ufo.Time.max() - ufo.Time.min()).days
# allow plots to appear in the notebook
%matplotlib inline
# count the number of UFO reports per year
ufo['Year'] = ufo.Time.dt.year
ufo.Year.value_counts().sort_index().head()
# plot the number of UFO reports per year (line plot is the default)
ufo.Year.value_counts().sort_index().plot()
# read a dataset of movie reviewers into a DataFrame
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('../input/use.csv', sep='|', header=None, names=user_cols, index_col='user_id')
users.head()
users.shape
# detect duplicate zip codes: True if an item is identical to a previous item
users.zip_code.duplicated().tail()
# count the duplicate items (True becomes 1, False becomes 0)
users.zip_code.duplicated().sum()
# detect duplicate DataFrame rows: True if an entire row is identical to a previous row
users.duplicated().tail()
# count the duplicate rows
users.duplicated().sum()
# examine the duplicate rows (ignoring the first occurrence)
users.loc[users.duplicated(keep='first'), :]
# examine the duplicate rows (ignoring the last occurrence)
users.loc[users.duplicated(keep='last'), :]
# examine the duplicate rows (including all duplicates)
users.loc[users.duplicated(keep=False), :]
# drop the duplicate rows (inplace=False by default)
users.drop_duplicates(keep='first').shape
users.drop_duplicates(keep='last').shape
users.drop_duplicates(keep=False).shape
# only consider a subset of columns when identifying duplicates
users.duplicated(subset=['age', 'zip_code']).sum()
users.drop_duplicates(subset=['age', 'zip_code']).shape
# read a dataset of top-rated IMDb movies into a DataFrame
#movies = pd.read_csv('http://bit.ly/imdbratings')
movies.head()
# count the missing values in the 'content_rating' Series
movies.content_rating.isnull().sum()
# examine the DataFrame rows that contain those missing values
movies[movies.content_rating.isnull()]
# examine the unique values in the 'content_rating' Series
movies.content_rating.value_counts()
# first, locate the relevant rows
movies[movies.content_rating=='NOT RATED'].head()
# then, select the 'content_rating' Series from those rows
movies[movies.content_rating=='NOT RATED'].content_rating.head()
# finally, replace the 'NOT RATED' values with 'NaN' (imported from NumPy)
import numpy as np
movies[movies.content_rating=='NOT RATED'].content_rating = np.nan
# the 'content_rating' Series has not changed
movies.content_rating.isnull().sum()
# replace the 'NOT RATED' values with 'NaN' (does not cause a SettingWithCopyWarning)
movies.loc[movies.content_rating=='NOT RATED', 'content_rating'] = np.nan
# this time, the 'content_rating' Series has changed
movies.content_rating.isnull().sum()
# create a DataFrame only containing movies with a high 'star_rating'
top_movies = movies.loc[movies.star_rating >= 9, :]
top_movies
# overwrite the relevant cell with the correct duration
top_movies.loc[0, 'duration'] = 150
# 'top_movies' DataFrame has been updated
top_movies
# 'movies' DataFrame has not been updated
movies.head(1)
# explicitly create a copy of 'movies'
top_movies = movies.loc[movies.star_rating >= 9, :].copy()
# pandas now knows that you are updating a copy instead of a view (does not cause a SettingWithCopyWarning)
top_movies.loc[0, 'duration'] = 150
# 'top_movies' DataFrame has been updated
top_movies
# read a dataset of alcohol consumption into a DataFrame
drinks = pd.read_csv('../input/drinks.csv')
# only 60 rows will be displayed when printing
drinks
# check the current setting for the 'max_rows' option
pd.get_option('display.max_rows')
# overwrite the current setting so that all rows will be displayed
pd.set_option('display.max_rows', None)
drinks
# reset the 'max_rows' option to its default
pd.reset_option('display.max_rows')
# the 'max_columns' option is similar to 'max_rows'
pd.get_option('display.max_columns')
# read the training dataset from Kaggle's Titanic competition into a DataFrame
#train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()
# an ellipsis is displayed in the 'Name' cell of row 1 because of the 'max_colwidth' option
pd.get_option('display.max_colwidth')
# overwrite the current setting so that more characters will be displayed
pd.set_option('display.max_colwidth', 1000)
train.head()
# overwrite the 'precision' setting to display 2 digits after the decimal point of 'Fare'
pd.set_option('display.precision', 2)
train.head()
# add two meaningless columns to the drinks DataFrame
drinks['x'] = drinks.wine_servings * 1000
drinks['y'] = drinks.total_litres_of_pure_alcohol * 1000
drinks.head()
# use a Python format string to specify a comma as the thousands separator
pd.set_option('display.float_format', '{:,}'.format)
drinks.head()
# 'y' was affected (but not 'x') because the 'float_format' option only affects floats (not ints)
drinks.dtypes
# view the option descriptions (including the default and current values)
pd.describe_option()
# search for specific options by name
pd.describe_option('rows')
# reset all of the options to their default values
pd.reset_option('all')
# create a DataFrame from a dictionary (keys become column names, values become data)
pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']})
# optionally specify the order of columns and define the index
df = pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']}, columns=['id', 'color'], index=['a', 'b', 'c'])
df
# create a DataFrame from a list of lists (each inner list becomes a row)
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns=['id', 'color'])
# create a NumPy array (with shape 4 by 2) and fill it with random numbers between 0 and 1
import numpy as np
arr = np.random.rand(4, 2)
arr
# create a DataFrame from the NumPy array
pd.DataFrame(arr, columns=['one', 'two'])
# create a DataFrame of student IDs (100 through 109) and test scores (random integers between 60 and 100)
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)})
# 'set_index' can be chained with the DataFrame constructor to select an index
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)}).set_index('student')
# create a new Series using the Series constructor
s = pd.Series(['round', 'square'], index=['c', 'b'], name='shape')
s
# concatenate the DataFrame and the Series (use axis=1 to concatenate columns)
pd.concat([df, s], axis=1)
# read the training dataset from Kaggle's Titanic competition into a DataFrame
#train = pd.read_csv('http://bit.ly/kaggletrain')
train.head()
# map 'female' to 0 and 'male' to 1
train['Sex_num'] = train.Sex.map({'female':0, 'male':1})
train.loc[0:4, ['Sex', 'Sex_num']]
# calculate the length of each string in the 'Name' Series
train['Name_length'] = train.Name.apply(len)
train.loc[0:4, ['Name', 'Name_length']]
# round up each element in the 'Fare' Series to the next integer
import numpy as np
train['Fare_ceil'] = train.Fare.apply(np.ceil)
train.loc[0:4, ['Fare', 'Fare_ceil']]
# we want to extract the last name of each person
train.Name.head()
# use a string method to split the 'Name' Series at commas (returns a Series of lists)
train.Name.str.split(',').head()
# define a function that returns an element from a list based on position
def get_element(my_list, position):
    return my_list[position]
# apply the 'get_element' function and pass 'position' as a keyword argument
train.Name.str.split(',').apply(get_element, position=0).head()
# alternatively, use a lambda function
train.Name.str.split(',').apply(lambda x: x[0]).head()
# read a dataset of alcohol consumption into a DataFrame
#drinks = pd.read_csv('http://bit.ly/drinksbycountry')
drinks.head()
# select a subset of the DataFrame to work with
drinks.loc[:, 'beer_servings':'wine_servings'].head()
# apply the 'max' function along axis 0 to calculate the maximum value in each column
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)
# apply the 'max' function along axis 1 to calculate the maximum value in each row
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=1).head()
# use 'np.argmax' to calculate which column has the maximum value for each row
drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis=1).head()
# convert every DataFrame element into a float
drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float).head()
# overwrite the existing DataFrame columns
drinks.loc[:, 'beer_servings':'wine_servings'] = drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float)
drinks.head()