# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Tutorial from: 
# https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68



# Load the data
df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')
df.head()

# Summarythe Data 
# Shape it (correspond to size() in Octave)
print(df.shape)

# Columns
print(df.columns)

# General info
print(df.info())
# The code above is different of "print(df.info)" - without parenthesis inside


# See that dtypes indicates data types:
# bool = logical
# object = categorical
# float and int = numeric

# with this code we can see if there are MISSING VALUES. It's not the case here because we have, in each column, 3333 entries, exactly the
# total number of rows of the table (we see that with SHAPE code)

# To CONVERT one column of one data type to another data type. Using "astype" code
df['churn'] = df['churn'].astype('int64') # Here, from bool (logical) to int64 (numerical)

# And see again
print(df.info()) # Very funny

# Using "describe" code, the "summary" function of Python
df.describe()          # Again, is different use with parenthesis and without parenthesis



# Count is the number of non-missing values
# The others you can know by reading...
# But the code above works only for numerical features. To use for non-numerical features we run:

df.describe(include=['object', 'bool'])

# Another way for non-numerical is
df['churn'].value_counts()
df['international plan'].value_counts()

# To see the proportion
df['churn'].value_counts(normalize=True)
df['international plan'].value_counts(normalize=True)
# Sort the Data
df.sort_values(by = 'total day charge', ascending=False).head()

# Sort by multiple columns
df.sort_values(by=['churn', 'total day charge'], ascending=[False, False]).head()
# Indexing and retrieving data
# Andwer the question: what is the proportion of churned users in our dataframe?
df["churn"].mean() # Since this feature was bool type (0 and 1 entries), the mean will give us the proportion of the True Answers

# Another Question:
# What are average values of numerical variables for churned users?
df[df['churn'] == 1].mean() # That is, mean of the (numerical) records in "df" that satisfy the condition (churn == 1). The mean() function
# applies only for numerical variables

# One more
# How much time (on average) do churned users spend on phone during daytime?
df[df['churn'] == 1]['total day minutes'].mean()

# That is more than the total average, shit!
df['total day minutes'].mean()
# What is the maximum length of international calls among loyal users (Churn == 0) 
# who do not have an international plan? - TWO CONDITIONS

df[(df['churn'] == 0) & (df['international plan'] == 'no')]['total intl minutes'].max()


# Other way for Indexing
df.loc[0:5, 'state':'area code']  # The loc method is used for indexing by name (ROWS don't have name...)

# Or, that returns the same:
df.iloc[0:5, 0:3]  # iloc() is used for indexing by number.

# In Python, the first (1) row/column is indexed as zero(0). And, the last indexed row/column isn't included.
# You can see in the code above, the row five is indexed, but not included in the output.
# The same for columns. The column three is indexed, but not included in the output.

# If we need the first line

df[:1] # with all columns. The number we put before the two points is the number of lines the output will have (Starting by the begin of the Table)

# or last line of the data frame
df[-1:]  # with all columns. It's like we said: "Need from the -1 row (the last row) to the last".

# Applying Functions to Cells, Columns and Rows

# The apply() function
df.apply(np.max)
# lambda function
df[df['state'].apply(lambda state: state[0] == 'W')].head()
# Map method
d = {'no' : False, 'yes' : True} 
df['international plan'] = df['international plan'].map(d) 
df
# Replace method

df = df.replace({'voice mail plan': d}) 
df.head()
# Example of Grouping
columns_to_show = ['total day minutes', 'total eve minutes', 
                   'total night minutes']
df.groupby(['churn'])[columns_to_show].describe(percentiles=[])
# Another example using agg(), passing a list of functions

columns_to_show = ['total day minutes', 'total eve minutes', 
                   'total night minutes']
df.groupby(['churn'])[columns_to_show].agg([np.mean, np.std, 
                                            np.min, np.max])
# Crosstab method
pd.crosstab(df['churn'], df['international plan'])
pd.crosstab(df['churn'], df['voice mail plan'], normalize=True)
# The normalize argument we use to see the proportions
# pivot_table method
df.pivot_table(['total day calls', 'total eve calls', 'total night calls'], ['area code'], aggfunc='mean')
# Total calls
total_calls = df['total day calls'] + df['total eve calls'] + \
              df['total night calls'] + df['total intl calls'] 
df.insert(loc=len(df.columns), column='total calls', value=total_calls) 
df.head()

# See the "\" to jump from one line to another (WTF)
# The same act (now with charges), without the intermediate object

df['total charge'] = df['total day charge'] + df['total eve charge'] + \
                     df['total night charge'] + df['total intl charge']
df.head()
# Deleting rows or columns

# get rid of just created columns 
df.drop(['total charge', 'total calls'], axis=1, inplace=True) 
# axis = 1 means delete column, axis = 0 (or default/nothing) means delete rows
# inplace argument tells if we want to change the existing DataFrame (True) or not(False)
df
# and here’s how you can delete rows 
df.drop([1, 2]).head() # means delete row 1 and 2
# axis here is default (0)
# inplace here is default (False)
# Contingency Table
pd.crosstab(df['churn'], df['international plan'], margins=True)
# The margins argument add the All column and Row
# some imports and "magic" commands to set up plotting 
%matplotlib inline                          # What is it?! 
import matplotlib.pyplot as plt 
# pip install seaborn 
import seaborn as sns
plt.rcParams['figure.figsize'] = (10, 10) # adjust the size of the Picture (nice!) - From the matplotlib package
sns.countplot(x='international plan', hue='churn', data=df) # Bar Plot (Count Plot, from the Seaborn package)
# Crossing another variables
pd.crosstab(df['churn'], df['customer service calls'], margins=True)
sns.countplot(x='customer service calls', hue='churn', data=df)
# Let's add a new feature. A binary variable = Customer service calls > 3 (why that? Because we have a important change in the value 3)
df['many_service_calls'] = (df['customer service calls'] > 3).astype('int') # Creating the feature direct, without an intermediate object
pd.crosstab(df['many_service_calls'], df['churn'], margins=True) # Making a Cross Table
sns.countplot(x='many_service_calls', hue='churn', data=df); # Making a plot
pd.crosstab(df['many_service_calls'] & df['international plan'] , df['churn'])
