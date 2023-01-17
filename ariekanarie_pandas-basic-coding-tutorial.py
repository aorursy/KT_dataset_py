# Load libraries:

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Read a data set from a CSV file:

df = pd.read_csv('../input/telecom_churn.csv')

# CSV is a simple text-based comma-separated format

# In the example the first few lines of the above file are:

#

# State,Account length,Area code,International plan,Voice mail plan,Number vmail messages,Total day minutes,Total day calls,Total day charge,Total eve minutes,Total eve calls,Total eve charge,Total night minutes,Total night calls,Total night charge,Total intl minutes,Total intl calls,Total intl charge,Customer service calls,Churn

# KS,128,415,No,Yes,25,265.1,110,45.07,197.4,99,16.78,244.7,91,11.01,10.0,3,2.7,1,False

# OH,107,415,No,Yes,26,161.6,123,27.47,195.5,103,16.62,254.4,103,11.45,13.7,3,3.7,1,False
# A quick look at the first few rows, nicely formatted:

df.head()
# Indexing by a column name gives a one-dimensional Series:

df.head()['State']
# Basic info about the dataset:

print('Number of rows, columns', df.shape)

print('Columns (features):')

print(' ',df.columns)
# Some more details

# - type of each feature

# - how many missing values for each feature?

# - memory usage

df.info()
# Get a section of rows by row number:

dfSmall = df[0:6]

dfSmall
# get one column (as a one-dimensional Series):

dfSmall['International plan']
# get one value:

dfSmall['International plan'][0]
# Selecting specified rows and columns by index:

dfSmall.loc[:4,['State','Voice mail plan']]
# Selecting specified rows and columns by index:

dfSmall.loc[:4,'State':'Voice mail plan']
# Selecting specified rows and columns by number (position)

df.iloc[[1,4,5],1:4]
# compute a simple boolean condition on rows of a DataFrame:

dfSmall['International plan']=='Yes'
# compute a more complicated boolean condition on rows of a DataFrame:

(dfSmall['International plan']=='Yes') & (dfSmall['State']=='OH')
# compute a really complicated boolean condition on rows of a DataFrame:

dfSmall.apply(lambda row : row['State'].startswith('O')

                              and row['International plan']=='Yes',

              axis=1)
dfSmall[dfSmall['International plan']=='Yes']
# Summary stats on numeric features:

df.describe()
# Summary stats on categorical features:

df.describe(include=['object','bool'])
# describe returns a DataFrame!

df.describe().loc[['mean','std'],['Account length','Total night calls']]
# individual stat functions on a DataFrame - returns a Series:

df.mean()
# individual stat functions on a Series - returns a value:

df['Account length'].mean()
# value counts for a categorical feature - returns a Series:

df['State'].value_counts()
df['State'].value_counts(normalize=True).loc[['NY','DC']]
df['State'].value_counts(normalize=True)['NY']
# Summary stats on numeric features for those with an international plan:

df[df['International plan']=='Yes'].describe()
# Churn percentage for those living in NY or NJ:

df[(df['State']=='NY') | (df['State']=='NJ')]['Churn'].value_counts(normalize=True)
# Churn percentage by state:

df.groupby(['State'])['Churn'].agg([np.mean])

# can also use multiple features in each of the above lists
# Churn percentage by state, sorted in increasing order:

df.groupby(['State'])['Churn'].agg([np.mean,len]).sort_values(by='mean')
# Another way to get churn by state:

pd.crosstab(df['Churn'],df['State'])
# And yet another - this time showing both churn and total day minutes:

df.pivot_table(['Churn','Total day minutes'],['State'],aggfunc='mean')
# Add a column in end position:

meanDayMinutes = df['Total day minutes'].mean()

df['Heavy day user'] = df['Total day minutes'] > meanDayMinutes

df.head()
# Delete a column:

df.drop('Heavy day user',axis=1,inplace=True)

# use axis=0 to delete rows

df.head()
# Add a column in a specified position:

df.insert(loc=1, column='Heavy day user', value = df['Total day minutes']>meanDayMinutes)

df.head()
df['Heavy day user'].value_counts()
# Are heavy day users more likely to churn?

df.pivot_table(['Churn'],['Heavy day user'],aggfunc=['mean',len])
# Drop rows with churn unavailable

# (will do nothing on this dataset):

df.dropna(axis=0, subset=['Churn'], inplace=True)

df.info()
# convert 'International plan' and 'Voice mail plan' to int 0/1:

df['International plan'] = (df['International plan']=='Yes').astype('int')

# Or could use map (supply a list of original -> transformed values):

# df['Voice mail plan'] = df['Voice mail plan'].map({'Yes':1,'No':0})

# Or could use apply (most general - supply a function that converts the values):

df['Voice mail plan'] = df['Voice mail plan'].apply(lambda x : int(x=='Yes'))



# convert 'Heavy day user' and 'Churn' to int 0/1:

df['Heavy day user'] = df['Heavy day user'].astype('int')

df['Churn'] = df['Churn'].astype('int')

df.head()
# Convert DataFrame column to numpy array (ex. for use by ML algorithms):

df['Churn'].values
# Convert DataFrame to numpy matrix (ex. for use by ML algorithms):

df.drop(['State','Churn'],axis=1).values