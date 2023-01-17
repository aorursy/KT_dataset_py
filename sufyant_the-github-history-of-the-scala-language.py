# Importing pandas

import pandas as pd

# Loading in the data

pulls_one = pd.read_csv('../input/scala-language/pulls_2011-2013.csv')

pulls_two = pd.read_csv('../input/scala-language/pulls_2014-2018.csv')

pull_files = pd.read_csv('../input/scala-language/pull_files.csv') 
# Append pulls_one to pulls_two

pulls = pulls_one.append(pulls_two)

# Convert the date for the pulls object

pulls['date'] = pd.to_datetime(pulls.date,utc=True)

# Merge the two DataFrames

data = pd.merge(pulls,pull_files,on='pid')
%matplotlib inline



# Create a column that will store the month

data['month'] = data['date'].dt.month



# Create a column that will store the year

data['year'] = data['date'].dt.year



# Group by the month and year and count the pull requests

counts = data.groupby(['month','year']).count()



# Plot the results

counts.plot(kind='bar', figsize = (12,4))
data.head()
# Required for matplotlib

%matplotlib inline



# Group by the submitter

by_user = data.groupby('user').count()



# Plot the histogram

by_user.plot(kind='hist', figsize = (12,4))
# Identify the last 10 pull requests

last_10 = pulls.nlargest(10, 'date')



# Join the two data sets

joined_pr = pull_files.merge(last_10, on='pid')



# Identify the unique files

files = set(joined_pr['file'].unique())



# Print the results

files
# This is the file we are interested in:

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Identify the commits that changed the file

file_pr = data[data['file'] == file]



# Count the number of changes made by each developer

author_counts = file_pr.groupby('user').count()



# Print the top 3 developers

print(author_counts.nlargest(3, 'pid'))
file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Select the pull requests that changed the target file

file_pr = pull_files[pull_files['file'] == file]



# Merge the obtained results with the pulls DataFrame

joined_pr = file_pr.merge(pulls, on='pid')



# Find the users of the last 10 most recent pull requests

users_last_10 = set(joined_pr.nlargest(10, 'date')['user'])



# Printing the results

users_last_10
%matplotlib inline



# The developers we are interested in

authors = ['xeno-by', 'soc']



# Get all the developers' pull requests

by_author = pulls[pulls['user'].isin(authors)]



# Count the number of pull requests submitted each year

counts = by_author.groupby(['user', by_author['date'].dt.year]).agg({'pid': 'count'}).reset_index()



# Convert the table to a wide format

counts_wide = counts.pivot_table(index='date', columns='user', values='pid', fill_value=0)



# Plot the results

ax = counts_wide.plot(kind='bar')

ax.set_title('Num of pull requests by user')

ax.set_xlabel('Year')

ax.set_ylabel('Pull requests')
authors = ['xeno-by', 'soc']

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Select the pull requests submitted by the authors, from the `data` DataFrame

by_author = data[data['user'].isin(authors)]



# Select the pull requests that affect the file

by_file = by_author[by_author['file'] == file]



# Group and count the number of PRs done by each user each year

grouped = by_file.groupby(['user', by_file['date'].dt.year]).count()['pid'].reset_index()



# Transform the data into a wide format

by_file_wide = grouped.pivot_table(index = 'date',columns = 'user',values = 'pid', fill_value = 0)



# Plot the results

by_file_wide.plot(kind='bar')