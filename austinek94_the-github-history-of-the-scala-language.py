# Importing pandas

import pandas as pd



# Loading in the data

pulls_one = pd.read_csv('../input/datacamp-projects/The GitHub History of the Scala Language/datasets/pulls_2011-2013.csv')

pulls_two = pd.read_csv('../input/datacamp-projects/The GitHub History of the Scala Language/datasets/pulls_2014-2018.csv')

pull_files =pd.read_csv('../input/datacamp-projects/The GitHub History of the Scala Language/datasets/pull_files.csv')

# Append pulls_one to pulls_two

pulls = pulls_one.append(pulls_two,ignore_index=True)



# Convert the date for the pulls object

pulls['date'] = pd.to_datetime(pulls['date'],utc=True)
# Merge the two DataFrames

data = pulls.merge(pull_files,on='pid')

print(data)
%matplotlib inline



# Create a column that will store the month

data['month'] = data['date'].dt.month



# Create a column that will store the year

data['year'] = data['date'].dt.year



# Group by the month and year and count the pull requests

counts = data.groupby(['month','year'])['pid'].count()



# Plot the results

counts.plot(kind='bar', figsize = (12,4))
# Required for matplotlib

%matplotlib inline

import matplotlib.pyplot as plt



# Group by the submitter

by_user =data.groupby('user').agg({

    'pid':'max'

    

})

# Plot the histogram

by_user.hist(bins=10)

plt.xlabel('Number of Contributions')

plt.ylabel('Number of Contributor')

plt.title('Is the project welcoming to the new Contributors ?')
# Identify the last 10 pull requests

last_10 = pulls.nlargest(10,'date',keep='last')



# Join the two data sets

joined_pr = last_10.merge(pull_files,on='pid')



# Identify the unique files

files = set(joined_pr['file'])



# Print the results

files
# This is the file we are interested in:

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Identify the commits that changed the file

file_pr = data[data['file']==file]





# Count the number of changes made by each developer

author_counts = file_pr.groupby('user').count()

                                           

#Print the top 3 developers

print(list(author_counts.nlargest(3, 'pid').index))

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Select the pull requests that changed the target file

file_pr = data[data['file']==file]



# Merge the obtained results with the pulls DataFrame

joined_pr = file_pr.merge(pulls,on='pid')



# Find the users of the last 10 most recent pull requests

users_last_10 = set(joined_pr.nlargest(10,'date_x')['user_x'])



# Printing the results

users_last_10
%matplotlib inline

import matplotlib.pyplot as plt



# The developers we are interested in

authors = ['xeno-by', 'soc']



# Get all the developers' pull requests

by_author = pulls[pulls['user'].isin(authors)]



# Count the number of pull requests submitted each year

counts = by_author.groupby(['user',pulls['date'].dt.year]).agg({'pid': 'count'}).reset_index()



# Convert the table to a wide format

counts_wide = counts.pivot_table(index='date', columns='user', values='pid', fill_value=0)



# Plot the results

counts_wide.plot(kind="bar", title="Number of Contributions per Year")

plt.xlabel('Year')

plt.ylabel('Number of Pull Requests')
authors = ['xeno-by', 'soc']

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Select the pull requests submitted by the authors, from the `data` DataFrame

by_author = data[data['user'].isin(authors)]



# Select the pull requests that affect the file

by_file =  data[data['file'] == file]



# Group and count the number of PRs done by each user each year

grouped = by_author.groupby(['user', by_file['date'].dt.year]).count()['pid'].reset_index()



# Transform the data into a wide format

by_file_wide = grouped.pivot_table(index='date', columns='user', values='pid', fill_value=0)



# Plot the results

by_file_wide.plot(kind='bar')