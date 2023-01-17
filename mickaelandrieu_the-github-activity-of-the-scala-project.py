# Importing pandas

import pandas as pd



# Loading in the data

pulls = pd.read_csv('../input/pulls.csv')

pull_files = pd.read_csv('../input/pull_files.csv')
# Convert the date for the pulls object

import pandas as pd

pulls['date'] = pd.to_datetime(pulls['date'], utc=True)
# Merge the two DataFrames

data = pd.merge(pulls, pull_files, on='pid')
%matplotlib inline



# Create a column that will store the month and the year, as a string

pulls['month_year'] = pulls.apply(lambda x: str(x['date'].month) + '-' + str(x['date'].year), axis='columns')



# Group by month_year and count the pull requests

counts = pulls.groupby('month_year').count()

counts['pull requests'] = counts['date'][:]

del(counts['date'], counts['user'], counts['pid'])

# Plot the results

counts.plot(kind='bar',

               title="Number of pull requests per month on the Scala project",

               rot=45)
# Required for matplotlib

%matplotlib inline

import matplotlib.pyplot as plt





# Group by the submitter

by_user = pulls.groupby('user').count()



by_user = by_user.sort_values('pid')

del(by_user['date'], by_user['month_year'])



# Plot the histogram

by_user.hist(bins=10)

plt.xlabel('Number of Contributions')

plt.ylabel('Number of Contributor')

plt.title('Is the project welcoming to the new Contributors ?')
# Identify the last 10 pull requests

last_10 = pulls.sort_values('date', ascending=False)[:10]



# Join the two data sets

joined_pr = pd.merge(last_10, pull_files, on="pid")



# Identify the unique files

files = set(joined_pr['file'].unique())



# Print the results

print(files)
# This is the file we are interested in:

file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Identify the commits that changed the file

file_pr = pull_files[pull_files['file'] == file]



# Count the number of changes made by each developer

file_pr = pd.merge(file_pr, pulls[['pid','user']], on='pid')





author_counts = file_pr.groupby('user').count()





# Print the top 3 developers

print(list(author_counts.nlargest(3, 'pid').index))
file = 'src/compiler/scala/reflect/reify/phases/Calculate.scala'



# Select the pull requests that changed the target file

file_pr = pull_files[pull_files['file'] == file]



# Merge the obtained results with the pulls DataFrame

joined_pr = pd.merge(file_pr, pulls, on='pid')



# Find the users of the last 10 most recent pull requests

users_last_10 = set(joined_pr.nlargest(10, 'date')['user'])



# Printing the results

users_last_10
%matplotlib inline



import matplotlib.pyplot as plt



# The developers we are interested in

authors = ['xeno-by', 'soc']



# Get all the developers' pull requests

by_author = pulls[pulls['user'].isin(authors)]



# Count the number of pull requests submitted each year

counts = by_author.groupby(['user', pulls['date'].dt.year]).agg({'pid': 'count'}).reset_index()



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

by_file = data[data['file'] == file]



# Group and count the number of PRs done by each user each year

grouped = by_file.groupby(['user', by_file['date'].dt.year]).count()['pid'].reset_index()



# Transform the data into a wide format

by_file_wide = counts.pivot_table(index='date', columns='user', values='pid', fill_value=0)



# Plot the results

by_file_wide.plot(kind='bar')