# Printing the content of git_log_excerpt.csv
with open("../input/github-logs/git_log_excerpt.csv") as f:
    print(f.read())
# Loading in the pandas module as 'pd'
import pandas as pd

# Reading in the log file
git_log = pd.read_csv("../input/github-logs/git.log",sep="#",encoding='latin-1',header=None,names=['timestamp', 'author'])
# Printing out the first 5 rows
git_log.head()
# calculating number of commits
number_of_commits = len(git_log)

# calculating number of authors
number_of_authors = len(git_log['author'].dropna().unique())

# printing out the results
print("%s authors committed %s code changes." % (number_of_authors, number_of_commits))
# Identifying the top 10 authors
top_10_authors = git_log['author'].value_counts().head(10)

# Listing contents of 'top_10_authors'
top_10_authors.head(10)
# converting the timestamp column
git_log['timestamp'] = pd.to_datetime(git_log['timestamp'], unit="s")

# summarizing the converted timestamp column
git_log['timestamp'].describe()
# determining the first real commit timestamp
first_commit_timestamp = git_log.iloc[-1]['timestamp']

# determining the last sensible commit timestamp
last_commit_timestamp = pd.to_datetime('2018')

# filtering out wrong timestamps
corrected_log = git_log[
    (git_log['timestamp'] >= first_commit_timestamp) &
    (git_log['timestamp'] <= last_commit_timestamp)]

# summarizing the corrected timestamp column
corrected_log['timestamp'].describe()
# Counting the no. commits per year
commits_per_year = corrected_log.groupby(
    pd.Grouper(key='timestamp', freq='AS')).count()

# Listing the first rows
commits_per_year.head()
# Counting the no. commits per year
commits_per_year = corrected_log.groupby(
    pd.Grouper(key='timestamp', freq='AS')).count()

# Listing the first rows
commits_per_year.head()
# calculating or setting the year with the most commits to Linux
year_with_most_commits = 2016 