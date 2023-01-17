import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def clean_ts(df):

    return df[(df['author_timestamp'] > 1104600000) & (df['author_timestamp'] < 1487807212)]

df = clean_ts(pd.read_csv('../input/linux_kernel_git_revlog.csv'))

df['author_dt'] = pd.to_datetime(df['author_timestamp'],unit='s')

df.head()
time_df = df.groupby(['author_timestamp', 'author_dt'])[['n_additions', 'n_deletions']].agg(np.sum).reset_index().sort_values('author_timestamp', ascending=True)

time_df['diff'] = time_df['n_additions'] - time_df['n_deletions']

time_df.head()
t = pd.Series(time_df['diff'].values, index=time_df['author_dt'])

t.plot(title='lines of code added', figsize=(12,8))
commits_over_time = df.groupby('author_dt')['commit_hash'].nunique().reset_index().sort_values('author_dt', ascending=True)

commits_series = pd.Series(commits_over_time['commit_hash'].values, index=commits_over_time['author_dt'])

commits_series.plot(title='number of commits on original time series', figsize=(12,8))
commits_series.resample('M').mean().plot(title='number of commits on monthly resampled data', figsize=(12,8))
files_changed_per_commit = df.groupby(['author_dt', 'commit_hash'])['filename'].agg('count').reset_index().sort_values('author_dt', ascending=True)

files_changed_per_commit = pd.Series(files_changed_per_commit['filename'].values, index=files_changed_per_commit['author_dt'])

files_changed_per_commit.plot(title='number files changed per commit', figsize=(12,8))
# trim distribution, there are a few heavy outliers in the data as we saw above 

n_files_changed_per_commit = df.groupby('commit_hash')['filename'].agg('count')

n_files_changed_per_commit = n_files_changed_per_commit[n_files_changed_per_commit < 20]

sns.distplot(n_files_changed_per_commit, kde=False)

plt.title('distribution of number of files changed per commit')

plt.xlabel('number of changed files')
# trim distribution, there are a few heavy outliers in the data as we saw above 

additions_per_commit = df.groupby('commit_hash')['n_additions'].agg(np.sum)

additions_per_commit = additions_per_commit[additions_per_commit < 100]

sns.distplot(additions_per_commit)
from sklearn.feature_extraction.text import HashingVectorizer

# we will consider each unique subject

unique_subjects = np.sort(df['subject'].unique())

print(unique_subjects)

print(unique_subjects.shape)



# now vectorize each subject

hashed_subjects = HashingVectorizer(n_features=1024).fit_transform(unique_subjects)

hashed_subjects
n_additions_per_subject = df.groupby('subject')['n_additions'].agg(np.sum).reset_index().sort_values('subject')



def bucketize(row):

    if row.n_additions > 80:

        return 'XXL'

    elif row.n_additions <= 80 and row.n_additions > 60:

        return 'XL'

    elif row.n_additions <= 60 and row.n_additions > 40:

        return 'L'

    elif row.n_additions <= 40 and row.n_additions > 20:

        return 'M'

    elif row.n_additions < 20:

        return 'S'



#y = n_additions_per_subject.apply(bucketize, axis=1)



#X = hashed_subjects

#X.shape, y.shape
files_changed_per_utc_offset = df.groupby('commit_utc_offset_hours')['filename'].agg('count').reset_index().sort_values('filename', ascending=False)

sns.barplot(x = 'commit_utc_offset_hours', y = 'filename', data = files_changed_per_utc_offset)
n_authors_by_offset = df.groupby('commit_utc_offset_hours')['author_id'].nunique().reset_index().sort_values('author_id', ascending=False)

sns.barplot(x = 'commit_utc_offset_hours', y = 'author_id', data = n_authors_by_offset)
from collections import Counter

import operator



n_rows = 1e4

subject_words = []

for row_number, row in df.ix[0:n_rows].iterrows():

    ws = row['subject'].split(" ")

    subject_words = subject_words + [w.lower() for w in ws]



words = []

counts = []

for word, count in sorted(Counter(subject_words).items(), key=operator.itemgetter(1), reverse=True):

    words.append(word)

    counts.append(count)
wcdf = pd.DataFrame({'word': words, 'count': counts})

sns.barplot(y = 'word', x = 'count', data = wcdf[0:20])
from wordcloud import WordCloud



wordcloud = WordCloud().generate(" ".join(subject_words))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis("off")
df['subject_char_len'] = df['subject'].str.len()
df.groupby('commit_utc_offset_hours')['subject_char_len'].agg(np.mean).plot()
df['commit_activity'] = df['n_additions'] + df['n_deletions']

cmap = plt.get_cmap('viridis')

sns.heatmap(df[['commit_utc_offset_hours', 'commit_activity', 'subject_char_len']].corr(), cmap=cmap)
#sns.pairplot(df[['commit_utc_offset_hours', 'commit_activity', 'subject_char_len']])
sns.distplot(list(map(lambda w: len(w), words)))
n_rows = 1e4

word_lengths = []

timezones = []



for row_number, row in df.ix[0:n_rows].iterrows():

    ws = row['subject'].split(" ")

    word_lengths = word_lengths + list(map(lambda w: len(w), ws))

    timezones = timezones + [row['commit_utc_offset_hours'] for x in range(len(ws))]



tz_ws = pd.DataFrame({'tz': timezones, 'word_length': word_lengths})

tz_ws.head(5)
tz_ws.groupby('tz')['word_length'].agg(np.mean).plot()
len(np.unique(subject_words))
from stop_words import get_stop_words



stop_words = get_stop_words('english')



filtered_subject_words = [w for w in subject_words if w not in stop_words]



len(np.unique(filtered_subject_words))

words = []

counts = []

for word, count in sorted(Counter(subject_words).items(), key=operator.itemgetter(1), reverse=True):

    if word in get_stop_words('english'):

        continue

    words.append(word)

    counts.append(count)



wcdf = pd.DataFrame({'word': words, 'count': counts})

sns.barplot(y = 'word', x = 'count', data = wcdf[0:20])