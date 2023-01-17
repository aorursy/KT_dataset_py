!pip install joypy -q



import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from matplotlib import cm

import joypy

import seaborn as sns

plt.rcParams["figure.figsize"] = (16,8)



from scipy.stats import pearsonr

from scipy.stats import spearmanr
df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')
df.head()
print('The raw dataset has {} rows and {} columns.'.format(df.shape[0],df.shape[1]))

print('-----------------')

print('Columns with NaN values: ')

nan_cols = df.isna().any()[df.isna().any() == True]

if len(nan_cols)>0:

    print(nan_cols) 

else:

    print('none')

print('-----------------')

print('Columns with null values: ')

null_cols = df.isnull().any()[df.isnull().any() == True]

if len(null_cols)>0:

    print(null_cols)

else:

    print('none')
orig_rows = df.shape[0]

df.drop_duplicates(inplace=True)

print('After removing duplicate rows, the dataset has {} rows remaining. {} duplicate rows were removed.'.format(df.shape[0], orig_rows - df.shape[0]))
ax = sns.barplot(df.subject.value_counts().index, df.subject.value_counts().values)

ax.set(title='Number of Courses by Subject', xlabel='Subject', ylabel='Number of Courses')

plt.show()
print('The following courses have duplicate titles:')

for item in df.course_title.value_counts()[df.course_title.value_counts() > 1].index: print('\t' + item) 
df.loc[df.course_title == 'Acoustic Blues Guitar Lessons']
df.loc[df.course_title == 'Creating an animated greeting card via Google Slides']
print('There are {} paid courses and {} free courses.'.format(df[df.is_paid == True].shape[0], df[df.is_paid == False].shape[0]))
subject_names = [x for x in df.subject.unique()]

total_courses = [x for x in df.subject.value_counts().values]

paid_courses = [x for x in df[df.is_paid == True].subject.value_counts().values]

free_courses = [x for x in df[df.is_paid == False].subject.value_counts().values]

count_values = np.array([total_courses, paid_courses, free_courses])
pay_prop = pd.DataFrame(count_values, columns = df.subject.value_counts().index.to_list())

pay_prop['course_breakdown'] = ['All Courses', 'Paid Courses', 'Free Courses']

pay_prop.set_index('course_breakdown', inplace=True)

ax = pay_prop.plot(kind='barh', 

                   stacked=True,

                   title='Course Proportion Breakdown (Free, Paid, and Total)')

ax.set_xlabel('Number of Courses')

ax.set_ylabel('')

plt.show()
ax = sns.boxplot(y=df.price, orient='h', width=0.2)

ax.set(xlabel='Course Price', title='Distribution of Course Prices (All Courses)')

plt.show()
price_stats = df.price.describe()



print('Course prices range from ${:.2f} to ${:.2f}.'.format(round(price_stats['min'], 2), round(price_stats['max'], 2)))

print('The mean course price is ${:.2f}, and the standard deviation is ${:.2f}.'.format(round(price_stats['mean'], 2), round(price_stats['std'], 2)))

print('The median coure price is ${:.2f}.'.format(round(price_stats['50%'], 2)))

print('The middle 50% of course prices are between ${:.2f} and ${:.2f}.'.format(round(price_stats['25%'], 2), round(price_stats['75%'], 2)))
ax = sns.distplot(df.price)

ax.set(title='Distribution of Course Prices (All Courses)', xlabel='Course Price')

plt.show()
ax = sns.boxplot(x=df.subject, y=df.price)

ax.set(title='Course Price Distribution by Subject', xlabel='Subject', ylabel='Course Price')

plt.show()
price_summary = df.groupby('subject').describe().price.reset_index(drop=False)

price_dict = df.price.describe().to_dict()

price_dict['subject'] = 'ALL COURSES'

price_summary.append(price_dict, ignore_index=True)



price_summary
fig, ax = joypy.joyplot(df, by='subject', column='price', figsize=(12, 6), title='Distribution of Course Prices by Subject')

plt.xlabel('Course Price')

plt.show()
top_10_by_sub = df.num_subscribers.groupby(df.subject).nlargest(10).reset_index(drop=False)

top_10_by_sub['course_title'] = top_10_by_sub.level_1.apply(lambda x: df.iloc[x].course_title)

top_10_by_sub.drop('level_1', axis=1, inplace=True)



fig, axs = plt.subplots(4, 1, figsize=(14,16))

plt.subplots_adjust(hspace=0.6)



sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[0]], x='num_subscribers', y='course_title', ax=axs[0], color='b')

sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[1]], x='num_subscribers', y='course_title', ax=axs[1], color='g')

sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[2]], x='num_subscribers', y='course_title', ax=axs[2], color='r')

sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[3]], x='num_subscribers', y='course_title', ax=axs[3], color='orange')



for i in range(len(axs)):

    axs[i].set(ylabel='', xlabel='Number of Subscribers', title='Top 10 {} Courses Based on Number of Subscribers'.format(subject_names[i]), xlim=(0,300000))
top_10_by_rev = df.num_reviews.groupby(df.subject).nlargest(10).reset_index(drop=False)

top_10_by_rev['course_title'] = top_10_by_rev.level_1.apply(lambda x: df.iloc[x].course_title)

top_10_by_rev.drop('level_1', axis=1, inplace=True)



fig, axs = plt.subplots(4, 1, figsize=(14,16))

plt.subplots_adjust(hspace=0.6)



sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[0]], x='num_reviews', y='course_title', ax=axs[0], color='b')

sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[1]], x='num_reviews', y='course_title', ax=axs[1], color='g')

sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[2]], x='num_reviews', y='course_title', ax=axs[2], color='r')

sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[3]], x='num_reviews', y='course_title', ax=axs[3], color='orange')



for i in range(len(axs)):

    axs[i].set(ylabel='', xlabel='Number of Reviews', title='Top 10 {} Courses Based on Number of Reviews'.format(subject_names[i]), xlim=(0,30000))
df['engagement'] = df['num_reviews'] / df['num_subscribers']

df.fillna(0, inplace=True)

df.reset_index(drop=True, inplace=True)



top_10_by_eng = df.loc[df.num_subscribers > 50].engagement.groupby(df.subject).nlargest(10).reset_index(drop=False)

top_10_by_eng['course_title'] = top_10_by_eng.level_1.apply(lambda x: df.iloc[x].course_title)

top_10_by_eng.drop('level_1', axis=1, inplace=True)



fig, axs = plt.subplots(4, 1, figsize=(14,16))

plt.subplots_adjust(hspace=0.6)



sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[0]], x='engagement', y='course_title', ax=axs[0], color='b')

sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[1]], x='engagement', y='course_title', ax=axs[1], color='g')

sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[2]], x='engagement', y='course_title', ax=axs[2], color='r')

sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[3]], x='engagement', y='course_title', ax=axs[3], color='orange')



for i in range(len(axs)):

    axs[i].set(ylabel='', xlabel='Number of Subscribers', title='Top 10 {} Courses Based on Engagement'.format(subject_names[i]), xlim=(0,0.4))
paid_sub = df.groupby(['subject', 'is_paid']).agg('mean')['num_subscribers']

paid_sub = pd.DataFrame(paid_sub)

paid_sub.reset_index(drop=False, inplace=True)



g = sns.FacetGrid(paid_sub, col='subject', height=6, aspect=0.75)

g = g.map(sns.barplot, 'is_paid', 'num_subscribers')
paid_rev = df.groupby(['subject', 'is_paid']).agg('mean')['num_reviews']

paid_rev = pd.DataFrame(paid_rev)

paid_rev.reset_index(drop=False, inplace=True)



g = sns.FacetGrid(paid_rev, col='subject', height=6, aspect=0.75)

g = g.map(sns.barplot, 'is_paid', 'num_reviews')
paid_eng = df.groupby(['subject', 'is_paid']).agg('mean')['engagement']

paid_eng = pd.DataFrame(paid_eng)

paid_eng.reset_index(drop=False, inplace=True)



g = sns.FacetGrid(paid_eng, col='subject', height=6, aspect=0.75)

g = g.map(sns.barplot, 'is_paid', 'engagement')
# NOTE: We are removing two outliers that contain > 150,000 subscribers

ax = sns.regplot(data=df.loc[df.num_subscribers < 150000], x='num_subscribers', y='num_reviews')

ax.set(title='Number of Reviews vs. Number of Subscribers', xlabel='Number of Subscribers', ylabel='Number of Reviews')

plt.show()
from scipy.stats import pearsonr

from scipy.stats import spearmanr



sub_rev_pearson = pearsonr(df.loc[df.num_subscribers < 150000].num_subscribers, df.loc[df.num_subscribers < 150000].num_reviews)[0]

print("Pearson's correlation between number of subscribers and number of reviews (outlers removed): {}".format(sub_rev_pearson))

print('------------------')

sub_rev_spearman = spearmanr(df.loc[df.num_subscribers < 150000].num_subscribers, df.loc[df.num_subscribers < 150000].num_reviews)[0]

print("Spearman's correlation between number of subscribers and number of reviews (outlers removed): {}".format(sub_rev_spearman))

ax = sns.regplot(data=df, x='num_lectures', y='content_duration')

ax.set(title='Number of Lectures vs. Content Duration', xlabel='Number of Lectures', ylabel='Content Duration (hrs)')

plt.show()
sub_rev_pearson = pearsonr(df.num_lectures, df.content_duration)[0]

print("Pearson's correlation between number of lectures and content duration: {}".format(sub_rev_pearson))

print('------------------')

sub_rev_spearman = spearmanr(df.num_lectures, df.content_duration)[0]

print("Spearman's correlation between number of lectures and content duration: {}".format(sub_rev_spearman))
ax = sns.countplot(df.level)

ax.set(title='Number of Courses by Level (All Courses)', xlabel='Level', ylabel='Number of Courses')

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(16,10))

plt.subplots_adjust(hspace=0.4)



sns.countplot(df.loc[df.subject == subject_names[0]].level, ax=axs[0][0], color='b')

sns.countplot(df.loc[df.subject == subject_names[1]].level, ax=axs[0][1], color='g')

sns.countplot(df.loc[df.subject == subject_names[2]].level, ax=axs[1][0], color='r')

sns.countplot(df.loc[df.subject == subject_names[3]].level, ax=axs[1][1], color='orange')



axs[0][0].set(title='Number of {} Courses by Level'.format(subject_names[0]), ylabel='Number of Courses', xlabel='Course Level')

axs[0][1].set(title='Number of {} Courses by Level'.format(subject_names[1]), ylabel='Number of Courses', xlabel='Course Level')

axs[1][0].set(title='Number of {} Courses by Level'.format(subject_names[2]), ylabel='Number of Courses', xlabel='Course Level')

axs[1][1].set(title='Number of {} Courses by Level'.format(subject_names[3]), ylabel='Number of Courses', xlabel='Course Level')

plt.show()
df.content_duration.describe()
fig, ax = plt.subplots(figsize=(20,8))

ax = sns.boxplot(y=df.content_duration, orient='h', width=0.2)

ax.set(title='Content Duration Distribution', xlabel='Content Duration (hrs)')

plt.show()
df.published_timestamp = pd.to_datetime(df.published_timestamp)
df['year_published'] = df.published_timestamp.apply(lambda x: x.year)

df['month_published'] = df.published_timestamp.apply(lambda x: x.month)

df['day_of_week_published'] = df.published_timestamp.apply(lambda x: x.dayofweek)
year_group = df.groupby('year_published').agg('count').course_id

month_group = df.groupby('month_published').agg('count').course_id

day_of_week_group = df.groupby('day_of_week_published').agg('count').course_id
ax = sns.barplot(x=year_group.index, y=year_group.values)

ax.set(title='Number of Courses Added by Year', xlabel='Year', ylabel='Number of Courses')

plt.show()
ax = sns.barplot(x=month_group.index, y=month_group.values)

ax.set(title='Number of Courses Added by Month', xlabel='Month', ylabel='Number of Courses')

plt.show()
day_of_week_group.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

ax = sns.barplot(x=day_of_week_group.index, y=day_of_week_group.values)

ax.set(title='Number of Courses Added by Day of Week', xlabel='Day of Week', ylabel='Number of Courses')

plt.show()
df[df.course_title.str.contains('Urdu')]
!pip install langdetect
from langdetect import detect



df['language'] = df.course_title.apply(lambda x: detect(x))
df.language.value_counts()
misclass_lang_en = ['sw', 'vi', 'hr', 'et', 'id', 'sv', 'da', 'ro', 'af', 'nl', 'tl', 'no', 'ca']



df['language'] = df.language.apply(lambda x: 'en' if x in misclass_lang_en else x)

df.loc[df.course_title.str.contains('Urdu'), 'language'] = 'ur'
ax = sns.barplot(df.language.value_counts().values[1:6], df.language.value_counts().index[1:6])

ax.set(title='Top 5 Non-English Languages Represented in Course Titles', xlabel='Approx. Number of Courses', ylabel='ISO 639-1 Language Code')

plt.show()
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords



eng_stop_words = stopwords.words('english')

spa_stop_words = stopwords.words('spanish')



from nltk.tokenize import word_tokenize
words = [w.lower() for w in word_tokenize(" ".join(df.course_title.values.tolist()))]

words_nostop = [w for w in words if (w.isalpha()) and ((w not in eng_stop_words) and (w not in spa_stop_words))]
common_counter = Counter(words_nostop)

unigram_counts = common_counter.most_common()

top_unigrams = [x[0] for x in unigram_counts]

top_unigram_counts = [x[1] for x in unigram_counts]



ax = sns.barplot(top_unigrams[:10], top_unigram_counts[:10])

ax.set(title='Top Unigrams in Course Titles (All Courses)', xlabel='Unigram', ylabel='Count')

plt.show()