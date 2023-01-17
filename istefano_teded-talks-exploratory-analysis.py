import ast

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/ted-talks/ted_main.csv')

df.columns
df.head()
# Convert film date and published date to human readable format

df['film_date'] = pd.to_datetime(df['film_date'],unit='s')

df['published_date'] = pd.to_datetime(df['published_date'],unit='s')
df.head()
df['speaker_occupation'].fillna('', inplace=True)

df.info()
df.describe()
# Check for correlation between the numerical variables.

sns.heatmap(df.corr(), annot=True)
unpopular_talks = df[['title', 'main_speaker', 'published_date', 'views', 'comments', 'ratings']].sort_values('views')[:15].reset_index(drop=True)

unpopular_talks
sns.set_style("darkgrid")

plt.figure(figsize=(10,6))

sns.barplot(x=unpopular_talks.index, y='views', data=unpopular_talks)

plt.xlabel("Index")

 
sns.boxplot(y=df.comments)
# Check author occupation and talk tags to see if there are any insights on why the views are low

unpopular_talks_occupation = df[['title', 'main_speaker', 'speaker_occupation','tags', 'event', 'views']].sort_values('views')[:15].reset_index(drop=True)

unpopular_talks_occupation
unpopular_talks_occupation['tags'] = unpopular_talks_occupation['tags'].apply(lambda x: ast.literal_eval(x))

corpus = ' '

for x in unpopular_talks_occupation['tags']:

    str = ' '.join(x)

    corpus += ' ' + str
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
# convert stringified dictionary into python dictionary



unpopular_talks['eval_ratings'] = unpopular_talks['ratings'].apply(lambda x: eval(x))

counter = {'Funny':0, 'Beautiful':0, 'Ingenious':0, 'Courageous':0, 'Longwinded':0, 'Confusing':0, 'Informative':0, 'Fascinating':0, 'Unconvincing':0, 'Persuasive':0, 'Jaw-dropping':0, 'OK':0, 'Obnoxious':0, 'Inspiring':0}



for i in range(len(unpopular_talks['eval_ratings'])):

    for j in range(len(unpopular_talks['eval_ratings'][i])):

        counter[unpopular_talks['eval_ratings'][i][j]['name']] += unpopular_talks['eval_ratings'][i][j]['count']

    

frequencies = list(counter.values())

descr = counter.keys()

descriptors = [x for _,x in sorted(zip(frequencies,counter.keys()), reverse=True)]

neg_descriptors = {"Confusing", "Unconvincing", "Longwinded", "Obnoxious", "OK"}

neg_indices  = [x for x in range (len(descriptors)) if descriptors[x] in neg_descriptors]

frequencies.sort(reverse=True)

bar_colors = ['blue' if desc not in neg_descriptors else 'red' for desc in descriptors]

indices = np.arange(len(descriptors))

bar = sns.barplot(x=indices, y=frequencies, palette=bar_colors)

plt.xticks(indices, descriptors, rotation=45, ha="right")

plt.show()
neg_desc_count = sum([counter[desc] for desc in neg_descriptors])

total_desc_count = sum(frequencies)

unpopular_talks_pct_negative = 100 * (neg_desc_count / total_desc_count)

unpopular_talks_pct_negative
df_no_tedex = df[~df['event'].str.contains('TEDx')].sort_values('views')[:15].reset_index(drop=True)

unpopular_data_no_tedex = df_no_tedex[['title','event', 'main_speaker', 'published_date', 'views', 'tags', 'comments', 'ratings']]

unpopular_data_no_tedex
df_no_tedex = df[~df['event'].str.contains('TEDx') & ~df['event'].str.contains('TED-Ed')].sort_values('views')[:15].reset_index(drop=True)

unpopular_data_no_tedex = df_no_tedex[['title','event', 'main_speaker','speaker_occupation', 'film_date','published_date', 'views', 'tags', 'comments', 'ratings']]

unpopular_data_no_tedex
unpopular_data_no_tedex['year'] = unpopular_data_no_tedex['published_date'].apply(lambda x: x.year)

year_df = pd.DataFrame(unpopular_data_no_tedex['year'].value_counts().reset_index())

year_df.columns = ['year', 'talks']



plt.figure(figsize=(12,5))

_ = sns.barplot(x='year', y='talks', data=year_df)





unpopular_data_no_tedex.speaker_occupation.value_counts()
grid = sns.JointGrid(unpopular_data_no_tedex.index, unpopular_data_no_tedex.comments, space=0, size=6, ratio=50)

grid.plot_joint(plt.bar, color="y")

grid.ax_joint.plot([0,len(unpopular_data_no_tedex.comments)], [df.comments.median(), df.comments.median()], 'b-', linewidth = 2, label='Overall comments median value')

grid.ax_joint.plot([0,len(unpopular_data_no_tedex.comments)], [unpopular_data_no_tedex.comments.mean(), unpopular_data_no_tedex.comments.mean()], 'r-', linewidth = 2, label='Unpopular comments median value')

plt.xlabel("Index")

_ = plt.legend()
unpopular_data_no_tedex['tags_converted'] = unpopular_data_no_tedex['tags'].apply(lambda x: ast.literal_eval(x))

corpus = ' '

for x in unpopular_data_no_tedex['tags_converted']:

    str = ' '.join(x)

    corpus += ' ' + str
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
unpopular_data_no_tedex['eval_ratings'] = unpopular_data_no_tedex['ratings'].apply(lambda x: eval(x))

counter = {'Funny':0, 'Beautiful':0, 'Ingenious':0, 'Courageous':0, 'Longwinded':0, 'Confusing':0, 'Informative':0, 'Fascinating':0, 'Unconvincing':0, 'Persuasive':0, 'Jaw-dropping':0, 'OK':0, 'Obnoxious':0, 'Inspiring':0}



for i in range(len(unpopular_data_no_tedex['eval_ratings'])):

    for j in range(len(unpopular_data_no_tedex['eval_ratings'][i])):

        counter[unpopular_data_no_tedex['eval_ratings'][i][j]['name']] += unpopular_talks['eval_ratings'][i][j]['count']

    

frequencies = list(counter.values())

descr = counter.keys()

descriptors = [x for _,x in sorted(zip(frequencies,counter.keys()), reverse=True)]

neg_descriptors = {"Confusing", "Unconvincing", "Longwinded", "Obnoxious", "OK"}

neg_indices  = [x for x in range (len(descriptors)) if descriptors[x] in neg_descriptors]

frequencies.sort(reverse=True)

bar_colors = ['blue' if desc not in neg_descriptors else 'red' for desc in descriptors]

indices = np.arange(len(descriptors))

bar = sns.barplot(x=indices, y=frequencies, palette=bar_colors)

plt.xticks(indices, descriptors, rotation=45, ha="right")

plt.show() 
neg_desc_count = sum([counter[desc] for desc in neg_descriptors])

total_desc_count = sum(frequencies)

unpopular_data_no_tedex_pct_negative = 100 * (neg_desc_count / total_desc_count)

unpopular_data_no_tedex_pct_negative
plt.figure(figsize=(10,5))

sns.barplot(x = ['unpopular_talks_pct_negative', 'unpopular_data_no_tedex_pct_negative'], y = [unpopular_talks_pct_negative, unpopular_data_no_tedex_pct_negative])

_ = plt.ylabel("% of negative rating count")
df['year'] = df['published_date'].apply(lambda x: x.year)

year_df = pd.DataFrame(df['year'].value_counts().reset_index())

year_df.columns = ['year', 'talks']



plt.figure(figsize=(12,5))

_ = sns.barplot(x='year', y='talks', data=year_df)
views_per_year = df[['year', 'views']].groupby('year').sum().reset_index()

plt.figure(figsize=(12,5))

_ = sns.barplot(x='year', y='views', data=views_per_year)
df.describe()
def cal_neg_rating_ratio(ratings):

    counter = {'Funny':0, 'Beautiful':0, 'Ingenious':0, 'Courageous':0, 'Longwinded':0, 'Confusing':0, 'Informative':0, 'Fascinating':0, 'Unconvincing':0, 'Persuasive':0, 'Jaw-dropping':0, 'OK':0, 'Obnoxious':0, 'Inspiring':0}

    neg_descriptors = {"Confusing", "Unconvincing", "Longwinded", "Obnoxious", "OK"}

    for rating_list in ratings:

        counter[rating_list['name']] += rating_list['count']

    neg_desc_count = sum([counter[desc] for desc in neg_descriptors])

    total_desc_count = sum(list(counter.values()))

    unpopular_data_no_tedex_pct_negative = 100 * (neg_desc_count / total_desc_count)

    return unpopular_data_no_tedex_pct_negative

        

    
df['eval_ratings'] = df['ratings'].apply(lambda x: eval(x))
df['neg_rating_ratio'] = df.eval_ratings.apply(cal_neg_rating_ratio)
large_unpop_df = df[(df.views < df.views.median()) & (df.neg_rating_ratio >= 24) & (~df['event'].str.contains('TEDx') & ~df['event'].str.contains('TED-Ed'))].reset_index(drop=True)
large_unpop_df.info()
large_unpop_df.describe()
sns.boxplot(data=large_unpop_df.neg_rating_ratio)
large_unpop_df.sort_values('neg_rating_ratio', ascending=False).iloc[0].description
large_unpop_df['tags_converted'] = large_unpop_df['tags'].apply(lambda x: ast.literal_eval(x))

corpus = ' '

for x in large_unpop_df['tags_converted']:

    str = ' '.join(x)

    corpus += ' ' + str
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
year_df = pd.DataFrame(large_unpop_df['year'].value_counts().reset_index())

year_df.columns = ['year', 'talks']



plt.figure(figsize=(12,5))

_ = sns.barplot(x='year', y='talks', data=year_df)





occupation_count = large_unpop_df.speaker_occupation.value_counts()[:10]



plt.figure(figsize=(25,5))

sns.barplot(x=occupation_count.index, y=occupation_count.values)

_ = plt.ylabel('Number of Occurrences', fontsize=12)