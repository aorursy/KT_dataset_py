import numpy as np

import pandas as pd

import seaborn as sns

from PIL import Image

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
pd.options.display.max_columns = 100

pd.options.display.max_rows = 100

pd.options.display.width=100
df = pd.read_csv('/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv')

df.head()
df.dtypes
df.shape
df.isna().sum()
df.author_flair_text.nunique()
df.removed_by.nunique()
df.removed_by.unique()
df.total_awards_received.nunique()
df.total_awards_received.unique()
df.awarders.nunique()
df.awarders.unique()
df['total_awards_received'] = df['total_awards_received'].fillna(0)

df['total_awards_received'] = df['total_awards_received'].astype(int)
df['removed_by'] = df['removed_by'].fillna('not removed')

df['author_flair_text'] = df['author_flair_text'].fillna('no flair')
df = df[~df.title.isna()]
df.drop(columns=['id', 'awarders', 'full_link'], inplace=True)

df.head()
df['title_len'] = df['title'].astype(str).apply(len)
df['removed'] = df['removed_by'].apply(lambda val: False if val=='not removed' else True  )
df.describe()
sns.distplot(df['created_utc']) 

plt.title('Post submission in Time[ Coordinated Universal Time]', 

          fontdict={'verticalalignment': 'bottom', 'fontsize':16, 'horizontalalignment': 'center'})

plt.xlabel('Time', fontsize=12)

plt.ylabel('Probability Density', fontsize=12)

plt.show()
sns.distplot(df['title_len'])

plt.title('Title length', fontsize=16)

plt.xlabel('Length', fontsize=12)

plt.ylabel('Probability Density', fontsize=12)

plt.show()
temp = df['over_18'].value_counts()

temp.plot(kind='bar')

plt.title('NSFW Post', fontsize=16)

plt.ylabel('Number of posts', fontsize=12)

plt.xticks(rotation=0)

for i, v in enumerate(temp):

    plt.text(i-0.1, v+3000, str(v))

plt.show()

plt.show()
temp = df['removed'].value_counts()

temp.plot(kind='bar')

plt.title('Post Removed', fontsize=16)

plt.xticks(rotation=0)

plt.ylabel('Number of Posts', fontsize=12)

for i, v in enumerate(temp):

    plt.text(i-0.1, v+3000, str(v))

plt.show()
df['author'].value_counts()[1:6].plot(kind='bar')

plt.title('Top 5 Author', fontsize=16)

plt.xlabel('User Name', fontsize=12)

plt.ylabel('Number of posts', fontsize=12)

plt.xticks(rotation=0)

plt.show()
sns.distplot(df['num_comments'], hist=False)

plt.title('Number of comments', fontsize=16)

plt.xlabel('')

plt.show()
s = df[df['removed']]['removed_by'].value_counts()

plt.stem(s.values, use_line_collection=True, markerfmt='D')

plt.xticks(range(0,5), s.index, rotation=0)

plt.title('Post removed by', fontsize=16)

plt.xlabel('Remover', fontsize=12)

plt.ylabel('Number of posts', fontsize=12)

plt.show()
stop_words = set(['NOC', 'Chart', 'Bar', 'Created', 'Confirmed', 'How', 'can', 'the'])

stopwords = STOPWORDS.update(stop_words)

plt.figure(figsize=(20, 20))

wc_1 = WordCloud(width=1600, height=800, background_color='black', stopwords=STOPWORDS, 

                 max_words=1000, min_word_length=3, min_font_size=3).generate(str(' '.join(df.title.values)))



plt.imshow(wc_1, interpolation='bilinear')

plt.axis('off')

plt.title('Most frequent words in Titles', fontsize=40, verticalalignment='bottom')

plt.tight_layout(pad=0)

plt.show()
plt.figure(figsize=(20, 20))

wc_2 = WordCloud(width=1600, height=800, background_color='black', max_words=1000, 

                 stopwords=stopwords, 

                 min_word_length=3).generate(str(' '.join(df[df['over_18']].title.values)))

plt.imshow(wc_2, interpolation='bilinear')

plt.axis('off')

plt.title('Most frequent words in NSFW Titles', fontsize=40, verticalalignment='bottom')

plt.tight_layout(pad=0)

plt.show()
plt.figure(figsize=(20, 20))

wc_3 = WordCloud(width=1600, height=800, background_color='black', 

                 stopwords=stopwords, min_word_length=3, 

                 max_words=1000).generate(str(' '.join(df[df['removed']].title.values)))

plt.imshow(wc_3, interpolation='bilinear')

plt.axis('off')

plt.title('Most Frequent words in Removed posts', fontsize=40, verticalalignment='bottom')

plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 20))

wc_3 = WordCloud(width=600, height=400, background_color='black',

                 stopwords=stopwords, 

                 min_word_length=3).generate(str(' '.join(df['title'][df['removed'] & df['over_18']])))

plt.imshow(wc_3, interpolation='bilinear')

plt.axis('off')

plt.tight_layout()

plt.title('Most frequent words in NSFW removed posts', fontsize=40, verticalalignment='bottom')

plt.show()
plt.figure(figsize=(20, 20))

wc_3 = WordCloud(width=1600, height=800, background_color='black',

                 stopwords=stopwords, min_word_length=3, 

                 max_words=2000).generate(str(' '.join(df['title'][df['removed'] & ~df['over_18']])))

plt.imshow(wc_3, interpolation='bilinear')

plt.axis('off')

plt.title('Most frequent words in SFW removed posts', fontsize=40, verticalalignment='bottom')

plt.tight_layout()

plt.show()
