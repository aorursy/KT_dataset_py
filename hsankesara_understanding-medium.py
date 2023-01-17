import pandas as pd
import numpy as np
import nltk
df = pd.read_csv('../input/articles.csv')
df.describe()
df.info()
df.head()
df['len_text'] = df['text'].str.len()
df['len_title'] = df['title'].str.len()
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
df['claps'] = df['claps'].apply(lambda s: int(float(s[:-1]) * 1000) if s[-1] == 'K' else int(s))
df.drop('link', axis = 1, inplace=True)
sns.distplot(df['len_text'], color="b")
plt.show()
sns.distplot(df['len_title'], color="b")
plt.show()
sns.distplot(df['claps'], color="b")
plt.show()
df.head()
sns.pointplot('reading_time', 'claps', data=df)
plt.show()
sns.regplot('reading_time', 'claps', data=df, order=3)
plt.show()
sns.regplot('len_text', 'claps', data=df, order=3)
plt.show()
a4_dims = (25.7, 40.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('len_title', 'claps', data=df, orient='h')
plt.show()
## Let's check relation between title length and article length
sns.lmplot('len_title', 'len_text', data=df,order=3)
plt.show()
## Let's check relation between total claps recieved and article length
a4_dims = (12, 6)
fig, ax = plt.subplots(figsize=a4_dims)
sns.pointplot('len_title', 'claps', data=df)
plt.show()
sns.heatmap(df[['claps', 'len_text', 'len_title', 'reading_time']].corr(),annot=True, cmap='BrBG')
plt.show()
a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('claps', 'author', data = df, orient='h')
plt.show()
a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot('len_text', 'author', data = df, orient='h')
plt.show()
a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('len_text', 'author', data = df, orient='h')
plt.show()
a4_dims = (25.7, 35.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('len_title', 'author', data = df, orient='h')
plt.show()
## Finding top articls
df[df['claps'] >= df['claps'].quantile(0.95)][['author', 'title', 'claps']]
df.head()
df['title'] = df['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['title'].head()
df['title'] = df['title'].str.replace('[^\w\s]','')
df['title'].head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['title'] = df['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df.head()
def get_words_count(df_series, col):
    words_count = {}
    m = df_series.shape[0]
    for i in range(m):
        words = df[col].iat[i].split()
        for word in words:
            if word.lower() in words_count:
                words_count[word.lower()] += 1
            else:
                words_count[word.lower()] = 1
    return words_count
title_words = get_words_count(df, 'title')
title_words = pd.DataFrame(list(title_words.items()), columns=['words', 'count'])
sns.distplot(title_words['count'], color='b')
plt.show()
## List of 15 most frequent words occurred in title
title_words.sort_values(by='count', ascending=False).head(15)
from wordcloud import WordCloud
fig = plt.figure(dpi=100)
a4_dims = (6, 12)
fig, ax = plt.subplots(figsize=a4_dims)
wordcloud = WordCloud(background_color ='white', max_words=200,max_font_size=40,random_state=3).generate(str(title_words.sort_values(by='count', ascending=False)['words'].values[:20]))
plt.imshow(wordcloud)
plt.title = 'Top Word in the title of Medium Articles'
plt.show()
title_words.head()
## Let's get list of top ten words
topten_title_words = title_words.sort_values(by='count', ascending=False)['words'].values[:10]
## Count occurence of top ten words in every title in dataframe
df['topten_title_count'] = df['title'].apply(lambda s: sum(s.count(topten_title_words[i]) for  i in range(10)))
df.head()
sns.regplot('topten_title_count', 'claps', data = df, order=3)
plt.show()
sns.barplot('topten_title_count', 'claps', data = df)
plt.show()
sns.distplot(df['topten_title_count'], color='b')
plt.show()
sns.heatmap(df[['topten_title_count', 'len_text', 'len_title']].corr(), annot=True, cmap='BrBG')
plt.show()
sns.jointplot('topten_title_count', 'claps', data=df, kind='hex')
plt.show()
df.head()
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['text'].head()
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'].head()
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df.head()
text_words = get_words_count(df, 'text')
text_words = pd.DataFrame(list(text_words.items()), columns=['words', 'count'])
sns.distplot(text_words['count'], color='b')
plt.show()
## Most frequent 15 words in articles
text_words.sort_values(by='count', ascending=False).head(15)
fig = plt.figure(dpi=100)
a4_dims = (6, 12)
fig, ax = plt.subplots(figsize=a4_dims)
wordcloud = WordCloud(background_color ='white', max_words=200,max_font_size=40,random_state=3).generate(str(text_words.sort_values(by='count', ascending=False)['words'].values[:20]))
plt.imshow(wordcloud)
plt.title = 'Top Word in the text of Medium Articles'
plt.show()
## get list of most frequent 10 words in text
topten_text_words = text_words.sort_values(by='count', ascending=False)['words'].values[:10]
df['topten_text_count'] = df['text'].apply(lambda s: sum(s.count(topten_text_words[i]) for  i in range(10)))
df.head()
sns.regplot('topten_text_count', 'claps', data = df, order=3)
plt.show()
a4_dims = (6, 25)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('claps', 'topten_text_count', data = df, orient='h')
plt.show()
sns.distplot(df['topten_text_count'], color='b')
plt.show()
sns.heatmap(df[['topten_text_count', 'len_text', 'len_title']].corr(), annot=True, cmap='BrBG')
plt.show()
sns.jointplot('topten_text_count', 'claps', data=df, kind='hex')
plt.show()
df.head()
df_author = df.groupby(['author']).mean().reset_index()
df_top30 = df_author.sort_values(ascending=False, by='claps')[:30]
a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot('reading_time', 'claps', data=df_top30)
plt.show()
a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.distplot(df_top30['len_text'])
plt.show()
sns.kdeplot(df_top30['topten_text_count'], df_top30['topten_title_count'], shade=True, cbar=True)
plt.show()
## Relationship  between length of text with lenght of title 
sns.kdeplot(df_top30['len_text'], df_top30['len_title'], shade=True, cbar=True)
plt.show()
sns.clustermap(df_top30[['reading_time', 'claps', 'len_title', 'len_text', 'topten_text_count', 'topten_title_count']],cmap="mako", robust=True)
plt.show()