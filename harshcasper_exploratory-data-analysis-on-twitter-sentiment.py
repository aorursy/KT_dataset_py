# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

print(df_train.shape)

print(df_test.shape)
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.figure_factory as ff

import re

import string

from collections import Counter

import plotly.express as px

from nltk.corpus import stopwords

import spacy
df_train.dropna(inplace=True)

df_train.info()
df_test.info()
df_train.head()
df_train.tail()
temp = df_train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text', ascending=False)  

temp
plt.figure(figsize=(12, 6))

sns.countplot(x='sentiment', data=df_train)
def jaccard(str1, str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    

    return float(len(c))/len(a) + len(b) - len(c)
results_jaccard = []



for ind, row in df_train.iterrows():

    sent1 = row.text

    sent2 = row.selected_text

    

    jaccard_score = jaccard(sent1, sent2)

    results_jaccard.append([sent1, sent2, jaccard_score])
df_jaccard = pd.DataFrame(results_jaccard, columns=['text', 'selected_text', 'jaccard_score'])

df_train = df_train.merge(df_jaccard, how='outer')
# Number of words in selected text

df_train['Num_words_ST'] = df_train['selected_text'].apply(lambda x:len(str(x).split()))



# Number of words in text

df_train['Num_words_Text'] = df_train['text'].apply(lambda x:len(str(x).split()))



# Difference in number of words in text and selected text

df_train['difference_in_words'] = df_train['Num_words_Text'] - df_train['Num_words_ST']
df_train.head()
hist_data = [df_train['Num_words_ST'], df_train['Num_words_Text']]



group_labels = ['Selected_Text', 'Text']



# Create displot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, show_curve=False)

fig.update_layout(title_text='Distribution of Number of words')

fig.update_layout(

    autosize=False, 

    width=900,

    height=700

)



fig.show()
plt.figure(figsize=(12, 6))

p1 = sns.kdeplot(df_train['Num_words_ST'], shade=True, color='r').set_title('Kernel Distribution of Words')



p1 = sns.kdeplot(df_train['Num_words_Text'], shade=True, color='b')
plt.figure(figsize=(12, 6))

p1 = sns.kdeplot(df_train[df_train['sentiment']=='positive']['difference_in_words'], shade=True, color='r').set_title('Kernel Distribution of Difference in Number of Words')

p1 = sns.kdeplot(df_train[df_train['sentiment']=='negative']['difference_in_words'], shade=True, color='b')
plt.figure(figsize=(12, 6))

sns.distplot(df_train[df_train['sentiment']=='neutral']['difference_in_words'], kde=False)
plt.figure(figsize=(12, 6))

p1 = sns.kdeplot(df_train[df_train['sentiment']=='positive']['jaccard_score'], shade=True, color='b').set_title('Kerel Distribution of Jaccard Score across different Sentiments')

p1 = sns.kdeplot(df_train[df_train['sentiment']=='negative']['jaccard_score'], shade=True, color='r')

plt.legend(labels=['positive', 'negative'])
plt.figure(figsize=(12, 6))

sns.distplot(df_train[df_train['sentiment']=='neutral']['jaccard_score'], kde=False)
k = df_train[df_train['Num_words_Text']<=2]
k.groupby('sentiment').mean()['jaccard_score']
k[k['sentiment']=='positive']
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
df_train['text'] = df_train['text'].apply(lambda x:clean_text(x))

df_train['selected_text'] = df_train['selected_text'].apply(lambda x:clean_text(x))
# Most Common Word



df_train['temp_list'] = df_train['selected_text'].apply(lambda x : str(x).split())

temp = Counter(item for sublist in df_train['temp_list'] for item in sublist)

top = pd.DataFrame(temp.most_common(20))

top.columns = ['Common_words', 'count']

top
fig = px.bar(top, x='count', y='Common_words', title='Common words in Selected Text', orientation='h', width=700, height=700, color='Common_words')

fig.show()
def remove_stopwords(x):

    return [y for y in x if y not in stopwords.words('english')]



df_train['temp_list'] = df_train['temp_list'].apply(lambda x : remove_stopwords(x))
top = Counter(item for sublist in df_train['temp_list'] for item in sublist)

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words', 'count']

temp
fig = px.treemap(temp, path=['Common_words'], values='count', title='Tree of the most Common Words')

fig.show()
# Most Common Words



df_train['temp_list1'] = df_train['text'].apply(lambda x : str(x).split())

df_train['temp_list1'] = df_train['temp_list1'].apply(lambda x : remove_stopwords(x))
top = Counter([item for sublist in df_train['temp_list1'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words', 'Count']

temp
fig = px.bar(temp, x='Count', y='Common_words', title='Common words in text', orientation='h', width=700, height=700, color='Common_words')

fig.show()
Positive_sent = df_train[df_train['sentiment']=='positive']

Negative_sent = df_train[df_train['sentiment']=='negative']

Neutral_sent = df_train[df_train['sentiment']=='neutral']
top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words', 'Count']

temp
fig = px.bar(temp, x='Count', y='Common_words', title='Most Common Positive Words', orientation='h', width=700, height=700, color='Common_words')

fig.show()
top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words', 'Count']

temp
fig = px.bar(temp, x='Count', y='Common_words', title='Most Common Negative Words', orientation='h', width=700, height=700, color='Common_words')

fig.show()
top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words', 'Count']

temp
fig = px.bar(temp, x='Count', y='Common_words', title='Most Common Neutral Words', orientation='h', width=700, height=700, color='Common_words')

fig.show()
raw_text = [word for word_list in df_train['temp_list'] for word in word_list]

def words_unique(sentiment, numwords, raw_text):

    

    allother = []

    for sublist in df_train[df_train.sentiment != sentiment]['temp_list']:

        for item in sublist:

            allother.append(item)

            

    allother = set(allother)

    

    specificonly = [x for x in raw_text if x not in allother]

    

    word_count = Counter()

    

    for sublist in df_train[df_train.sentiment == sentiment]['temp_list']:

        for item in sublist:

            word_count[item] += 1

    

    for word in list(word_count):

        if word not in specificonly:

            del word_count[word]

        

    unique_words = pd.DataFrame(word_count.most_common(numwords), columns=['word', 'count'])

    

    return unique_words
unique_positive = words_unique('positive', 20, raw_text)

print('Top 20 words in positive tweets are:')

unique_positive
fig = px.treemap(unique_positive, path=['word'], values='count', title='Tree of Unique Positive Words')

fig.show()
unique_negative = words_unique('negative', 20, raw_text)

print('Top 20 words in negative tweets are:')

unique_negative
fig = px.treemap(unique_negative, path=['word'], values='count', title='Tree of Unique Negative Words')

fig.show()
unique_neutral = words_unique('neutral', 20, raw_text)

print('Top 20 words in neutral tweets are:')

temp
fig = px.treemap(unique_neutral, path=['word'], values='count', title='Tree of Unique Neutral Words')

fig.show()