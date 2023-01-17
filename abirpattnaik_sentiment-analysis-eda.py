# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import regex

import nltk

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer,PorterStemmer

from nltk.corpus import stopwords

import re

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score

from PIL import Image

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample_submission=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train_data.head()
print('The train set contains {0} rows and {1} columns '.format(train_data.shape[0],train_data.shape[1]))
def count_target_plot(data,target):

    plt.figure(figsize=(8,8))

    ax=sns.countplot(data=data,x=data[target],order=data[target].value_counts().index)

    plt.xlabel('Target Variable- Sentiment')

    plt.ylabel('Count of tweets')

    plt.title('Count of Sentiment tweets')

    total = len(data)

    for p in ax.patches:

            ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

count_target_plot(train_data,'sentiment')
count_target_plot(test_data,'sentiment')
train_data.tail()
lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer() 

def preprocess(sentence):

    sentence=str(sentence)

    sentence = sentence.lower()

    sentence=sentence.replace('{html}',"") 

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, '', sentence)

    rem_url=re.sub(r'http\S+', '',cleantext)

    rem_num = re.sub('[0-9]+', '', rem_url)

    tokenizer = RegexpTokenizer(r'\w+')

    tokens = tokenizer.tokenize(rem_num)  

    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]

    stem_words=[stemmer.stem(w) for w in filtered_words]

    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]

    return " ".join(filtered_words)
train_data['text']=train_data['text'].map(lambda s:preprocess(s))

train_data['selected_text']=train_data['selected_text'].map(lambda s:preprocess(s))
test_data.head()
test_data['text']=test_data['text'].map(lambda s:preprocess(s))
train_data.head()
print('Checking null values for train data')

print(train_data.isnull().sum())

print('Checking null values for train data')

print(test_data.isnull().sum())
# https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

# Interesting article

def get_top_n_words(corpus, n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    

    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 

    [('python', 2),

     ('world', 2),

     ('love', 2),

     ('hello', 1),

     ('is', 1),

     ('programming', 1),

     ('the', 1),

     ('language', 1)]

    """

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
top_40_words=pd.DataFrame(get_top_n_words(train_data['text'],40),columns=['words','count'])
def count_words_barplot(data_set):

    plt.figure(figsize=(20,20))

    ax = sns.barplot(y="count", x="words", data=data_set)

    plt.xlabel('Count of words')

    plt.ylabel('Words')

    plt.title('Count of Top 30 words used in the tweet')
count_words_barplot(top_40_words)
##Dividing on the basis of tweets

neutral_set=train_data[train_data['sentiment']=='neutral'].reset_index()

positive_set=train_data[train_data['sentiment']=='positive'].reset_index()

negative_set=train_data[train_data['sentiment']=='negative'].reset_index()
top_30_words_neutral=pd.DataFrame(get_top_n_words(neutral_set['text'],30),columns=['words','count'])

top_30_words_positive=pd.DataFrame(get_top_n_words(positive_set['text'],30),columns=['words','count'])

top_30_words_negative=pd.DataFrame(get_top_n_words(negative_set['text'],30),columns=['words','count'])
count_words_barplot(top_30_words_neutral)
fig = px.treemap(top_30_words_positive,path=['words'], values='count',title='List of top 30 words that affect positive tweets')

fig.show()
top_300_words_negative=pd.DataFrame(get_top_n_words(negative_set['text'],300),columns=['words','count'])
fig = px.sunburst(top_300_words_negative,path=['words'], values='count',color='words',title='Top negative words that are present in the data')

fig.show()
bigram_pos=pd.DataFrame()

bigram_neg=pd.DataFrame()

bigram_neu=pd.DataFrame()
#train_data_len=len(train_data)



neutral_set_len=len(neutral_set)

positive_set_len=len(positive_set)

negative_set_len=len(negative_set)
for index in range(0,neutral_set_len):

    bigrm = pd.DataFrame(nltk.bigrams(neutral_set['text'][index].split()))

    bigram_neu=pd.concat([bigram_neu,bigrm])



#bigram_neu.head()
for index in range(0,positive_set_len):

    bigrm = pd.DataFrame(nltk.bigrams(positive_set['text'][index].split()))

    bigram_pos=pd.concat([bigram_pos,bigrm])



#bigram_pos.head()
for index in range(0,negative_set_len):

    bigrm = pd.DataFrame(nltk.bigrams(negative_set['text'][index].split()))

    bigram_neg=pd.concat([bigram_neg,bigrm])



#bigram_neg.head()
#bigram_neu['bigram']=bigram_neu[['0','1']].apply(lambda x:' '.join(x),axis=1)

#bigram_pos=bigram_pos.reset_index()

#bigram_neg=bigram_neg.reset_index()

bigram_neu=bigram_neu.rename(columns={0:'first',1:'second'}).reset_index()

bigram_pos=bigram_pos.rename(columns={0:'first',1:'second'}).reset_index()

bigram_neg=bigram_neg.rename(columns={0:'first',1:'second'}).reset_index()
bigram_neu['combined'] = bigram_neu[['first', 'second']].apply(lambda x: ' '.join(x), axis = 1) 

bigram_pos['combined'] = bigram_pos[['first', 'second']].apply(lambda x: ' '.join(x), axis = 1) 

bigram_neg['combined'] = bigram_neg[['first', 'second']].apply(lambda x: ' '.join(x), axis = 1) 

bigram_neu_count=pd.DataFrame(bigram_neu['combined'].value_counts()).reset_index()

bigram_pos_count=pd.DataFrame(bigram_pos['combined'].value_counts()).reset_index()

bigram_neg_count=pd.DataFrame(bigram_neg['combined'].value_counts()).reset_index()
fig = px.scatter(bigram_neu_count[:30], x="index", y="combined",color='combined',

	         size="combined", size_max=20,title='Neutral count of top bigram words')

fig.show()
fig = px.bar(bigram_pos_count[:30], x="combined", y="index", color='combined',orientation='h',title='Positive count words top 30')

fig.show()
fig = px.line(bigram_neg_count[:30], x="index", y="combined",title='Negative count of top 30 bigram words')

fig.show()
wordcloud = WordCloud(

                          background_color='white',

                          max_words=100,

                          max_font_size=80, 

                          random_state=42,

    collocations=False,

    colormap="Oranges_r"

                         ).generate(' '.join(top_40_words['words']))

#.join(text2['Crime Type']))



plt.figure(figsize=(10,10))

plt.title('Major keywords for tweets', fontsize=10)

plt.imshow(wordcloud)



plt.axis('off')

plt.show()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



A= "Going lucky or going hard is the only thing you could do"

B="going is important or only thing is needed"

C="It is going lucky or going hard is only needed thing you"



print(jaccard(A,B))

print(jaccard(A,C))