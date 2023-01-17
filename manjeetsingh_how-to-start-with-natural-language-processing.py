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
# text processing libraries

import re

import string

import nltk

from nltk.corpus import stopwords



# sklearn 

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt

import seaborn as sns



# File system manangement

import os



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})
percent_missing = df_train.isnull().sum() * 100 / len(df_train)

percent_missing
percent_missing2 = df_test.isnull().sum() * 100 / len(df_test)

percent_missing2
df_train['target'].value_counts(normalize=True)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=df_train[df_train['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='blue')

ax1.set_title('disaster tweets')

tweet_len=df_train[df_train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='green')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()
def create_corpus(target):

    corpus=[]

    

    for x in df_train[df_train['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
from collections import defaultdict

stop=set(stopwords.words('english'))



corpus=create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    





x,y=zip(*top)

plt.bar(x,y)
import re

import string



def clean_text_round1(text):

    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



round1 = lambda x: clean_text_round1(x)

# Let's take a look at the updated text

df_train['text'] = pd.DataFrame(df_train['text'].apply(round1))

df_test['text'] = pd.DataFrame(df_test['text'].apply(round1))

df_train.head()
# Apply a second round of cleaning

def clean_text_round2(text):

    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''

    text = re.sub('[‘’“”…]', '', text)

    text = re.sub('\n', '', text)

    return text



round2 = lambda x: clean_text_round2(x)

# Let's take a look at the updated text

df_train['text'] = pd.DataFrame(df_train['text'].apply(round2))

df_test['text'] = pd.DataFrame(df_test['text'].apply(round2))

df_train.head()
def clean_text_round3(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    return text

round3 = lambda x: clean_text_round3(x)



# Applying the cleaning function to both test and training datasets

df_train['text'] = pd.DataFrame(df_train['text'].apply(round3))

df_test['text'] = pd.DataFrame(df_test['text'].apply(round3))

df_train.head()
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

df_train['text'] = df_train['text'].apply(lambda x: tokenizer.tokenize(x))

df_test['text'] = df_test['text'].apply(lambda x: tokenizer.tokenize(x))

df_train.head()
def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words



df_train['text'] = df_train['text'].apply(lambda x : remove_stopwords(x))

df_test['text'] = df_test['text'].apply(lambda x : remove_stopwords(x))

df_train.head()
from wordcloud import WordCloud, STOPWORDS

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='black',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(df_train[df_train["target"]==1], title="Word Cloud of disaster tweets")
plot_wordcloud(df_train[df_train["target"]==0], title="Word Cloud of not real disaster tweets")
def combine_text(list_of_text):

    combined_text = ' '.join(list_of_text)

    return combined_text



df_train['text'] = df_train['text'].apply(lambda x : combine_text(x))

df_test['text'] = df_test['text'].apply(lambda x : combine_text(x))

df_train.head()
def get_top_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

plt.figure(figsize=(10,5))

top_bigrams=get_top_bigrams(df_train['text'])[:10]

x,y=map(list,zip(*top_bigrams))

sns.barplot(x=y,y=x)
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(df_train['text'])

test_vectors = count_vectorizer.transform(df_test['text'])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(df_train['text'])

test_tfidf = tfidf.transform(df_test["text"])
clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_vectors, df_train["target"], cv=5, scoring="f1")

scores
clf_NB.fit(train_vectors, df_train["target"])
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, df_train["target"], cv=5, scoring="f1")

scores
clf_NB_TFIDF.fit(train_tfidf, df_train["target"])
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)

submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_vectors=test_tfidf

submission(submission_file_path,clf_NB_TFIDF,test_vectors)