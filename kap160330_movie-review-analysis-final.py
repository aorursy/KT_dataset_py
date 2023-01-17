import pandas as pd

import numpy as np

import seaborn as sns

import re

import string

from string import punctuation

import matplotlib.pyplot as plt

%matplotlib inline



from wordcloud import WordCloud

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer, WordNetLemmatizer

from bs4 import BeautifulSoup



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split 
train = pd.read_table('../input/movie-review-sentiment-analysis-kernels-only/train.tsv',delimiter="\t",encoding="utf-8")

test = pd.read_table('../input/movie-review-sentiment-analysis-kernels-only/test.tsv',delimiter="\t",encoding="utf-8")

submission = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
submission.head()
train.head()
test.head()
df = pd.concat([train, test], ignore_index=True)

print(df.shape)
df.head()
df.tail()
newStemmer = SnowballStemmer('english')

newLemma = WordNetLemmatizer()
def cleaning(review_col):

    review_corpus=[]

    for i in range(0,len(review_col)):

        review=str(review_col[i])

        review=re.sub('[^a-zA-Z]',' ',review)

        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]

        review=[newLemma.lemmatize(w) for w in word_tokenize(str(review).lower())]

        review=' '.join(review)

        review_corpus.append(review)

    return review_corpus
df['optimized_reviews']=cleaning(df.Phrase.values)

df.head()
tfidf=TfidfVectorizer(ngram_range=(1,2),max_df=0.95,min_df=10,sublinear_tf=True)
df_train=df[df.Sentiment!=-999]

df_train.shape
df_test=df[df.Sentiment==-999]

df_test.drop('Sentiment',axis=1,inplace=True)

print(df_test.shape)

df_test.head()
train.isna().sum()
train.isnull().sum()
train['sentiment_label'] = ''

train.loc[train.Sentiment == 0, 'sentiment_label'] = 'Negative'

train.loc[train.Sentiment == 1, 'sentiment_label'] = 'Somewhat Negative'

train.loc[train.Sentiment == 2, 'sentiment_label'] = 'Neutral'

train.loc[train.Sentiment == 3, 'sentiment_label'] = 'Somewhat Positive'

train.loc[train.Sentiment == 4, 'sentiment_label'] = 'Positive'
train.head()
train.sentiment_label.value_counts()
train.shape
train = train.drop(['PhraseId', 'SentenceId'], axis=1)
train.head()
train['lengthOfPhrase'] = [len(x) for x in train.Phrase]

train.head()
sns.set_palette("dark")
fig, ax = plt.subplots(1, 1,dpi=100, figsize=(10,5))

sentiment_labels = train.sentiment_label.value_counts().index

sentiment_count = train.sentiment_label.value_counts()

sns.barplot(x=sentiment_labels,y=sentiment_count)

ax.set_ylabel('Count', fontsize = 14)    

ax.set_xlabel('Sentiment Type', fontsize = 14)

ax.set_xticklabels(sentiment_labels , rotation=30)
fig = plt.figure(figsize=[10, 10])

sentiment_labels = train.sentiment_label.value_counts().index

sentiment_count = train.sentiment_label.value_counts()

plt.pie(x=sentiment_count, labels=sentiment_labels,autopct='%0.2f %%')

plt.show
Stopwords = list(ENGLISH_STOP_WORDS) + stopwords.words()
def textPreparation(text):

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('[%s]' % re.escape(string.digits), '', text)

    text = re.sub('[%s]' % re.escape(' +'), ' ', text)

    text = text.lower()

    text = text.strip()

    return text
train['cleaned_phrase'] = ''

train['cleaned_phrase'] = [textPreparation(phrase) for phrase in train.Phrase]

test['cleaned_phrase'] = ''

test['cleaned_phrase'] = [textPreparation(phrase) for phrase in test.Phrase]
def cloud(sentiment):

    stopwordslist = Stopwords

    ## extend list of stopwords with the common words between the 3 classes which is not helpful to represent them

    stopwordslist.extend(['movie','movies','film','nt','rrb','lrb','make','work','like','story','time','little'])

    reviews = train.loc[train.Sentiment.isin(sentiment)]

    print("Word Cloud for Sentiment Labels: ", reviews.sentiment_label.unique())

    phrases = ' '.join(reviews.cleaned_phrase)

    words = " ".join([word for word in phrases.split()])

    wordcloud = WordCloud(stopwords=stopwordslist,width=3000,height=2500,background_color='white',).generate(words)

    plt.figure(figsize=(10, 10))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()

cloud([3,4])
cloud([0,1])
cloud([2])
vectorizor = CountVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1,2))

vectorizor.fit(train.Phrase)
neutral_frequency = vectorizor.transform(train[train.Sentiment == 2].Phrase)
neutral_words = neutral_frequency.sum(axis=0)

neutral_words_frequency = [(word, neutral_words[0, idx]) for word, idx in vectorizor.vocabulary_.items()]

neutral_words_tf = pd.DataFrame(list(sorted(neutral_words_frequency, key = lambda x: x[1], reverse=True)), columns=['Terms', 'neutral'])

neutral_words_tf_df = neutral_words_tf.set_index('Terms')

neutral_words_tf_df.head()
term_freq_df = pd.concat([neutral_words_tf_df],axis=1)
term_freq_df['total'] = term_freq_df['neutral']

term_freq_df.sort_values(by='total', ascending=False).head(20)
position = np.arange(50)

plt.figure(figsize=(12,10))

plt.bar(position, term_freq_df.sort_values(by='neutral', ascending=False)['neutral'][:50], align='center', alpha=0.5)

plt.xticks(position, term_freq_df.sort_values(by='neutral', ascending=False)['neutral'][:50].index,rotation='vertical')

plt.ylabel('Frequency')

plt.xlabel('Top 50 neutral words')

plt.title('Top 50 words in neutral movie reviews')