import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import bokeh

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

from matplotlib import style

import re

import time

import string

import warnings



# for all NLP related operations on text

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from nltk.classify import NaiveBayesClassifier

from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB, MultinomialNB



# To identify the sentiment of text

from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer

from textblob.np_extractors import ConllExtractor



# ignoring all the warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



# downloading stopwords corpus

nltk.download('stopwords')

#nltk.download('wordnet')

#nltk.download('vader_lexicon')

#nltk.download('averaged_perceptron_tagger')

#nltk.download('movie_reviews')

#nltk.download('punkt')

#nltk.download('conll2000')

#nltk.download('brown')

stopwords = set(stopwords.words("english"))



# for showing all the plots inline

%matplotlib inline
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sub
print("Train File")

display(df_train.head())

print("Test File")

display(test.head())
print("Train File Shape",df_train.shape)

print("Test File Shape",test.shape)
print("Train File")

display(df_train.isnull().sum())
cols = ['keyword','location']



sns.barplot(x=df_train[cols].isnull().sum(),y=df_train[cols].isnull().sum())

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)
print("Train File")

display(df_train.columns)
print("Train File")

display(df_train['target'].value_counts())

display(df_train['location'].value_counts())

display(len(df_train['keyword'].value_counts()))
df_train['Text_Length']=df_train['text'].apply(lambda x:len(x) - x.count(" "))

df_train.head()
df_train['target_mean'] = df_train.groupby('keyword')['target'].transform('mean')



fig = plt.figure(figsize=(8, 72), dpi=100)



sns.countplot(y=df_train.sort_values(by='target_mean', ascending=False)['keyword'],

              hue=df_train.sort_values(by='target_mean', ascending=False)['target'])



plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=12)

plt.legend(loc=1)

plt.title('Target Distribution in Keywords')



plt.show()



df_train.drop(columns=['target_mean'], inplace=True)
sns.countplot(df_train['target'])

plt.tick_params(axis='x', labelsize=15)
def hashtag_extract(text_list):

    hashtags = []

    # Loop over the words in the tweet

    for text in text_list:

        ht = re.findall(r"#(\w+)", text)

        hashtags.append(ht)



    return hashtags



def generate_hashtag_freqdist(hashtags):

    a = nltk.FreqDist(hashtags)

    d = pd.DataFrame({'Hashtag': list(a.keys()),

                      'Count': list(a.values())})   

    d = d.nlargest(columns="Count", n = 50)

    plt.figure(figsize=(16,7))

    ax = sns.barplot(data=d, x= "Hashtag", y = "Count")

    plt.xticks(rotation=80)

    ax.set(ylabel = 'Count')

    plt.show()
hashtags = hashtag_extract(df_train['text'])

df_train['hashtags'] = hashtags

hashtags = sum(hashtags, [])
hashtags
generate_hashtag_freqdist(hashtags)
df_train.head()
df_train['hashtags'] = df_train['hashtags'].astype(str)
df_train['hashtags'] = df_train['hashtags'].str.strip('[')

df_train['hashtags'] = df_train['hashtags'].str.strip(']').astype(str)

df_train
df_train['hashtags'] = df_train['hashtags'].replace({'':np.nan})

df_train
# 1 way

def fetch_sentiment_using_SIA(text):

    sid = SentimentIntensityAnalyzer()

    polarity_scores = sid.polarity_scores(text)

    return 'neg' if polarity_scores['neg'] > polarity_scores['pos'] else 'pos'



# 2 way

def fetch_sentiment_using_textblob(text):

    analysis = TextBlob(text)

    return 'pos' if analysis.sentiment.polarity >= 0 else 'neg'
sentiments_using_SIA = df_train.text.apply(lambda tweet: fetch_sentiment_using_SIA(tweet))

display(pd.DataFrame(sentiments_using_SIA.value_counts()))



sentiments_using_textblob = df_train.text.apply(lambda tweet: fetch_sentiment_using_textblob(tweet))

display(pd.DataFrame(sentiments_using_textblob.value_counts()))
df_train['sentiment'] = sentiments_using_SIA

df_train.head()
sns.countplot(x= df_train['sentiment'])
train = df_train.drop(['Text_Length','hashtags','sentiment'],axis=1)

train
def remove_pattern(text, pattern_regex):

    r = re.findall(pattern_regex, text)

    for i in r:

        text = re.sub(i, '', text)

    

    return text 
# We are keeping cleaned tweets in a new column called 'tidy_tweets'

train['tidy_tweets'] = np.vectorize(remove_pattern)(train['text'], "@[\w]*: | *RT*")

train.head(10)



test['tidy_tweets'] = np.vectorize(remove_pattern)(test['text'], "@[\w]*: | *RT*")
cleaned_tweets = []

cleaned_tweets_test = []



for index, row in train.iterrows():

    # Here we are filtering out all the words that contains link

    words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]

    cleaned_tweets.append(' '.join(words_without_links))

    

for index, row in test.iterrows():

    # Here we are filtering out all the words that contains link

    words_without_links = [word for word in row.tidy_tweets.split() if 'http' not in word]

    cleaned_tweets_test.append(' '.join(words_without_links))



train['tidy_tweets'] = cleaned_tweets

test['tidy_tweets'] = cleaned_tweets_test

train.head(10)
train = train[train['tidy_tweets']!='']

test = test[test['tidy_tweets']!='']

train.head()
train.drop_duplicates(subset=['tidy_tweets'], keep=False)

test.drop_duplicates(subset=['tidy_tweets'], keep=False)

train.head()
train = train.reset_index(drop=True)

test = test.reset_index(drop=True)

train.head()
train['absolute_tidy_tweets'] = train['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")

test['absolute_tidy_tweets'] = test['tidy_tweets'].str.replace("[^a-zA-Z# ]", "")
stopwords_set = set(stopwords)

cleaned_tweets = []

cleaned_tweets_test = []



for index, row in train.iterrows():

    

    # filerting out all the stopwords 

    words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets.append(' '.join(words_without_stopwords))

    

    

for index, row in test.iterrows():

    

    # filerting out all the stopwords 

    words_without_stopwords = [word for word in row.absolute_tidy_tweets.split() if not word in stopwords_set and '#' not in word.lower()]

    

    # finally creating tweets list of tuples containing stopwords(list) and sentimentType 

    cleaned_tweets_test.append(' '.join(words_without_stopwords))

    

    

train['absolute_tidy_tweets'] = cleaned_tweets

test['absolute_tidy_tweets'] = cleaned_tweets_test

train.head(10)
tokenized_tweet = train['absolute_tidy_tweets'].apply(lambda x: x.split())



tokenized_tweet_test = test['absolute_tidy_tweets'].apply(lambda x: x.split())



tokenized_tweet.head()
word_lemmatizer = WordNetLemmatizer()



tokenized_tweet = tokenized_tweet.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])



tokenized_tweet_test = tokenized_tweet_test.apply(lambda x: [word_lemmatizer.lemmatize(i) for i in x])

tokenized_tweet.head()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")



tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])



tokenized_tweet_test = tokenized_tweet_test.apply(lambda x: [stemmer.stem(i) for i in x])

tokenized_tweet.head()
for i, tokens in enumerate(tokenized_tweet):

    tokenized_tweet[i] = ' '.join(tokens)





for i, tokens in enumerate(tokenized_tweet_test):

    tokenized_tweet_test[i] = ' '.join(tokens)



    

train['absolute_tidy_tweets'] = tokenized_tweet

test['absolute_tidy_tweets'] = tokenized_tweet_test

train.head(10)
class PhraseExtractHelper(object):

    def __init__(self):

        self.lemmatizer = nltk.WordNetLemmatizer()

        self.stemmer = nltk.stem.porter.PorterStemmer()

    

    def leaves(self, tree):

        """Finds NP (nounphrase) leaf nodes of a chunk tree."""

        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):

            yield subtree.leaves()



    def normalise(self, word):

        """Normalises words to lowercase and stems and lemmatizes it."""

        word = word.lower()

        # word = self.stemmer.stem_word(word) # We will loose the exact meaning of the word 

        word = self.lemmatizer.lemmatize(word)

        return word



    def acceptable_word(self, word):

        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""

        accepted = bool(3 <= len(word) <= 40

            and word.lower() not in stopwords

            and 'https' not in word.lower()

            and 'http' not in word.lower()

            and '#' not in word.lower()

            )

        return accepted



    def get_terms(self, tree):

        for leaf in self.leaves(tree):

            term = [ self.normalise(w) for w,t in leaf if self.acceptable_word(w) ]

            yield term
sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'

grammar = r"""

    NBAR:

        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        

    NP:

        {<NBAR>}

        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...

"""

chunker = nltk.RegexpParser(grammar)
key_phrases = []

key_phrases_test = []

phrase_extract_helper = PhraseExtractHelper()



for index, row in train.iterrows(): 

    toks = nltk.regexp_tokenize(row.tidy_tweets, sentence_re)

    postoks = nltk.tag.pos_tag(toks)

    tree = chunker.parse(postoks)



    terms = phrase_extract_helper.get_terms(tree)

    tweet_phrases = []



    for term in terms:

        if len(term):

            tweet_phrases.append(' '.join(term))

    

    key_phrases.append(tweet_phrases)

    

key_phrases[:10]



for index, row in test.iterrows(): 

    toks = nltk.regexp_tokenize(row.tidy_tweets, sentence_re)

    postoks = nltk.tag.pos_tag(toks)

    tree = chunker.parse(postoks)



    terms = phrase_extract_helper.get_terms(tree)

    tweet_phrases_test = []



    for term in terms:

        if len(term):

            tweet_phrases_test.append(' '.join(term))

    

    key_phrases_test.append(tweet_phrases_test)

    

key_phrases_test[:10]
textblob_key_phrases = []

textblob_key_phrases_test = []

extractor = ConllExtractor()



for index, row in train.iterrows():

    # filerting out all the hashtags

    words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]

    

    hash_removed_sentence = ' '.join(words_without_hash)

    

    blob = TextBlob(hash_removed_sentence, np_extractor=extractor)

    textblob_key_phrases.append(list(blob.noun_phrases))



textblob_key_phrases[:10]



for index, row in test.iterrows():

    # filerting out all the hashtags

    words_without_hash = [word for word in row.tidy_tweets.split() if '#' not in word.lower()]

    

    hash_removed_sentence = ' '.join(words_without_hash)

    

    blob = TextBlob(hash_removed_sentence, np_extractor=extractor)

    textblob_key_phrases_test.append(list(blob.noun_phrases))



textblob_key_phrases_test[:10]
train['key_phrases'] = textblob_key_phrases



test['key_phrases'] = textblob_key_phrases_test

train.head(10)
def generate_wordcloud(all_words):

    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=100, relative_scaling=0.5, colormap='Dark2').generate(all_words)



    plt.figure(figsize=(14, 10))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off')

    plt.show()
all_words = ' '.join([text for text in train['absolute_tidy_tweets'][train.target == 1]])

generate_wordcloud(all_words)
all_words = ' '.join([text for text in train['absolute_tidy_tweets'][train.target == 0]])

generate_wordcloud(all_words)
tweets_df = train[train['key_phrases'].str.len()>0]
tweets_df
phrase_sents = tweets_df['key_phrases'].apply(lambda x: ' '.join(x))

tweets_df['Concatenated'] = tweets_df['absolute_tidy_tweets'] + phrase_sents



phrase_sents_test = test['key_phrases'].apply(lambda x: ' '.join(x))

test['Concatenated'] = test['absolute_tidy_tweets'] + phrase_sents_test
bow_word_vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words='english')

bow_word_feature = bow_word_vectorizer.fit_transform(tweets_df['absolute_tidy_tweets'])



phrase_sents = tweets_df['key_phrases'].apply(lambda x: ' '.join(x))



bow_phrase_vectorizer = CountVectorizer(max_df=0.90, min_df=2)

bow_phrase_feature = bow_phrase_vectorizer.fit_transform(phrase_sents)
# TF-IDF features

tfidf_word_vectorizer = TfidfVectorizer(analyzer='word', binary=True,max_df=0.99, min_df=1, stop_words='english',ngram_range=(1,3))



tfidf_word_vectorizer.fit(tweets_df['Concatenated'])



# TF-IDF feature matrix

tfidf_word_feature = tfidf_word_vectorizer.fit_transform(tweets_df['Concatenated'])



test_vectors = tfidf_word_vectorizer.transform(test['Concatenated'])

# TF-IDF feature matrix

#tfidf_word_feature_test = tfidf_word_vectorizer.fit_transform(test['Concatenated'])
phrase_sents = tweets_df['key_phrases'].apply(lambda x: ' '.join(x))

# TF-IDF phrase feature

tfidf_phrase_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, ngram_range=(1,2))

tfidf_phrase_feature = tfidf_phrase_vectorizer.fit_transform(phrase_sents)
target_variable = tweets_df['target']
def plot_confusion_matrix(matrix):

    plt.clf()

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Set2_r)

    classNames = ['Positive', 'Negative']

    plt.title('Confusion Matrix')

    plt.ylabel('Predicted')

    plt.xlabel('Actual')

    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames)

    plt.yticks(tick_marks, classNames)

    s = [['TP','FP'], ['FN', 'TN']]



    for i in range(2):

        for j in range(2):

            plt.text(j,i, str(s[i][j])+" = "+str(matrix[i][j]))

    plt.show()
def naive_model(X_train, X_test, y_train, y_test):

    naive_classifier = MultinomialNB()

    naive_classifier.fit(X_train.toarray(), y_train)



    # predictions over test set

    predictions = naive_classifier.predict(X_test.toarray())

    print(f1_score(y_test,predictions))



    # calculating Accuracy Score

    print(f'Accuracy Score :: {accuracy_score(y_test, predictions)}')

    conf_matrix = confusion_matrix(y_test, predictions, labels=[True, False])

    plot_confusion_matrix(conf_matrix)

    

    return naive_classifier
X_train, X_test, y_train, y_test = train_test_split(tfidf_word_feature, target_variable, test_size=0.3, random_state=272)

TF_word = naive_model(X_train, X_test, y_train, y_test)

TF_word
X_train, X_test, y_train, y_test = train_test_split(tfidf_phrase_feature, target_variable, test_size=0.3, random_state=272)

TF_phrase = naive_model(X_train, X_test, y_train, y_test)

TF_phrase
X_train, X_test, y_train, y_test = train_test_split(bow_phrase_feature, target_variable, test_size=0.3, random_state=272)

BOW_phrase = naive_model(X_train, X_test, y_train, y_test)
X_train, X_test, y_train, y_test = train_test_split(bow_word_feature, target_variable, test_size=0.3, random_state=272)

BOW_words = naive_model(X_train, X_test, y_train, y_test)
from sklearn.metrics import f1_score
test['target'] = TF_word.predict(test_vectors)
test
submission= test[['id','target']]
submission.set_index('id',inplace=True)

submission.head()
submission.to_csv('submission.csv')