# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import seaborn as sns

import matplotlib.pyplot as plt

import re

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

import re

from sklearn.feature_extraction.text import TfidfTransformer

from scipy.sparse import coo_matrix

import string

from nltk.tokenize import word_tokenize

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train
test
x=train.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')
train['word_count'] = train['text'].apply(lambda x: len(str(x).split(" ")))

train[['text','word_count']].head()
train.word_count.describe()
#Most common words

freq = pd.Series(' '.join(train['text']).split()).value_counts()[:20]

freq
#Identify uncommon words

freq1 =  pd.Series(' '.join(train 

         ['text']).split()).value_counts()[-20:]

freq1
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



# Applying the cleaning function to both test and training datasets

train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))
text = "Are you coming , aren't you"

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')



print("Example Text: ",text)

print("------------------------------------------------------------------------------------------------")

print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))

print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))

print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))

print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words

train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head()
def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))

train['text']

train.head()
def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return combined_text
##Creating a list of stop words and adding custom stopwords

stop_words = set(stopwords.words("english"))

##Creating a list of custom stopwords

new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

stop_words = stop_words.union(new_words)
corpus = []

for i in range(0, 3847):

    #Remove punctuations

    text = re.sub('[^a-zA-Z]', ' ', train['text'][i])

    

    #Convert to lowercase

    text = text.lower()

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    

    # remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    

    ##Convert to list from string

    text = text.split()

    

    ##Stemming

    ps=PorterStemmer()

    #Lemmatisation

    lem = WordNetLemmatizer()

    text = [lem.lemmatize(word) for word in text if not word in  

            stop_words] 

    text = " ".join(text)

    corpus.append(text)
#Word cloud

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

%matplotlib inline

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=100,

                          max_font_size=50, 

                          random_state=42

                         ).generate(str(corpus))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)
#Most frequently occuring words

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in      

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot

top_words = get_top_n_words(corpus, n=20)

top_df = pd.DataFrame(top_words)

top_df.columns=["Word", "Freq"]

#Barplot of most freq words

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

g = sns.barplot(x="Word", y="Freq", data=top_df)

g.set_xticklabels(g.get_xticklabels(), rotation=30)
#Most frequently occuring Bi-grams

def get_top_n2_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(2,2),  

            max_features=2000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top2_words = get_top_n2_words(corpus, n=20)

top2_df = pd.DataFrame(top2_words)

top2_df.columns=["Bi-gram", "Freq"]

print(top2_df)

#Barplot of most freq Bi-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)

h.set_xticklabels(h.get_xticklabels(), rotation=45)
#Most frequently occuring Tri-grams

def get_top_n3_words(corpus, n=None):

    vec1 = CountVectorizer(ngram_range=(3,3), 

           max_features=2000).fit(corpus)

    bag_of_words = vec1.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]

top3_words = get_top_n3_words(corpus, n=20)

top3_df = pd.DataFrame(top3_words)

top3_df.columns=["Tri-gram", "Freq"]

print(top3_df)

#Barplot of most freq Tri-grams

import seaborn as sns

sns.set(rc={'figure.figsize':(13,8)})

j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)

j.set_xticklabels(j.get_xticklabels(), rotation=45)
example="We always try to bring the heavy. #metal #RT http://t.co/YAo1e0xngw"
#remove URL

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



remove_URL(example)
train['text']=train['text'].apply(lambda x : remove_URL(x))

test['text']=test['text'].apply(lambda x : remove_URL(x))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



remove_emoji("Omg another Earthquake 😔😔")
train['text']=train['text'].apply(lambda x: remove_emoji(x))

test['text']=test['text'].apply(lambda x: remove_emoji(x))
import string

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas"

print(remove_punct(example))
train['text']=train['text'].apply(lambda x : remove_punct(x))

test['text']=test['text'].apply(lambda x : remove_punct(x))
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

scores
clf.fit(train_vectors, train["target"])
# Fitting a simple Logistic Regression on TFIDF

clf_tfidf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

scores
# Fitting a simple Naive Bayes on Counts

clf_NB = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB, train_vectors, train["target"], cv=5, scoring="f1")

scores
clf_NB.fit(train_vectors, train["target"])
# Fitting a simple Naive Bayes on TFIDF

clf_NB_TFIDF = MultinomialNB()

scores = model_selection.cross_val_score(clf_NB_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
clf_NB_TFIDF.fit(train_tfidf, train["target"])
import xgboost as xgb

clf_xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb, train_vectors, train["target"], cv=5, scoring="f1")

scores
import xgboost as xgb

clf_xgb_TFIDF = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf_xgb_TFIDF, train_tfidf, train["target"], cv=5, scoring="f1")

scores
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.predict(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_vectors=test_tfidf

submission(submission_file_path,clf_NB_TFIDF,test_vectors)