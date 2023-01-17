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
import pandas as pd

import numpy as np



# Text preprocessing

import re

import string

import nltk

from nltk.corpus import stopwords



#XGboost

import xgboost as xbg

from xgboost import XGBClassifier



# Sklearn

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing ,decomposition,model_selection,metrics,pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV



# Matplotlib and Seaborn

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize']=10,6

plt.rcParams['axes.grid']=True

plt.gray()



import seaborn as sns



import os



import warnings

warnings.filterwarnings('ignore')
#Training data

train = pd.read_csv('../input/nlp-getting-started/train.csv')

print('Training data shape: ', train.shape)

train.head()
# Testing data

test = pd.read_csv('../input/nlp-getting-started/test.csv')

print('Testing data shape: ',test.shape)

test.head()
def missing_value_of_data(data):

    total = data.isnull().sum().sort_values(ascending=False)

    percentage= round(total/data.shape[0]*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
# Train missing Percentage

missing_value_of_data(train)
# Test Missing Percentage

missing_value_of_data(test)
def count_values_in_column(data,feature):

    total=data.loc[:,feature].value_counts(dropna=False)

    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)

    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
count_values_in_column(train,'target')
g =sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='winter')

# g.set_xticks(range(len(train))) # <--- set the ticks first

g.set_xticklabels(['Good','Bad'])

plt.show()
# A disaster tweet

disaster_tweets = train[train['target']==1]['text']

disaster_tweets.values[1]
#not a disaster tweet

non_disaster_tweets = train[train['target']==0]['text']

non_disaster_tweets.values[1]
sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20],

            orient='h')
train.loc[train['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
# Replacing the ambigious locations name with Standard names

train['location'].replace({'United States':'USA',

                           'New York':'USA',

                            "London":'UK',

                            "Los Angeles, CA":'USA',

                            "Washington, D.C.":'USA',

                            "California":'USA',

                             "Chicago, IL":'USA',

                             "Chicago":'USA',

                            "New York, NY":'USA',

                            "California, USA":'USA',

                            "FLorida":'USA',

                            "Nigeria":'Africa',

                            "Kenya":'Africa',

                            "Everywhere":'Worldwide',

                            "San Francisco":'USA',

                            "Florida":'USA',

                            "United Kingdom":'UK',

                            "Los Angeles":'USA',

                            "Toronto":'Canada',

                            "San Francisco, CA":'USA',

                            "NYC":'USA',

                            "Seattle":'USA',

                            "Earth":'Worldwide',

                            "Ireland":'UK',

                            "London, England":'UK',

                            "New York City":'USA',

                            "Texas":'USA',

                            "London, UK":'UK',

                            "Atlanta, GA":'USA',

                            "Mumbai":"India"},inplace=True)



sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],

            orient='h')
def unique_values_in_column(data,feature):

    unique_val=pd.Series(data.loc[:,feature].unique())

    return pd.concat([unique_val],axis=1,keys=['Unique Values'])
unique_values_in_column(train,'keyword')
def duplicated_values_data(data):

    dup=[]

    columns=data.columns

    for i in data.columns:

        dup.append(sum(data[i].duplicated()))

    return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])
duplicated_values_data(train)
train.describe()
def find_url(string): 

    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)

    return "".join(text) # converting return value from list to string
train['url']=train['text'].apply(lambda x: find_url(x))
train.head()
def find_emoji(text):

    emo_text=emoji.demojize(text)

    line=re.findall(r'\:(.*?)\:',emo_text)

    return line



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
def find_email(text):

    line = re.findall(r'[\w\.-]+@[\w\.-]+',str(text))

    return ",".join(line)
train['email']=train['text'].apply(lambda x: find_email(x))
def find_hash(text):

    line=re.findall(r'(?<=#)\w+',text)

    return " ".join(line)
train['hash']=train['text'].apply(lambda x: find_hash(x))
# @ David that is mention

def find_at(text):

    line=re.findall(r'(?<=@)\w+',text)

    return " ".join(line)
train['at_mention']=train['text'].apply(lambda x: find_at(x))
# Pick only number from the snetence

def find_number(text):

    line=re.findall(r'[0-9]+',text)

    return " ".join(line)
train['number']=train['text'].apply(lambda x: find_number(x))
def find_phone_number(text):

    line=re.findall(r"\b\d{10}\b",text)

    return "".join(line)
train['phonenumber']=train['text'].apply(lambda x: find_phone_number(x))
def find_year(text):

    line=re.findall(r"\b(19[40][0-9]|20[0-1][0-9]|2020)\b",text)

    return line
train['year']=train['text'].apply(lambda x: find_year(x))
def find_nonalp(text):

    line = re.findall("[^A-Za-z0-9 ]",text)

    return line
train['non_alp']=train['text'].apply(lambda x: find_nonalp(x))
#Retrieve punctuations from sentence.

def find_punct(text):

    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)

    string="".join(line)

    return list(string)
train['punctuation']=train['text'].apply(lambda x : find_punct(x))


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





def text_preprocessing(text):

    """

    Cleaning and parsing the text.



    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text

#
# Applying the cleaning function to both test and training datasets

train['text_clean'] = train['text'].apply(str).apply(lambda x: text_preprocessing(x))
# Applying the cleaning function to both test and training datasets

test['text_clean'] = test['text'].apply(str).apply(lambda x: text_preprocessing(x))
# Analyzing Text statistics



train['text_len'] = train['text_clean'].astype(str).apply(len)

train['text_word_count'] = train['text_clean'].apply(lambda x: len(str(x).split()))
non_disaster_tweets.head()
disaster_tweets.head()
# A disaster tweet

disaster_tweets = train[train['target']==1]

disaster_tweets.values[1]

'Forest fire near La Ronge Sask. Canada'

#not a disaster tweet

non_disaster_tweets = train[train['target']==0]

non_disaster_tweets.values[1]
disaster_tweets.head()
# Sentence length analysis



fig, ax = plt.subplots(1, 2, figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.hist(disaster_tweets['text_len'],bins=50,color='r',alpha=0.5)

plt.title('Disaster Text Length Distribution')

plt.xlabel('text_len')

plt.ylabel('count')





plt.subplot(1, 2, 2)

plt.hist(non_disaster_tweets['text_len'],bins=50,color='g',alpha=0.5)

plt.title('Non Disaster Text Length Distribution')

plt.xlabel('text_len')

plt.ylabel('count')

#source of code : https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d

def get_top_n_words(corpus, n=None):

    """

    List the top n words in a vocabulary according to occurrence in a text corpus.

    """

    vec = CountVectorizer(stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
#Distribution of top unigrams

disaster_tweets_unigrams = get_top_n_words(disaster_tweets['text_clean'],20)

non_disaster_tweets_unigrams = get_top_n_words(non_disaster_tweets['text_clean'],20)





df1 = pd.DataFrame(disaster_tweets_unigrams, columns = ['Text' , 'count'])

df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='r')

plt.ylabel('Count')

plt.title('Top 20 unigrams in Disaster text')

plt.show()



df2 = pd.DataFrame(non_disaster_tweets_unigrams, columns = ['Text' , 'count'])

df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')

plt.title('Top 20 unigram in Non Disaster text')

plt.show()

def get_top_n_gram(corpus,ngram_range,n=None):

    vec = CountVectorizer(ngram_range=ngram_range,stop_words = 'english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
#Distribution of top Bigrams

disaster_tweets_bigrams = get_top_n_gram(disaster_tweets['text_clean'],(2,2),20)

non_disaster_tweets_bigrams = get_top_n_gram(non_disaster_tweets['text_clean'],(2,2),20)



df1 = pd.DataFrame(disaster_tweets_bigrams, columns = ['Text' , 'count'])

df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='r')

plt.ylabel('Count')

plt.title('Top 20 Bigrams in Disaster text')

plt.show()



df2 = pd.DataFrame(non_disaster_tweets_bigrams, columns = ['Text' , 'count'])

df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')

plt.title('Top 20 Bigram in Non Disaster text')

plt.show()
# Finding top trigram

disaster_tweets_trigrams = get_top_n_gram(disaster_tweets['text_clean'],(3,3),20)

non_disaster_tweets_trigrams = get_top_n_gram(non_disaster_tweets['text_clean'],(3,3),20)



df1 = pd.DataFrame(disaster_tweets_trigrams, columns = ['Text' , 'count'])

df1.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='g')

plt.ylabel('Count')

plt.title('Top 20 trigrams in Disaster text')

plt.show()



df2 = pd.DataFrame(non_disaster_tweets_trigrams, columns = ['Text' , 'count'])

df2.groupby('Text').sum()['count'].sort_values(ascending=True).plot(kind='barh',color='red')

plt.title('Top 20 trigram in Non Disaster text')

plt.show()
#Wordclouds

# Wordclouds to see which words contribute to which type of polarity.



from wordcloud import WordCloud

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(disaster_tweets))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster text',fontsize=40);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(non_disaster_tweets))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non Disater text',fontsize=40);



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
# Stemming and Lemmatization examples

text = "feet cats wolves talked"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
# After preprocessing, the text format

def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))

train['text']

train.head()
# text preprocessing function

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
count_vectorizer = CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train['text'])

test_vectors = count_vectorizer.transform(test["text"])



## Keeping only non-zero elements to preserve space 

print(train_vectors[0].todense())
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
# Fitting a simple Logistic Regression on Counts

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

scores



clf.fit(train_vectors, train["target"])
# Fitting a simple Logistic Regression on TFIDF

clf_tfidf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf_tfidf, train_tfidf, train["target"], cv=5, scoring="f1")

scores
# Naives Bayes Classifier

# Well, this is a decent score. Let's try with another model that is said to work well with text data : Naive Bayes.



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
# XGBOOST

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