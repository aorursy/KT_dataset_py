%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



# Please write all the code with proper documentation

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV

import xgboost as xgb



from sklearn.metrics import confusion_matrix

from sklearn import metrics



from nltk.stem.porter import PorterStemmer

from sklearn import tree

import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Training data

train = pd.read_csv('../input/nlp-getting-started/train.csv')

print('Training data shape: ', train.shape)

test = pd.read_csv('../input/nlp-getting-started/test.csv')

print('Testing data shape: ', test.shape)

train.head(5)
test.head()
#Missing values in training set

print("Missing values in train data")

train.isnull().sum()

#Missing values in testing set

print("Missing values in test data")

test.isnull().sum()
train['target'].value_counts()
sns.barplot(train['target'].value_counts().index,train['target'].value_counts(),palette='rocket')
train_positive=train[train['target']==1]['text']

train_positive.values[120]
train_positive.values[125]
train_negative=train[train['target']==0]['text']

train_negative.values[12]
train_negative.values[11]
train['keyword'].value_counts()
sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20],orient='h')

plt.show()
train.loc[train['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
train['location'].value_counts()
sns.barplot(y=train['location'].value_counts()[:20].index,x=train['location'].value_counts()[:20],orient='h')

plt.show()
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



sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],orient='h')

plt.show()
# https://stackoverflow.com/questions/16206380/python-beautifulsoup-how-to-remove-all-tags-from-an-element

# Refer my Github for more similar examples

# Below is a genric function which can be found at a lot of surces to clean text

from bs4 import BeautifulSoup

# https://www.kaggle.com/parulpandey/getting-started-with-nlp-a-general-intro

import re

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



# Let's take a look at the updated text

print('Text after cleaning')

train['text'][120]

eng_stopwords = set(stopwords.words("english"))

train["num_stopwords"] = train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.violinplot(x = 'target', y = 'num_stopwords', data = train)

plt.subplot(1,2,2)

sns.distplot(train[train['target'] == 1.0]['num_stopwords'][0:] , label = "1", color = 'red')

sns.distplot(train[train['target'] == 0.0]['num_stopwords'][0:] , label = "0" , color = 'blue' )

plt.legend()

plt.show()
train["num_words"] = train["text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(12, 8))

plt.subplot(1,2,1)

sns.violinplot(x = 'target', y = 'num_words', data = train[0:])

plt.subplot(1,2,2)

sns.distplot(train[train['target'] == 1.0]['num_words'][0:] , label = "1", color = 'red')

sns.distplot(train[train['target'] == 0.0]['num_words'][0:] , label = "0" , color = 'blue' )

plt.legend()

plt.show()
from wordcloud import WordCloud



wordcloud = WordCloud( background_color='black',

                        width=600,

                        height=400).generate(" ".join(train_positive))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Disaster Tweets',fontsize=40);



wordcloud = WordCloud( background_color='black',

                        width=600,

                        height=400).generate(" ".join(train_negative))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Non Disaster Tweets',fontsize=40);
# https://stackoverflow.com/a/47091490/4084039

import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase



stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
# Combining all the above stundents 

from tqdm import tqdm

preprocessed_reviews = []

# tqdm is for printing the status bar

for sentance in tqdm(train['text'].values):

    sentance = re.sub(r"http\S+", "", sentance)

    sentance = BeautifulSoup(sentance, 'lxml').get_text()

    sentance = decontracted(sentance)

    sentance = re.sub("\S*\d\S*", "", sentance).strip()

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # https://gist.github.com/sebleier/554280

    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)

    preprocessed_reviews.append(sentance.strip())

X=preprocessed_reviews[:]

y=train['target'][:]

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.30, random_state=42)
# bow is bag of words

bow = CountVectorizer()

train_vectors = bow.fit_transform(train['text'])

test_vectors = bow.transform(test["text"])
# Naive Bayes

# It is generally used as a benchmark for many NLP tasks



clf = MultinomialNB()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of Naive Bayes: ')

print(scores)



# Fitting a simple Logistic Regression on BOW

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('Logistic Regression : ')

print(scores)

# Fitting a simple Decision Trees on BOW

clf = tree.DecisionTreeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of Decision Trees')

print(scores)

# Fitting a simple Logistic Regression on Counts

clf = RandomForestClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of Random Forests')

print(scores)



clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of XGBoost')

print(scores)
# bow is bag of words

bow = CountVectorizer(ngram_range=(1, 2))

train_vectors = bow.fit_transform(train['text'])

test_vectors = bow.transform(test["text"])
# Naive Bayes

# It is generally used as a benchmark for many NLP tasks



clf = MultinomialNB()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of Naive Bayes: ')

print(scores)



# Fitting a simple Logistic Regression on BOW

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('Logistic Regression : ')

print(scores)

# Fitting a simple Decision Trees on BOW

clf = tree.DecisionTreeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of Decision Trees')

print(scores)

# Fitting a simple Logistic Regression on Counts

clf = RandomForestClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of Random Forests')

print(scores)



clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")

print('scores of XGBoost')

print(scores)
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tfidf = tfidf.fit_transform(train['text'])

test_tfidf = tfidf.transform(test["text"])
# Naive Bayes

# It is generally used as a benchmark for many NLP tasks



clf = MultinomialNB()

scores = model_selection.cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")

print('scores of Naive Bayes: ')

print(scores)



# #Fitting a simple Logistic Regression on BOW

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")

print('Logistic Regression : ')

print(scores)

## Fitting a simple Decision Trees on BOW

clf = tree.DecisionTreeClassifier()

scores = model_selection.cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")

print('scores of Decision Trees')

print(scores)

# #Fitting a simple Logistic Regression on Counts

clf = RandomForestClassifier()

scores = model_selection.cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")

print('scores of Random Forests')

print(scores)



clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(clf, train_tfidf, train["target"], cv=5, scoring="f1")

print('scores of XGBoost')

print(scores)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

# fix random seed for reproducibility

np.random.seed(7)
X=train['text']

y=train['target'][:]

X_train=X[:6500]

X_test=X[6500:]

y_train=y[:6500]

y_test=y[6500:]
max(train['text'].apply(lambda x: len(x)))
np.mean(train['text'].apply(lambda x: len(x)))
print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(X_train)



X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

test=tokenizer.texts_to_sequences(test['text'])
## Zero Padding

max_review_length = 157

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
test = sequence.pad_sequences(test, maxlen=max_review_length)
print(X_train[45])
# create the model

top_words=10000

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())
def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=2, validation_data=(X_test, y_test))







score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])
# fig,ax = plt.subplots(1,1)

# ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# # list of epoch numbers

# x = list(range(1,16))

# vy = history.history['val_loss']

# ty = history.history['loss']

# plt_dynamic(x, vy, ty, ax)

# create the model

top_words=30000

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))



print(model.summary())

epoch=15

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=epoch, verbose=2, validation_data=(X_test, y_test))







score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epoch+1))

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)

# create the model

top_words=30000

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))

model.add(LSTM(128))

model.add(Dense(1, activation='relu'))



print(model.summary())

epoch=5

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=epoch, verbose=2, validation_data=(X_test, y_test))



score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epoch+1))

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
# create the model

from keras.layers import Bidirectional

from keras.layers import Dropout

top_words=30000

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))



model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(1, activation='sigmoid'))



# model.add(LSTM(128))

# model.add(Dense(1, activation='relu'))



print(model.summary())

epoch=5

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=epoch, verbose=2, validation_data=(X_test, y_test))



score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epoch+1))

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
# create the model

from keras.layers import Bidirectional

from keras.layers import Dropout

top_words=30000

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))



model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(1, activation='sigmoid'))



# model.add(LSTM(128))

# model.add(Dense(1, activation='relu'))



print(model.summary())

epoch=5

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=epoch, verbose=2, validation_data=(X_test, y_test))



score = model.evaluate(X_test, y_test, verbose=0) 

print('Test score:', score[0]) 

print('Test accuracy:', score[1])



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epoch+1))

vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
# clf_nb = MultinomialNB()

# scores = model_selection.cross_val_score(clf_nb, train_tfidf, train["target"], cv=5, scoring="f1")

# print('scores of Naive Bayes: ')

# print(scores)
# clf_nb.fit(train_tfidf, train["target"])
def submission(submission_file_path,model,test_vectors):

    sample_submission = pd.read_csv(submission_file_path)

    sample_submission["target"] = model.evaluate(test_vectors)

    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

sample_submission = pd.read_csv(submission_file_path)

sample_submission["target"] = model.predict_classes(test)

sample_submission.to_csv("submission.csv", index=False)
# submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

# test_vectors=test

# submission(submission_file_path,model,test_vectors)