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
# Fetching training data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()
# Fetching testing data

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

test.head()
# Checking the dimensions of datasets

print(train.shape)

print(test.shape)
# Checking missing values in both train and test set

print(train.isnull().sum())

print(test.isnull().sum())
# Filling the missing values of "keyword" as Unknown so that those can be counted as well

train.keyword.fillna("Unknown", inplace=True)

test.keyword.fillna("Unknown", inplace=True)
# Dropping the non-required fields i.e. location and id from the dataset

train.drop("location", axis=1, inplace=True)

test.drop("location", axis=1, inplace=True)

train.drop("id", axis=1, inplace=True)

test.drop("id", axis=1, inplace=True)
# Final check to verify the missing values

print(train.isnull().sum())

print(test.isnull().sum())
# Importing stopwords from NLTK

import nltk

nltk.download('stopwords')
# Pre-processing the text with the removal of irrelevant characters, symbols and stopwords

import re

import string

from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):

    """

        text: a string   

        return: modified initial string

    """

    text = text.lower() #lowercase text  

    text = REPLACE_BY_SPACE_RE.sub(' ',text) #replace REPLACE_BY_SPACE_RE symbols by space in text    

    text = BAD_SYMBOLS_RE.sub('',text) #delete symbols which are in BAD_SYMBOLS_RE from text 

    text = re.sub('https','',text)

    text = re.sub('http','',text)

    text = re.sub('tco','',text)

    temp = [s.strip() for s in text.split() if s not in STOPWORDS] #delete stopwords from text

    new_text = ''

    for i in temp:

        new_text +=i+' '

    text = new_text

    return text.strip()
# Applying the above preprocessing to the train set

train['text'] = train['text'].map(text_prepare)

train['text'].head()
# Applying the above preprocessing to the test set

test['text'] = test['text'].map(text_prepare)

test['text'].head()
# Cleaning the text using tokens and stemming

stopword =  nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()

def clean(text):

    no_punct = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+',no_punct) #tokenization

    text_stem = ([ps.stem(word) for word in tokens if word not in stopword]) #stemming

    return text_stem
import seaborn as sns

sns.countplot(train.target)
from wordcloud import WordCloud

import matplotlib.pyplot as plt



# Common words in the tweets

plt.figure(figsize = (16,24))

wordcloud = WordCloud(min_font_size = 5,  max_words = 500 , width = 1800 , height = 1000, stopwords= STOPWORDS).generate(" ".join(train['text']))

plt.imshow(wordcloud,interpolation = 'bilinear')
# Common words in the disaster tweets

disaster = train.text[train.target[train.target==1].index]

plt.figure(figsize = (16,24))

wordcloud = WordCloud(min_font_size = 5,  max_words = 500 , width = 1800 , height = 1000, stopwords= STOPWORDS).generate(" ".join(disaster))

plt.imshow(wordcloud,interpolation = 'bilinear')
# Common words in the non-disaster tweets

ndisaster = train.text[train.target[train.target==0].index]

plt.figure(figsize = (16,24))

wordcloud = WordCloud(min_font_size = 5,  max_words = 500 , width = 1800 , height = 1000, stopwords= STOPWORDS).generate(" ".join(ndisaster))

plt.imshow(wordcloud,interpolation = 'bilinear')
# Tweet words frequency plot

train_new = train.copy(deep=True)

train_new['words'] = train_new['text'].apply(lambda x: clean(x))

All_words = []

for words in train_new['words']:

    for word in words:

            All_words.append(word)

All_words_freq = nltk.FreqDist(All_words)

Freq_word_DF = pd.DataFrame({"Data":All_words_freq.most_common(15)})

Freq_word_DF['Words'] = Freq_word_DF['Data'].apply(lambda x : x[0])

Freq_word_DF['freq'] = Freq_word_DF['Data'].apply(lambda x : x[1])

sns.set()

sns.set(style='whitegrid', rc={"grid.linewidth": 0.2})

sns.set_context("paper", font_scale=2)

fig=plt.figure(figsize =(20,8),dpi=50)

sns.barplot('Words','freq',data = Freq_word_DF)
# Number of characters in tweets

def length(text):    

    '''a function which returns the length of text'''

    return len(text)

train_new['length'] = train_new['text'].apply(length)

plt.rcParams['figure.figsize'] = (18.0, 6.0)

bins = 150

plt.hist(train_new[train_new['target'] == 0]['length'], alpha = 0.6, bins=bins, label='Non-Disaster')

plt.hist(train_new[train_new['target'] == 1]['length'], alpha = 0.8, bins=bins, label='Disaster')

plt.xlabel('length')

plt.ylabel('numbers')

plt.legend(loc='upper right')

plt.xlim(0,150)

plt.grid()

plt.show()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=train_new[train_new['target']==1]['text'].str.len()

ax1.hist(tweet_len,color='blue')

ax1.set_title('disaster tweets')

tweet_len=train_new[train_new['target']==0]['text'].str.len()

ax2.hist(tweet_len,color='red')

ax2.set_title('Not disaster tweets')

fig.suptitle('Characters in tweets')

plt.show()
# Number of words in a tweet

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len=train_new[train_new['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='blue')

ax1.set_title('disaster tweets')

tweet_len=train_new[train_new['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='red')

ax2.set_title('Not disaster tweets')

fig.suptitle('Words in a tweet')

plt.show()
# Average word length in a tweet

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

word=train_new[train_new['target']==1]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='blue')

ax1.set_title('disaster')

word=train_new[train_new['target']==0]['text'].str.split().apply(lambda x : [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')

ax2.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')
# Performing text vectorization so that data can be fed to the model

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf = TfidfVectorizer(analyzer= clean) #Using the above text cleaning to clean the data

Xtf_idfVector = tf_idf.fit_transform(train['text'])

Xtest_idfVector = tf_idf.transform(test['text'])
# Performing one hot encoding for categorical variables

train_mod = pd.get_dummies(data=train, columns=['keyword'])

test_mod = pd.get_dummies(data=test, columns=['keyword'])
# Dropping irrelevant columns from the modified data

train_mod.drop('text', axis=1, inplace=True)

train_mod.drop('target', axis=1, inplace=True)

test_mod.drop('text', axis=1, inplace=True)
# Creating the final dataframe consisting of keywords and text i.e. X Variable

Xfeatures_data = pd.concat([train_mod, 

                            pd.DataFrame(Xtf_idfVector.toarray())], axis = 1)

X_test = pd.concat([test_mod, 

                    pd.DataFrame(Xtest_idfVector.toarray())], axis = 1)



# Target variable i.e. Y Variable

y_data = train.target
# Using randomforest classifier to train the data

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators= 50, max_depth= 20, n_jobs= -1)

model1 = rf.fit(Xfeatures_data,y_data)



#Predict Output using RF

predicted_RF= rf.predict(X_test) 

predicted_RF
# Using naive_bayes classifier to train the data

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

model2 = nb.fit(Xfeatures_data,y_data)



#Predict Output using NB

predicted_NB= nb.predict(X_test) 

predicted_NB
# Using xgboost classifier to train the data

from xgboost import XGBClassifier

xgb = XGBClassifier()

model3 = xgb.fit(Xfeatures_data,y_data)



#Predict Output using XGB

predicted_XG= xgb.predict(X_test) 

predicted_XG
# Using SVC classifier to train the data

from sklearn.svm import SVC

svc = SVC()

model4 = svc.fit(Xfeatures_data,y_data)



#Predict Output using SVC

predicted_SC= svc.predict(X_test) 

predicted_SC
# Using Voting classifier to train the data

from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[('SVC', svc), ('XGB', xgb), ('NB', nb), ('RF', rf)], voting='hard')

model5 = voting_clf.fit(Xfeatures_data,y_data)



#Predict Output using Voting

predicted_VC= voting_clf.predict(X_test) 

predicted_VC
# Using Neural_networks to train the data

from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, Embedding, LSTM

from keras.models import Sequential

from keras.utils import to_categorical

model = Sequential()

model.add(Dense(units = 512 , activation = 'relu' , input_dim = Xfeatures_data.shape[1]))

model.add(Dense(units = 256 , activation = 'relu'))

model.add(Dense(units = 100 , activation = 'relu'))

model.add(Dense(units = 10 , activation = 'relu'))

model.add(Dense(units = 1 , activation = 'sigmoid'))

model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model6 = model.fit(Xfeatures_data,y_data, batch_size=128, epochs=10)



#Predict Output using NN

predicted_NN= model.predict_classes(X_test) 

predicted_NN
#Saving output results of NB to csv file

predicted_df = pd.DataFrame(predicted_NB)

predicted_df.to_csv('out.csv')