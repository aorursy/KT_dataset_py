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
import nltk

from os import getcwd

import re

import string

import numpy as np



from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer

from nltk.corpus import twitter_samples 
# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path

# this enables importing of these files without downloading it again when we refresh our workspace



filePath = f"{getcwd()}/../tmp2/"

nltk.data.path.append(filePath)
# select the set of positive and negative tweets

all_positive_tweets = twitter_samples.strings('positive_tweets.json')

all_negative_tweets = twitter_samples.strings('negative_tweets.json')
# split the data into two pieces, one for training and one for testing (validation set) 

test_pos = all_positive_tweets[4000:]

train_pos = all_positive_tweets[:4000]

test_neg = all_negative_tweets[4000:]

train_neg = all_negative_tweets[:4000]



train_x = train_pos + train_neg 

test_x = test_pos + test_neg
# combine positive and negative labels

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)

test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
train_x[0]
import pandas as pd 

from pandas import DataFrame 



data_ = [train_x, train_y]

data = DataFrame (data_).transpose()

data.columns = ['body_text', 'label']

print(data)
#Function to remove Punctuation

def remove_punct(text):

    text_nopunct = "".join([char for char in text if char not in string.punctuation])# It will discard all punctuations

    return text_nopunct



data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))



data.head()
import re



# Function to Tokenize words

def tokenize(text):

    tokens = re.split('\W+', text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.

    return tokens



data['body_text_tokenized'] = data['body_text_clean'].apply(lambda x: tokenize(x.lower())) 

#We convert to lower as Python is case-sensitive. 



data.head()
import nltk

stopword = nltk.corpus.stopwords.words('english')# All English Stopwords
# Function to remove Stopwords

def remove_stopwords(tokenized_list):

    text = [word for word in tokenized_list if word not in stopword]# To remove all stopwords

    return text



data['body_text_nostop'] = data['body_text_tokenized'].apply(lambda x: remove_stopwords(x))



data.head()
ps = nltk.PorterStemmer()



def stemming(tokenized_text):

    text = [ps.stem(word) for word in tokenized_text]

    return text



data['body_text_stemmed'] = data['body_text_nostop'].apply(lambda x: stemming(x))



data.head()
nltk.download('wordnet')

wn = nltk.WordNetLemmatizer()



def lemmatizing(tokenized_text):

    text = [wn.lemmatize(word) for word in tokenized_text]

    return text



data['body_text_lemmatized'] = data['body_text_nostop'].apply(lambda x: lemmatizing(x))



data.head(10)
import pandas as pd

import re

import string

pd.set_option('display.max_colwidth', 100) # To extend column width





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopword]

    return text
train = ['Today i feel so good', 'Not bad', 'It is ok']

data__ = [train]

data_train = DataFrame (data__).transpose()

data_train.columns = ['train_data']

print(data_train)
from sklearn.feature_extraction.text import CountVectorizer



count_vect = CountVectorizer(analyzer=clean_text)

X_counts = count_vect.fit_transform(data_train['train_data'])

print(X_counts.shape)

print(count_vect.get_feature_names())
X_counts_df = pd.DataFrame(X_counts.toarray(), columns=count_vect.get_feature_names())

X_counts_df.head()
ngram_vect = CountVectorizer(ngram_range=(2,2),analyzer=clean_text) # It applies only bigram vectorizer

X_counts = ngram_vect.fit_transform(data_train['train_data'])

print(X_counts.shape)

print(ngram_vect.get_feature_names())

X_counts_df = pd.DataFrame(X_counts.toarray(), columns=ngram_vect.get_feature_names())

X_counts_df.head()
from sklearn.feature_extraction.text import CountVectorizer

ps = nltk.PorterStemmer()



count_vect = CountVectorizer(analyzer=clean_text)

count_vect_fit = count_vect.fit(train_x)

count_vect_train = count_vect_fit.transform(train_x)

count_vect_test = count_vect_fit.transform(test_x)



train_x_vect = pd.DataFrame(count_vect_train.toarray())

test_x_vect = pd.DataFrame(count_vect_test.toarray())



train_x_vect.head()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import accuracy_score as acs

import matplotlib.pyplot as plt

import seaborn as sns
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



rf_model = rf.fit(train_x_vect, train_y)



pred_y = rf_model.predict(test_x_vect)



precision, recall, fscore, train_support = score(test_y, pred_y, pos_label=1.0, average='binary')

print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(

    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(test_y,pred_y), 3)))



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, pred_y)

class_label = ["ham", "spam"]

df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)

sns.heatmap(df_cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
from sklearn.feature_extraction.text import CountVectorizer



ngram_vect = CountVectorizer(ngram_range=(2,2),analyzer=clean_text) # It applies only bigram vectorizer

ngram_vect_fit = ngram_vect.fit(train_x)

ngram_vect_train = ngram_vect_fit.transform(train_x)

ngram_vect_test = ngram_vect_fit.transform(test_x)



train_x_vect = pd.DataFrame(ngram_vect_train.toarray())

test_x_vect = pd.DataFrame(ngram_vect_test.toarray())



train_x_vect.head()
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



rf_model = rf.fit(train_x_vect, train_y)



pred_y = rf_model.predict(test_x_vect)



precision, recall, fscore, train_support = score(test_y, pred_y, pos_label=1.0, average='binary')

print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(

    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(test_y,pred_y), 3)))



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, pred_y)

class_label = ["ham", "spam"]

df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)

sns.heatmap(df_cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

tfidf_vect_fit = tfidf_vect.fit(train_x)

tfidf_vect_train = tfidf_vect_fit.transform(train_x)

tfidf_vect_test = tfidf_vect_fit.transform(test_x)



train_x_vect = pd.DataFrame(tfidf_vect_train.toarray())

test_x_vect = pd.DataFrame(tfidf_vect_test.toarray())



train_x_vect.head()
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



rf_model = rf.fit(train_x_vect, train_y)



pred_y = rf_model.predict(test_x_vect)



precision, recall, fscore, train_support = score(test_y, pred_y, pos_label=1.0, average='binary')

print('Precision: {} / Recall: {} / F1-Score: {} / Accuracy: {}'.format(

    round(precision, 3), round(recall, 3), round(fscore,3), round(acs(test_y,pred_y), 3)))



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, pred_y)

class_label = ["ham", "spam"]

df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)

sns.heatmap(df_cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()