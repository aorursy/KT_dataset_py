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
import matplotlib.pyplot as plt

import seaborn as sns
tweets_data = pd.read_csv("../input/twitter-sentiment-analysis-hatred-speech/train.csv")
tweets_data.head()
tweets_data.shape
tweets_data.describe()
tweets_data.info()
tweets_data['tweet']
tweets_data = tweets_data.drop(['id'], axis = 1)
tweets_data
tweets_data.isnull()
sns.heatmap(tweets_data.notnull(), yticklabels = False, cbar = False, cmap = "Blues")
#Plot a sns countplot graph for total counting of positive (0) and negative (1) tweets

sns.countplot(tweets_data['label'], label = 'Count')
tweets_data['length'] = tweets_data['tweet'].apply(len)
tweets_data
tweets_data['length'].plot(bins = 100, kind = 'hist')
tweets_data.describe()
tweets_data[tweets_data['length'] == 11].iloc[0]
tweets_data[tweets_data['length'] == 84].iloc[0]
pos_tweets = tweets_data[tweets_data['label'] == 0]

pos_tweets
neg_tweets = tweets_data[tweets_data['label'] == 1]

neg_tweets
sentences = tweets_data['tweet'].tolist()
sentences
len(sentences)
sen_one_string = " ".join(sentences)
sen_one_string
from wordcloud import WordCloud
plt.figure(figsize = (20,20))

plt.imshow(WordCloud().generate(sen_one_string))

neg_list = neg_tweets['tweet'].tolist()

neg_one_string = " ".join(neg_list)

plt.figure(figsize = (20,20))

plt.imshow(WordCloud().generate(neg_one_string))
pos_list = pos_tweets['tweet'].tolist()

pos_one_string = " ".join(pos_list)

plt.figure(figsize = (20,20))

plt.imshow(WordCloud().generate(pos_one_string))
import string

string.punctuation
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
def clean(sen):

    punc_remov = [char for char in sen if char not in string.punctuation]

    punc_remov_join = ''.join(punc_remov)

    final_clean = [word for word in punc_remov_join.split() if word.lower() not in stopwords.words('english')]

    return final_clean
tweets_clean = tweets_data['tweet'].apply(clean)
print(tweets_data['tweet'][5])

print(tweets_clean[5])
vectorizer = CountVectorizer(analyzer = clean)

tweets_vectorizer = CountVectorizer(analyzer = clean, dtype = 'uint8').fit_transform(tweets_data['tweet']).toarray()
tweets_vectorizer.shape
X = tweets_vectorizer

y = tweets_data['label']
X.shape

y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()

NB_classifier.fit(X_train, y_train)

                            
from sklearn.metrics import classification_report, confusion_matrix

y_predict_test = NB_classifier.predict(X_test)

cm = confusion_matrix(y_test, y_predict_test)

sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_predict_test))