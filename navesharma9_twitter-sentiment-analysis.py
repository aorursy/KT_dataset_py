import re

import nltk

import string

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



pd.set_option("display.max_colwidth",200)

warnings.filterwarnings("ignore",category=DeprecationWarning)
train_tweets = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')

test_tweets = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')
train_tweets.head()
train_tweets.info()
sns.countplot(data=train_tweets, x='label', hue='label')

plt.title('Types of comments : 0 - > Non Rasict/Sexist , 1 - > Rasict/Sexist')

plt.xlabel('Tweets')

plt.show()
train_tweets[train_tweets['label']==1].head()
train_tweets[train_tweets['label']==0].head()
train_tweets['label'].value_counts()
test_tweets.head()
train_len = train_tweets['tweet'].str.len()

test_len = test_tweets['tweet'].str.len()
print("train data length :" , train_len)

print("test data length :" , test_len)
plt.hist(train_len, bins=20,label='train_tweets')

plt.hist(test_len , bins=20, label='test_tweets')

plt.legend()

plt.show()
dataset = train_tweets.append(test_tweets,ignore_index=True)
dataset.head()
dataset.shape
def remove_pattern(input_text,pattern):

    r = re.findall(pattern, input_text)

    for i in r:

        input_text = re.sub(i,"",input_text)

    return input_text
dataset['tidy_tweet'] = np.vectorize(remove_pattern)(dataset['tweet'],"@[\w]*")
dataset.head()
dataset['tidy_tweet'] = dataset['tidy_tweet'].str.replace('[^a-zA-Z#]'," ")
dataset.head()
stop_words = nltk.corpus.stopwords.words('english')
stop_words[:10]
def remove_stopword(input_text):

    txt_clean = " ".join([word for word in input_text.split() if len(word)>3])

    return txt_clean
dataset['tidy_tweet'] = dataset['tidy_tweet'].apply(lambda x:remove_stopword(x))
dataset.head()
tokenized_tweet = dataset['tidy_tweet'].apply(lambda x: x.split())

tokenized_tweet.head()
from nltk.stem import PorterStemmer
pstem = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x:[pstem.stem(i) for i in x])
tokenized_tweet
for i in range(len(tokenized_tweet)):

    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

dataset['tidy_tweet'] = tokenized_tweet
dataset.head()
from wordcloud import WordCloud

all_words = ' '.join([text for text in dataset['tidy_tweet']])  

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
all_words = ' '.join([text for text in dataset['tidy_tweet'][dataset['label']==0]])  

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
all_words = ' '.join([text for text in dataset['tidy_tweet'][dataset['label']==1]])  

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 

plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
bow_vector = CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')

bow = bow_vector.fit_transform(dataset['tidy_tweet'])

bow.shape
bow.data
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split 

from sklearn.metrics import f1_score
X = bow[:31962,:]
y = bow[31962:,:]
x_train,x_test,y_train,y_test = train_test_split(X,train_tweets['label'],test_size=0.3)
lg = LogisticRegression()
lg.fit(x_train,y_train)
pred = lg.predict_proba(x_test)
pred
pred_int = pred[:,1]>=0.3
pred_int = pred_int.astype(np.int)
pred_int
f1_score(y_test,pred_int)