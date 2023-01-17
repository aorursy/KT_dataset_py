import random



import numpy as np

import pandas as pd



import tensorflow as tf

import tensorflow_hub as hub



import keras

from keras.layers import Input, Dense, LeakyReLU, Dropout, Softmax

from keras.models import Model

from keras.utils import to_categorical



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



random.seed(42)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

print (train, test, submission)

train.head()
train.duplicated().sum()
sns.countplot(y=train.target);
train.isnull().sum()
top_d = train.groupby('keyword').mean()['target'].sort_values(ascending=False).head(10)

top_nd = train.groupby('keyword').mean()['target'].sort_values().head(10)



plt.figure(figsize=(13,5))

plt.subplot(121)

sns.barplot(top_d, top_d.index, color='pink')

plt.title('Keywords with highest % of disaster tweets')

plt.subplot(122)

sns.barplot(top_nd, top_nd.index, color='yellow')

plt.title('Keywords with lowest % of disaster tweets')

plt.show()
train["text"].head(10)
import numpy as np

import pandas as pd 

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import f1_score



import warnings

warnings.filterwarnings('ignore')
corpus  = []

pstem = PorterStemmer()

for i in range(train['text'].shape[0]):

    #Remove unwanted words

    tweet = re.sub("[^a-zA-Z]", ' ', train['text'][i])



    #Transform words to lowercase

    tweet = tweet.lower()

    tweet = tweet.split()

    #Remove stopwords then Stemming it

    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet = ' '.join(tweet)

    #Append cleaned tweet to corpus

    corpus.append(tweet)

    

print("Corpus created successfully")
print(pd.DataFrame(corpus)[0].head(10))
counVec = CountVectorizer(max_features = 1000)

bagOfWords = counVec.fit_transform(corpus).toarray()
X_train= bagOfWords

y_train= train['target']

print("X shape = ",X_train.shape)

print("y shape = ",y_train.shape)
model = LogisticRegression = LogisticRegression(penalty='l2', solver='saga', random_state = 55)  



model.fit(X_train,y_train)



print("LogisticRegression Classifier model run successfully")
print ('Training accuracy: %.4f' % model.score(X_train, y_train))
corpus  = []

pstem = PorterStemmer()

for i in range(test['text'].shape[0]):

    #Remove unwanted words

    tweet = re.sub("[^a-zA-Z]", ' ', train['text'][i])

    #Transform words to lowercase

    tweet = tweet.lower()

    tweet = tweet.split()

    #Remove stopwords then Stemming it

    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]

    tweet = ' '.join(tweet)

    #Append cleaned tweet to corpus

    corpus.append(tweet)

    

print("Corpus created successfully")
print(pd.DataFrame(corpus)[0].head(10))
counVec = CountVectorizer(max_features = 1000)

bagOfWords = counVec.fit_transform(corpus).toarray()
X_test = bagOfWords

print(X_test.shape)



y_test = model.predict(X_test)
print(type(y_test))
print(y_test.sum())
submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target'] = y_test.round().astype(int)

submission.to_csv('sample_submission.csv', index=False)
submission.head(30)