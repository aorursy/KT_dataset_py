# necessary imports 

import numpy as np

import pandas as pd 

import re

import string

import matplotlib.pyplot as plt 

import seaborn as sns 



import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.tokenize import TweetTokenizer



from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
# load the data

df = pd.read_csv('../input/nlp-getting-started/train.csv',index_col='id')

df.head()
# check null, type, size 

df.info()
# class distribution

sns.catplot(kind='count',data=df,x='target',aspect=3)

plt.show()
# drop unnecessary columns

df.drop(['location','keyword'],inplace=True,axis=1)

df.head
def process_tweet(tweet):

    stopwords_english = set(stopwords.words('english'))

    stemmer = PorterStemmer()

    

    tweet = re.sub(r'\$\w*','',tweet) # remove words with pattern as $word

    tweet = re.sub(r'^RT[\s]+','',tweet) # remove retweet text "RT"

    tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet) # remove hyperlinks

    tweet = re.sub(r'#','',tweet) # remove #tag sign

    

    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True, reduce_len=True)

    tokens = tokenizer.tokenize(tweet)

    

    clean = []

    for word in tokens:

        if( word not in stopwords_english and word not in string.punctuation):

            stem_word = stemmer.stem(word)

            clean.append(stem_word)

            

    return clean
# Instead of using tf-idf vectorizer or count vectorizer directly, we want to bring all this in 3 dimensional features

# Hence, we want features like (bias term, word in positive tweet, word in negative tweet)

# For that, we can get help from build_freqs() function given below.

def build_freqs(tweets,targets):

    yslist= np.squeeze(targets).tolist()

    

    freqs = {}

    for target, tweet in zip(targets,tweets):

        for word in process_tweet(tweet):

            pair = (word, target)

            freqs[pair] = freqs.get(pair,0) + 1

    return freqs
X = df['text'].values

y = df['target'].values

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,stratify=y,random_state=43) # stratify makes sure distribution is even
freqs = build_freqs(X_train,y_train) # frequencies of word associated with label
# transform word in tweet to (bias, word associated with +ve, word associated with -ve label)



def extract_features(tweet,freqs):

    word_l = process_tweet(tweet)

    x = np.zeros((1,3)) # feature vector

    x[0,0] = 1 # for bias

    

    for word in word_l:

        x[0,1] += freqs.get((word,1),0)

        x[0,2] += freqs.get((word,0),0)

    return x
def train(x,y,xVal,yVal,freqs):

    xTRAIN = np.zeros((len(x),3))

    for i in range(len(x)):

        xTRAIN[i,:] = extract_features(x[i],freqs)

        

    model = SGDClassifier(loss='log',n_jobs=-1,max_iter=800,random_state=31) # 31

    model.fit(xTRAIN,y)

    

    # training f1-score: 

    preds = model.predict(xTRAIN)

    print('Training f1-score is {}'.format(f1_score(y,preds)))

    

    # validation f1-score:

    xVAL = np.zeros((len(xVal),3))

    for i in range(len(xVal)):

        xVAL[i,:] = extract_features(xVal[i],freqs)

    

    preds = model.predict(xVAL)

    print('Validation f1-score is {}'.format(f1_score(yVal,preds)))

    return model

    

model = train(X_train,y_train,X_val, y_val, freqs)
# load test set

tf = pd.read_csv('../input/nlp-getting-started/test.csv',index_col='id')

tf.drop(['keyword','location'],axis=1,inplace=True)



def predict_tweet(model,tweet, freqs):

    x = extract_features(tweet,freqs)

    pred = model.predict(x)

    return pred



target = []

xTest = tf['text'].values

for i in range(len(xTest)):

    target.append(predict_tweet(model,xTest[i],freqs)[0])
sub = pd.DataFrame({'id':tf.index, 'target':target}) 

# Prediction counts

sns.catplot(kind='count',y='target',data=sub)

plt.show()
# save submission file

sub.to_csv('submission.csv', index=False)