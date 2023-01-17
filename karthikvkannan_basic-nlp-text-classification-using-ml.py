import pandas as pd 
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

print(train.info())
print('\n \n')
print(test.info())
print('Number of null values in train set')
print(train.isnull().sum())
print('\n\n')
print('Number of null values in test')
print(test.isnull().sum())
train.dropna(subset=['keyword'],inplace=True)
test.dropna(subset=['keyword'],inplace=True)
train.dropna(axis=1,inplace=True)
test.dropna(axis=1,inplace=True)
print(train.info())
print('\n \n')
print(test.info())
print(train.describe(include='all'))
print('\n\n')
print(test.describe(include='all'))
print(train['keyword'].value_counts())
print('\n\n')
print(test['keyword'].value_counts())
for i in ['fatalities','deluge','armageddon','harm','damage','rubble','snowstorm','demolished','forest%20fire']:
  keyword_list = list(train.loc[train['keyword']==i]['target'])
  import matplotlib.pyplot as plt
  print(i)
  plt.hist(keyword_list)
  plt.show()


import re
def stemming_keyword(text):
  text = re.sub('deluged','deluge',text)
  return text
train['keyword'] = train['keyword'].apply(lambda x:stemming_keyword(x))
test['keyword'] = test['keyword'].apply(lambda x:stemming_keyword(x))
def len_(text):
  length = len(text)
  return length
train['len_text'] = train['text'].apply(lambda x:len_(x))
test['len_text'] = test['text'].apply(lambda x:len_(x))
for i in [10,20,max(train['len_text'])]:
  len_t = list(train.loc[train['len_text'] < i]['target'])
  plt.hist(len_t)
  plt.show()

from textblob import TextBlob
import re
def clean_text(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

def get_tweet_sentiment(text): 
    analysis = TextBlob(clean_text(text)) 
    return analysis.sentiment.polarity

def sentiment_text(text):
    parsed_text = get_tweet_sentiment(text)  

    return parsed_text

train['sent_text'] = train['text'].apply(lambda x:sentiment_text(x))
test['sent_text'] = test['text'].apply(lambda x:sentiment_text(x))
for i in [0.9,0,-1]:
  sent = list(train.loc[train['sent_text'] > i]['target'])
  plt.hist(sent)
  plt.show()

print(list(train.loc[train['sent_text']==1.0]['text'])[0:10])
print('\n\n')
print(list(train.loc[train['sent_text']==-1.0]['text'])[0:10])

import re
import nltk
from nltk.corpus import wordnet
def tokenize(text):
        from nltk.tokenize import TweetTokenizer
        tt = TweetTokenizer()
        token = tt.tokenize(text.lower())
        return token

train['texts'] = train['text'].apply(lambda x : tokenize(x))
test['texts'] = test['text'].apply(lambda x : tokenize(x))

import nltk
stpw = nltk.corpus.stopwords.words('english')
def rmstp(text):
    tt = [char for char in text if char not in stpw and len(char) > 3 and wordnet.synsets(char)]
    return tt
train['texts'] = train['texts'].apply(lambda x:rmstp(x))
test['texts'] = test['texts'].apply(lambda x : rmstp(x))
print(train.head(10))
print('\n\n')
print(test.head(10))
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk import word_tokenize, pos_tag

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
def rmstmer(text):
    tt = [lemmatizer.lemmatize(char.lower(),tag_map[j[0]]) for char,j in pos_tag(text) if len(char) >= 3]
    return tt
train['texts'] = train['texts'].apply(lambda x:rmstmer(x))
test['texts'] = test['texts'].apply(lambda x:rmstmer(x))
print(train.head())
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])
print('Shape of train vectorizer:',end='')
print(train_tfidf.shape)
print('\n\n')
print('Shape of test vectorizer:',end='')
print(test_tfidf.shape)
dict_k={}
unq_k = set(train['keyword'])
for i,k in enumerate(unq_k):
  dict_k[k] = i

train['keyword'] = train['keyword'].map(dict_k)

dict_k={}
unq_k = set(test['keyword'])
for i,k in enumerate(unq_k):
  dict_k[k] = i

test['keyword'] = test['keyword'].map(dict_k)
df = train.drop(['id','text','target','texts','sent_text'],axis=1)
df.index = range(0,7552)
Xvect = pd.DataFrame(train_tfidf.toarray())#len_text senttext
Xvect = pd.concat([df,Xvect],axis=1)
y = train['target']
print(Xvect)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xvect,y,test_size=0.2,random_state=1)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
model_Mnb = clf.fit(xtrain,ytrain)
pred_train = model_Mnb.predict(xtrain)
result = accuracy_score(pred_train,ytrain)
print(result)
pred_test = model_Mnb.predict(xtest)
result_test = accuracy_score(pred_test,ytest)
print(result_test)
import xgboost as xgb
from sklearn import model_selection
clf_xgb_TFIDF = xgb.XGBClassifier(max_depth=15, n_estimators=300, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)
model = clf_xgb_TFIDF.fit(xtrain,ytrain)

from sklearn.metrics import accuracy_score
pred = model.predict(xtrain)
result = accuracy_score(pred,ytrain)
print(result)
pred = model.predict(xtest)
result_test = accuracy_score(pred,ytest)
print(result_test)
df_test = test.drop(['id','text','texts','sent_text'],axis=1)
df_test.index = range(0,3237)
Xvect_test = pd.DataFrame(test_tfidf.toarray())#len_text senttext
Xvect_test = pd.concat([df_test,Xvect_test],axis=1)
print(df_test)
submission = model.predict(Xvect_test)
submission = pd.DataFrame(submission)
print(submission)
submission.to_csv()
