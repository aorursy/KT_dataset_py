!wget https://machinehack-be.s3.amazonaws.com/product_sentiment_classification_weekend_hackathon_19/Participants_Data.zip
!unzip Participants_Data.zip
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('Participants_Data/Train.csv')

test = pd.read_csv('Participants_Data/Test.csv')

sub = pd.read_csv('Participants_Data/Sample Submission.csv')
train.head(5)
test.head(5)
from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(background_color='white',

                    stopwords =  set(STOPWORDS),

                    max_words = 50, 

                    random_state = 42,)

wc.generate(' '.join(train['Product_Description']))

plt.imshow(wc)
train.isnull().sum(),test.isnull().sum(),train.shape,test.shape,train.dtypes
df=train.append(test,ignore_index=True)
import nltk

nltk.download('stopwords')

nltk.download('vader_lexicon')

#df['punctuation_count'] = df['Product_Description'].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))

df['numerics'] = df['Product_Description'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

df['upper'] = df['Product_Description'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.tokenize import TreebankWordTokenizer

cvec = TfidfVectorizer(max_features=10000, norm = 'l1', lowercase=True, smooth_idf=False, sublinear_tf=False, ngram_range=(1,4), tokenizer=TreebankWordTokenizer().tokenize)

df_info = pd.DataFrame(cvec.fit_transform(df['Product_Description']).todense())

df_info.columns = ['Product_Description_Top_' + str(c) for c in df_info.columns]

df = pd.concat([df, df_info], axis=1)
import re

def clean_text(text):

    text = text.lower()

    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)   

    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)   

    text = re.sub(r'www.[^ ]+', '', text)  

    text = re.sub(r'[a-zA-Z0-9]*www[a-zA-Z0-9]*com[a-zA-Z0-9]*', '', text)  

    text = re.sub(r'[^a-zA-Z]', ' ', text)   

    text = [token for token in text.split() if len(token) > 2]

    text = ' '.join(text)

    return text



df['Product_Description'] = df['Product_Description'].apply(clean_text)

from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(background_color='white',

                    stopwords =  set(STOPWORDS),

                    max_words = 50, 

                    random_state = 42,)

wc.generate(' '.join(df['Product_Description']))

plt.imshow(wc)
import string

punctuation=string.punctuation

df['word_count']=df['Product_Description'].apply(lambda x: len(str(x).split(" ")))

df['char_count'] = df['Product_Description'].str.len()

def avg_word(sentence):

    words = sentence.split()

    return (sum(len(word) for word in words)/(len(words)+1))



df['avg_word'] = df['Product_Description'].apply(lambda x: avg_word(x))

from nltk.corpus import stopwords

stop = stopwords.words('english')



df['stopwords'] = df['Product_Description'].apply(lambda x: len([x for x in x.split() if x in stop]))

df['word_density'] = df['char_count'] / (df['word_count']+1)

j=[]

for i in df['Product_Description']:

  j.append(len(i))

df['len']=j
from textblob import TextBlob

df['polarity'] = df.apply(lambda x: TextBlob(x['Product_Description']).sentiment.polarity, axis=1)

df['subjectivity'] = df.apply(lambda x: TextBlob(x['Product_Description']).sentiment.subjectivity, axis=1)
df['ID_Type']=df['Product_Type']+df['Text_ID']
df.head(5)
del df['Product_Description']

train = df[df['Sentiment'].isnull()==False]

test = df[df['Sentiment'].isnull()==True]

del test['Sentiment']
train['Sentiment'].value_counts()
train_df=train

test_df=test
X = train_df.drop(labels=['Sentiment'], axis=1)

y = train_df['Sentiment'].values
from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.10, random_state=101, stratify=y)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
from sklearn.metrics import log_loss


import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)

test_data = lgb.Dataset(X_cv, label=y_cv)



param = {'objective': 'multiclass',

         'num_class': 4,

         'boosting': 'gbdt',  

         'metric': 'multi_logloss',

         'learning_rate': 0.01, 

         'num_iterations': 1000,

         'num_leaves': 31,

         'max_depth': -1,

         'min_data_in_leaf': 15,

         'bagging_fraction':0.9,

         'bagging_freq': 2,

         'feature_fraction': 0.9,

         'lambda_l2': 0.9,

         'min_data_per_group': 75,

         'max_bin': 255,

         'is_unbalance':True

         }



clf = lgb.train(params=param, 

                early_stopping_rounds=200,

                verbose_eval=100,

                train_set=train_data,

                valid_sets=[test_data])



y_pred = clf.predict(X_cv)
log_loss(y_cv, y_pred)
Xtest = test_df


from sklearn.model_selection import KFold, StratifiedKFold



errlgb = []

y_pred_totlgb = []



fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2**31)



for train_index, test_index in fold.split(X, y):

    

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    train_data = lgb.Dataset(X_train, label=y_train)

    test_data = lgb.Dataset(X_test, label=y_test)

    

    clf = lgb.train(params=param, 

                     early_stopping_rounds=200,

                     verbose_eval=100,

                     train_set=train_data,

                     valid_sets=[test_data])



    y_pred = clf.predict(X_test)

    print("Log Loss: ", (log_loss(y_test, y_pred)))

    

    errlgb.append(log_loss(y_test, y_pred))

    p = clf.predict(Xtest)

    y_pred_totlgb.append(p)
np.mean(errlgb,0)
'''

x=[]

for i in errlgb:

  if i>0.43:

    xx=errlgb.index(i)

    x.append(xx)

x=sorted(x, reverse=True)

print(x)

for i in x:

  del y_pred_totlgb[i]

  del errlgb[i]

'''
y_pred = np.mean(y_pred_totlgb,0)
submission = pd.DataFrame(data=y_pred, columns=sub.columns)

submission.head()
submission.to_csv('Mh13.csv', index=False)