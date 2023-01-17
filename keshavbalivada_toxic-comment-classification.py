import pandas as pd
import numpy as np
import string
import re
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
test_data.head(5)

train_data.fillna('Unknown', inplace=True)
import string
exclude = set(string.punctuation)



def remove_punctuation(x):
    try:
        x = ''.join(ch for ch in x if ch not in exclude)
    except:
        pass
    return x
train_data.comment_text = train_data.comment_text.apply(remove_punctuation)
train_data['comment_text'].replace(to_replace="\n", value=r" ", regex=True, inplace=True)
train_data = train_data.apply(lambda x: x.astype(str).str.lower())
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

train_data['Tokenized_words'] = train_data['comment_text'].apply(word_tokenize)
stop_words = set(stopwords.words("english"))
train_data['Keywords_nostopwords']= train_data['Tokenized_words'].apply(lambda x: [item for item in x if item not in stop_words])
train_data.head(5)
#split_data = nltk.pos_tag()
#sent = train_data["Keywords_nostopwords"].apply(split_data)
#[b  for b in sent if b[-1] not in ['CC','CD','DT','EX','IN','LS','MD','PDT','POS','PRP','PRP$','WRB','WP','VBD','VBN','VBP','VB','RB']]

tok_and_tag = lambda x: pos_tag(x)
#tok_and_tag(train_data['Keywords_nostopwords'][0])
train_data["pos_Tag"]=train_data["Keywords_nostopwords"].apply(tok_and_tag)
train_data["comment_text"]
train_context = train_data["comment_text"]
test_context = test_data["comment_text"]


test_context.head(5)
train_context.head(5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=1,max_features=10000,ngram_range=(1,3),stop_words = 'english')

X_train_dtm = tfidf.fit_transform(train_context)
X_train_dtm
X_test_dtm = tfidf.transform(test_context)
X_test_dtm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.linear_model import SGDClassifier

train_data.head(5)
dataframesample= pd.DataFrame(test_data.id)
label = list(train_data.iloc[:,2:].columns) 
for i in range(2,8):
    lr = SGDClassifier(loss='log', penalty='l1', alpha=1e-06)
    y_train = train_data.iloc[:,i]
    %time lr.fit(X_train_dtm, y_train)
    y_pred_class = lr.predict_proba(X_test_dtm)
    dataframesample[label[i-2]] = (y_pred_class[:,1])
dataframesample.to_csv('output_hackathon.csv',index=False)

