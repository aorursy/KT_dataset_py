import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import MultinomialNB,GaussianNB

import xgboost as xgb

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import log_loss
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head().to_hdf('tmp1.hdf','tmp1')
train_text = train['comment_text']

test_text = test['comment_text']

train_text = train_text.fillna(" ")

test_text = test_text.fillna(" ")

# tfidf, punctuations, length, capitle letters, spelling mistakes, LSTM 

tfidf_model = TfidfVectorizer(decode_error='ignore',use_idf=True,smooth_idf=True,

                       min_df=10,ngram_range=(1,3),lowercase=True, stop_words='english')



tfidf_model = tfidf_model.fit(pd.concat([train_text,test_text],axis=0))



train_tfidf = tfidf_model.transform(train_text)

test_tfidf = tfidf_model.transform(test_text) #transform only test
#split to train and test

all_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#call fit on every single col value 

#normal lr

loss = []

lr = dict()

lr_preds = dict()#np.zeros((test.shape[0],len(all_labels)))

for label in all_labels:

    lr[label] = LogisticRegression(C=4)

    lr[label].fit(X_train,y_train)

    lr_preds[label] = lr.predict_proba(test_tfidf)[:,1]

    train_preds = lr.predict_proba(trainDtm)[:,1]

    loss.append(log_loss(train[j],train_preds))

print(loss)
import nltk
nltk.h