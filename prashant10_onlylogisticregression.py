# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import log_loss

from sklearn.naive_bayes import MultinomialNB,GaussianNB

from sklearn import svm

import xgboost as xgb

from sklearn.decomposition import TruncatedSVD



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sampleSubmission = pd.read_csv("../input/sample_submission.csv")
col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

trainTxt = train['comment_text']

testTxt = test['comment_text']

trainTxt = trainTxt.fillna("unknown")

testTxt = testTxt.fillna("unknown")

combinedTxt = pd.concat([trainTxt,testTxt],axis=0)
vect = TfidfVectorizer(decode_error='ignore',use_idf=True,smooth_idf=True,min_df=10,ngram_range=(1,3),lowercase=True,

                      stop_words='english')
combinedDtm = vect.fit_transform(combinedTxt) #fit on combine

trainDtm = combinedDtm[:train.shape[0]]

testDtm = vect.transform(testTxt) #transform only test
lrpreds = np.zeros((test.shape[0],len(col)))

nbpreds = np.zeros((test.shape[0],len(col)))

svmpreds = np.zeros((test.shape[0],len(col)))

xgbpreds = np.zeros((test.shape[0],len(col)))
svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)

trainDtmSvd = svd.fit_transform(trainDtm)

testDtmSvd = svd.transform(testDtm)
#call fit on every single col value 

#normal lr

loss = []

for i,j in enumerate(col):

    lr = LogisticRegression(C=4)

    lr.fit(trainDtm,train[j]) #train[j] is each type of comment

    lrpreds[:,i] = lr.predict_proba(testDtm)[:,1]

    train_preds = lr.predict_proba(trainDtm)[:,1]

    loss.append(log_loss(train[j],train_preds))

np.mean(loss)
#lr with Svd

loss = []

for i,j in enumerate(col):

    lr = LogisticRegression(C=4)

    lr.fit(trainDtmSvd,train[j]) #train[j] is each type of comment

    lrpreds[:,i] = lr.predict_proba(testDtmSvd)[:,1]

    train_preds = lr.predict_proba(trainDtmSvd)[:,1]

    loss.append(log_loss(train[j],train_preds))

np.mean(loss)
#normal nb

loss = []

for i,j in enumerate(col):

    nb = MultinomialNB()

    nb.fit(trainDtm,train[j]) #train[j] is each type of comment

    nbpreds[:,i] = nb.predict_proba(testDtm)[:,1]

    train_preds = nb.predict_proba(trainDtm)[:,1]

    loss.append(log_loss(train[j],train_preds))

np.mean(loss)
#GaussianNB  with svd

loss = []

for i,j in enumerate(col):

    nb = GaussianNB()

    nb.fit(trainDtmSvd,train[j]) #train[j] is each type of comment

    nbpreds[:,i] = nb.predict_proba(testDtmSvd)[:,1]

    train_preds = nb.predict_proba(trainDtmSvd)[:,1]

    loss.append(log_loss(train[j],train_preds))

np.mean(loss)
#normal xgb

loss = []

for i,j in enumerate(col):

    xg = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

    xg.fit(trainDtm,train[j]) #train[j] is each type of comment

    xgbpreds[:,i] = xg.predict_proba(testDtm)[:,1]

    train_preds = xg.predict_proba(trainDtm)[:,1]

    loss.append(log_loss(train[j],train_preds))

np.mean(loss)
#xgb with svd

loss = []

for i,j in enumerate(col):

    xg = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)

    xg.fit(trainDtmSvd,train[j]) #train[j] is each type of comment

    xgbpreds[:,i] = xg.predict_proba(testDtmSvd)[:,1]

    train_preds = xg.predict_proba(trainDtmSvd)[:,1]

    loss.append(log_loss(train[j],train_preds))

np.mean(loss)
# predsMix = 0.6*lrpreds+0.3*xgbpreds+0.1*nbpreds

predsMix = xgbpreds

predsDf = pd.DataFrame(predsMix,columns = col)

subid = pd.DataFrame({'id':sampleSubmission['id']})

finalPreds = pd.concat([subid,predsDf],axis=1)

finalPreds.to_csv("mix.csv",index=False)