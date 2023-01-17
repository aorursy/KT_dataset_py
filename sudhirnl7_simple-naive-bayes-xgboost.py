import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import re

import nltk

from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize

import string



from sklearn.model_selection import train_test_split,KFold

from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import xgboost as xgb 

seed = 4353
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('Number of rows and columns in data set',train.shape)

train.head()
print('Number of rows and columns in data set',test.shape)

test.head()
train['author'].value_counts()
plt.figure(figsize=(14,5))

sns.countplot(train['author'],)

plt.xlabel('Author')

plt.title('Target variable distribution')

plt.show()
print('Original text:\n',train['text'][0])

review = re.sub('[^A-Za-z0-9]'," ",train['text'][0]) 

print('\nAfter removal of punctuation:\n',review)
review = word_tokenize(train['text'][0]) 

print('Word Tokenize:\n',review)



review = [word for word in str(train['text'][0]).lower().split() if  word not in set(stopwords.words('english'))]

print('\nRemoval of Stopwords:\n',review)



review = [word for word in str(train['text'][0]).lower().split() if  word in set(stopwords.words('english'))]

print('\nStopwords in the sentence:\n',review)



ps = PorterStemmer()

review = [ps.stem(word) for word in str(train['text'][0]).lower().split()]

print('\nStemming of word:\n',review)
def clean_text(df):

    ps = PorterStemmer()

    corpus = []

    for i in range(0, df.shape[0]):        

        review = re.sub('[^A-Za-z0-9]'," ",df['text'][i])

        review = word_tokenize(review)        

        review = [word for word in review if word.lower() not in set(stopwords.words('english'))]

        review = [ps.stem(word) for word in review]

        review = ' '.join(review)

        corpus.append(review)

    

    return corpus
corp_train = clean_text(train)

corp_test = clean_text(test)

train['clean_text'] = corp_train

test['clean_text'] = corp_test

del corp_train,corp_test
def text_len(df):

    #i = ['text']

    df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))

    df['num_uniq_words'] = df['text'].apply(lambda x: len(set(str(x).split())))

    df['num_chars'] = df['text'].apply(lambda x: len(str(x)))

    df['num_stopwords'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() 

                                                          if w in set(stopwords.words('english'))]))

    df['num_punctuations'] = df['text'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))

    df['num_words_upper'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    df['num_words_title'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    df['mean_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



text_len(train)

text_len(test)
plt.figure(figsize=(14,6))

plt.subplot(211)

sns.heatmap(pd.crosstab(train['author'],train['num_words']),cmap='gist_earth',xticklabels=False)

plt.xlabel('Original text word count')

plt.ylabel('Author')



plt.subplot(212)

sns.heatmap(pd.crosstab(train['author'],train['num_uniq_words']),cmap='gist_heat',xticklabels=False)

plt.xlabel('Unique text word count')

plt.ylabel('Author')

plt.tight_layout()

plt.show()
plt.figure(figsize=(14,6))

sns.distplot(train['num_words'],bins=100,color='r')

plt.title('Distribution of original text words')

plt.show()
train['num_uniq_words'].value_counts()[0:10].plot(kind='bar',color=['r','y'])

plt.xlabel('Original text word count')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(14,6))

sns.heatmap(train.corr(),annot=True)

plt.show()
cv =CountVectorizer(max_features=2000,ngram_range=(1,3),dtype=np.int8,stop_words='english')

X_cv = cv.fit_transform(train['clean_text']).toarray()

X_test_cv = cv.fit_transform(test['clean_text']).toarray()
author_name = {'EAP':0,'HPL':1,'MWS':2}

y = train['author'].map(author_name) 
mNB = MultinomialNB()



kf = KFold(n_splits=10,shuffle=True,random_state=seed)

pred_test_full = 0

cv_score = []

i=1

for train_index,test_index in kf.split(X_cv):

    print('{} of KFlod {}'.format(i,kf.n_splits))    

    xtr,xvl = X_cv[train_index], X_cv[test_index]

    ytr,yvl = y[train_index], y[test_index]

    

    mNB.fit(xtr,ytr)

    y_mNB = mNB.predict(xvl)

    cv_score.append(log_loss(yvl,mNB.predict_proba(xvl)))    

    pred_test_full += mNB.predict_proba(X_test_cv)

    i+=1

#roc_auc_score(yvl,mNB.predict_proba(xvl)[:,1]) # not for multi class

print(cv_score)

print('Mean accuracy score',np.mean(cv_score))

print('confusion matrix:\n',confusion_matrix(yvl,y_mNB))

del xtr,ytr,xvl,yvl
y_pred = pred_test_full/10

submit = pd.DataFrame(test['id'])

submit = submit.join(pd.DataFrame(y_pred))

submit.columns = ['id','EAP','HPL','MWS'] 

#submit.to_csv('spooky_pred1.csv.gz',index=False,compression='gzip')

submit.to_csv('spooky_pred1.csv',index=False)
tfidf = TfidfVectorizer(max_features=2000,dtype=np.float32,analyzer='word',

                        ngram_range=(1, 3),use_idf=True, smooth_idf=True, 

                        sublinear_tf=True)

X_tf = tfidf.fit_transform(train['clean_text']).toarray()

X_test_tf = tfidf.fit_transform(test['clean_text']).toarray()
mNB = MultinomialNB()



kf = KFold(n_splits=10,shuffle=True,random_state=seed)

pred_test_full = 0

cv_score = []

i=1

for train_index,test_index in kf.split(X_tf):

    print('{} of KFlod {}'.format(i,kf.n_splits))    

    xtr,xvl = X_tf[train_index], X_tf[test_index]

    ytr,yvl = y[train_index], y[test_index]

    

    mNB.fit(xtr,ytr)

    y_mNB = mNB.predict(xvl)

    cv_score.append(log_loss(yvl,mNB.predict_proba(xvl)))    

    pred_test_full += mNB.predict_proba(X_test_tf)

    i+=1

#roc_auc_score(yvl,mNB.predict_proba(xvl)[:,1]) # not for multi class

print(cv_score)

print('Mean accuracy score',np.mean(cv_score))

print('confusion matrix:\n',confusion_matrix(yvl,y_mNB))

del xtr,ytr,xvl,yvl
y_pred = pred_test_full/10

submit = pd.DataFrame(test['id'])

submit = submit.join(pd.DataFrame(y_pred))

submit.columns = ['id','EAP','HPL','MWS'] 

#submit.to_csv('spooky_pred2.csv.gz',index=False,compression='gzip')

submit.to_csv('spooky_pred2.csv',index=False)
#filter data set

unwanted = ['text','id','clean_text']

X_tf = np.concatenate((X_tf,train.drop(unwanted+['author'],axis=1).values),axis=1)

X_test_tf = np.concatenate((X_test_tf,test.drop(unwanted,axis=1).values),axis=1)

def runXGB(xtrain,xvalid,ytrain,yvalid,xtest,eta=0.1,early_stop=50,max_depth=5,n_rounds=1000):

    

    params = {        

        'objective':'multi:softprob',

        'learning_rate':eta,

        'max_depth':max_depth,

        'num_class':3,

        'subsample':0.8,

        'colsample_bytree':0.8,

        'eval_metric':'mlogloss',

        'min_child_weight':10,

        'reg_alpha':1.5, 

        'reg_lambda':5,

        'scale_pos_weight':1,  

        #'verbose':0,

        'seed':seed,        

        'n_thread':-1 

    }

    

    #plst = list(params.items())

    dtrain =xgb.DMatrix(xtrain,label=ytrain)

    dvalid = xgb.DMatrix(xvalid,label=yvalid)    

    dtest = xgb.DMatrix(xtest)

    watchlist = [(dtrain,'train'),(dvalid,'test')]

    

    model = xgb.train(params,dtrain,n_rounds,evals=watchlist,early_stopping_rounds=early_stop,verbose_eval=10)

    pred = model.predict(dvalid,ntree_limit = model.best_ntree_limit)

    pred_test = model.predict(dtest,ntree_limit = model.best_ntree_limit)

    

    return pred_test,model
kf = KFold(n_splits=2,shuffle=True,random_state=seed)

pred_test_full = 0

cv_score = []

i=1

for train_index,test_index in kf.split(X_tf):

    print('{} of KFlod {}'.format(i,kf.n_splits))    

    xtr,xvl = X_tf[train_index], X_tf[test_index]

    ytr,yvl = y[train_index], y[test_index]

        

    pred_xgb,xg_model = runXGB(xtr,xvl,ytr,yvl,X_test_tf,n_rounds=200,eta=0.5)

    pred_test_full += pred_xgb

    cv_score.append(xg_model.best_score)

    i+=1

#roc_auc_score(yvl,mNB.predict_proba(xvl)[:,1]) # not for multi class

#print(cv_score)

#print('Mean accuracy score',np.mean(cv_score))

#del xtr,ytr,xvl,yvl,X_tf,X_test_tf
print(cv_score)

print('Mean accuracy score',np.mean(cv_score))
y_pred = pred_test_full/2

submit = pd.DataFrame(test['id'])

submit = submit.join(pd.DataFrame(y_pred))

submit.columns = ['id','EAP','HPL','MWS'] 

#submit.to_csv('spooky_pred3.csv.gz',index=False,compression='gzip')

submit.to_csv('spooky_pred3.csv',index=False)