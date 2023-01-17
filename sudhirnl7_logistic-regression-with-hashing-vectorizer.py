import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy import sparse
%matplotlib inline
seed = 2390
#path = 'file/'
path = '../input/'
train = pd.read_csv(path+'train.csv',nrows= None)
test = pd.read_csv(path+'test.csv', nrows= None)
print('Number of rows and columns in the train data set:',train.shape)
print('Number of rows and columns in the test data set:',test.shape)
train.head()
test.head()
target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
fig,ax = plt.subplots(2,3,figsize=(16,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
sns.countplot(train['toxic'],palette= 'magma',ax=ax1)
sns.countplot(train['severe_toxic'], palette= 'viridis',ax=ax2)
sns.countplot(train['obscene'], palette= 'Set1',ax=ax3)
sns.countplot(train['threat'], palette= 'viridis',ax = ax4)
sns.countplot(train['insult'], palette = 'magma',ax=ax5)
sns.countplot(train['identity_hate'], palette = 'Set1', ax = ax6);
k = pd.DataFrame()
k['train'] = train.isnull().sum()
k['test'] = test.isnull().sum()
k
#Hashing vectorizer
## Word
hash_word = HashingVectorizer(analyzer='word', stop_words= 'english' , ngram_range= (1,3), 
                              token_pattern= r'w{1,}', strip_accents= 'unicode',
                             dtype= np.float32, tokenizer= nltk.tokenize.word_tokenize )
#Char
hash_char = HashingVectorizer(analyzer='char', stop_words= 'english' , ngram_range= (3,6),
                              strip_accents= 'unicode',dtype= np.float32 )
# Word
tr_hash = hash_word.transform(train['comment_text'])
ts_hash = hash_word.transform(test['comment_text'])
# char

tr_hash_char = hash_char.transform(train['comment_text'])
ts_hash_char = hash_char.transform(test['comment_text'])

# Sparse
X = sparse.hstack([tr_hash, tr_hash_char])
X_test = sparse.hstack([ts_hash, ts_hash_char])
target_col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
y = train[target_col]
gc.collect()
lr = LogisticRegression(C=4,random_state = seed)
prd = np.zeros((X_test.shape[0],y.shape[1]))
cv_score_auc =[]
frp,trp = [], []
for i,col in enumerate( y.columns):
    print('Building model for column:',col) 
    lr.fit(X,y[col])
    
    # auc
    pred_prob = lr.predict_proba(X)[:,1]
    f,t,_ = roc_curve(y[col], pred_prob)
    frp.append(f)
    trp.append(t)
    cv_score_auc.append(auc(f,t))
    prd[:,i] = lr.predict_proba(X_test)[:,1]
    
# Mean Auc
np.mean(cv_score_auc)
print("Column:",col)
pred =  lr.predict(X)
print('\nConfusion matrix\n',confusion_matrix(y[col],pred))
print(classification_report(y[col],pred))
plt.figure(figsize=(14,10))

color = ['r','g','b','k','y','gray']
for i, c in enumerate(target_col):
    print("Column:",c)
    plt.plot([0,1],[0,1],color='b')
    plt.plot(frp[i],trp[i],color=color[i],label= c)
    plt.legend(loc='lower right')
    plt.xlabel('True positive rate')
    plt.ylabel('False positive rate')
    plt.title('Reciever Operating Characteristic')
prd_1 = pd.DataFrame(prd,columns=y.columns)
submit = pd.concat([test['id'],prd_1],axis=1)
#submit.to_csv('toxic_lr.csv.gz',compression='gzip',index=False)
submit.to_csv('toxic_lr_hash.csv',index=False)
submit.head()