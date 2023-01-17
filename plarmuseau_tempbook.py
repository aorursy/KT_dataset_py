import numpy as np

import pandas as pd

from collections import Counter

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# timing function

import time   

start = time.clock() #_________________ measure efficiency timing



input_folder='../input/'

train = pd.read_csv(input_folder + 'train.csv',encoding='utf8')[:5000]

test  = pd.read_csv(input_folder + 'test.csv',encoding='utf8')[:5000]



# lege opvullen

train.fillna(value='leeg',inplace=True)

test.fillna(value='leeg',inplace=True)



print("Original data: trainQ: {}, testQ: {}".format(train.shape, test.shape) )

end = time.clock()

print('open:',end-start)



def cleantxt(x):   

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':' , '[' ,']' ,'{' ,'}' ,'^']:

        x = x.replace(char, ' ' + char + ' ')

    return x

train['question1']=train['question1'].map(cleantxt)

train['question2']=train['question2'].map(cleantxt)

test['question1']=test['question1'].map(cleantxt)

test['question2']=test['question2'].map(cleantxt)
from nltk.tokenize import TreebankWordTokenizer

from math import*

print(train.head())



import re

r = re.compile("[ ,.?|']")



from nltk.corpus import stopwords

stops = set(stopwords.words('english'))





nr_tr=len(train)

nr_te=len(test)



train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist())

test_qs = pd.Series(test['question1'].tolist() + test['question2'].tolist())

all_qs = pd.DataFrame(train_qs.append(test_qs) )

all_qs = all_qs.reset_index()

all_qs.columns=['index','Q']



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer( ngram_range=(1,1),use_idf=True, tokenizer=TreebankWordTokenizer().tokenize)

tfidf2 = TfidfVectorizer( ngram_range=(2,2),use_idf=True, tokenizer=TreebankWordTokenizer().tokenize)

tfidfall=tfidf.fit_transform(all_qs['Q'])

tfidfall2=tfidf2.fit_transform(all_qs['Q'])#print(tfidf)#print(tfidf.stop_words_ )

wordnr = pd.DataFrame(pd.Series(list(tfidf.vocabulary_.keys()), index=tfidf.vocabulary_.values()),columns=['woord']) # #print(words)  #woordenboek !

nrword = pd.DataFrame.from_dict(tfidf.vocabulary_,orient='index')

nrword.columns=['tfnr']#nrword['idf']=tfidf.idf_#print(nrword)

end = time.clock()

print('__________tfidf:',end-start)



print('Test examples for first Q pair')

Qnr=0

print('cosinesim',  (tfidfall.getrow(Qnr)*tfidfall.getrow(Qnr+nr_tr).T).toarray()[0][0]     )

print('cosinesim2',tfidfall[Qnr:Qnr+1].todense()*tfidfall[nr_tr+Qnr:nr_tr+Qnr+1].todense().T)



coo = tfidfall[Qnr:Qnr+1].tocoo(copy=False)

vector1=pd.DataFrame(data=coo.data,index=coo.col,columns=['tfidf']) #print(vector1.T)

coo2 = tfidfall[Qnr+nr_tr:Qnr+nr_tr+1].tocoo(copy=False)

vector2=pd.DataFrame(data=coo2.data,index=coo2.col,columns=['tfidf']) #print(vector2.T)

cosin= vector1*vector2

vector1['tfidf2']=vector2['tfidf']

vector1['woord']=wordnr['woord'] #print(vector1)

vector1=vector1.fillna(0) #print(vector1.T)

eucl = (vector1['tfidf']-vector1['tfidf2'])**2

print('cosin',cosin.sum())

print('eucl',sqrt(eucl.sum()))

vector1['woord']=wordnr['woord'] #print('vector',vector1.T)

q1_word = train.ix[Qnr]['question1'].lower().split()

q2_word = train.ix[Qnr]['question2'].lower().split()

alien_word = [list(vector1[vector1['woord']==w]['tfidf'])[0] for w in q1_word if w not in q2_word]

print ('weighted sumdiff',sum(alien_word))

print(q1_word,q2_word,alien_word)



def word_match(row):

    #print(row)

    q1_word = {}

    q2_word = {}

    q1_word=TreebankWordTokenizer().tokenize(row['question1'].lower())

    q2_word=TreebankWordTokenizer().tokenize(row['question2'].lower())        

    q1_word = [w for w in q1_word if w not in stops]

    q2_word = [w for w in q2_word if w not in stops]



    if len(q1_word) == 0 or len(q2_word)==0:

        return 0,0,0,0,0,0,0,0,0,0,0,0,0,0

    shared_word_in_q1 = [w for w in q1_word if w in q2_word]

    shared_word_in_q2 = [w for w in q2_word if w in q1_word]

    alien_word_in_q1 = [w for w in q1_word if w not in q2_word]

    alien_word_in_q2 = [w for w in q2_word if w not in q1_word] 

    stop_word_in_q1 = [w for w in q1_word if w in stops]

    stop_word_in_q2 = [w for w in q2_word if w in stops] 



    nr_rij=row['id']

    #T= (tfidfall[nr_rij:nr_rij+1]*tfidfall[nr_rij+nr_tr:nr_rij+nr_tr+1].T).toarray()[0][0]      

    AD = len(shared_word_in_q2)/(len(shared_word_in_q2)+len(alien_word_in_q1)*2+len(alien_word_in_q2)*2)

    D = (len(shared_word_in_q1)+len(shared_word_in_q2))/(len(q1_word)+len(q2_word))

    K = len(shared_word_in_q1)/len(q1_word)*0.5+len(shared_word_in_q2)/len(q2_word)*0.5

    O = len(shared_word_in_q2)/sqrt(len(q1_word)*len(q2_word))

    S = (len(shared_word_in_q1)-len(stop_word_in_q1)+len(shared_word_in_q2)-len(stop_word_in_q2))/(len(q1_word)-len(stop_word_in_q1)+len(q2_word)-len(stop_word_in_q2))

    A1 = (len(alien_word_in_q1)-len(stop_word_in_q1))/(len(q1_word)-len(stop_word_in_q1))    

    A2 = (len(alien_word_in_q2)-len(stop_word_in_q2))/(len(q2_word)-len(stop_word_in_q2))

    coo1 = tfidfall.getrow(nr_rij).tocoo(copy=False)

    coo2 = tfidfall.getrow(nr_rij+nr_tr).tocoo(copy=False)

    vector1=pd.DataFrame(data=coo1.data,index=coo1.col,columns=['tfidf']) 

    vector2=pd.DataFrame(data=coo2.data,index=coo2.col,columns=['tfidf']) 

    #print('v1',vector1.T)

    cosin= vector1*vector2

    vector1['tfidf2']=vector2['tfidf']

    eucl = (vector1['tfidf']-vector1['tfidf2'])**2    

    T=(cosin.sum())[0]

    E=sqrt(eucl.sum())

    vector1['woord']=wordnr['woord']

    vector1=vector1.fillna(0)

    vector2['woord']=wordnr['woord']

    shared_word_weight1 = [list(vector1[vector1['woord']==w]['tfidf'])[0] for w in shared_word_in_q1]

    alien_word_weight1 = [list(vector1[vector1['woord']==w]['tfidf'])[0] for w in alien_word_in_q1]

    #print('aliens',alien_word_in_q2,'vector2',vector2.T)

    alien_word_weight2 = [list(vector2[vector2['woord']==w]['tfidf'])[0] for w in alien_word_in_q2]

    eucl_word_weight2 = [list( (vector2[vector2['woord']==w]['tfidf']-0)**2 ) [0] for w in alien_word_in_q2]

    ADw = sum(shared_word_weight1)/(sum(shared_word_weight1)+sum(alien_word_weight1*2)+sum(alien_word_weight2*2))

    ED=sqrt(sum(eucl_word_weight2))

    WK = sum(shared_word_weight1)/sum(vector1['tfidf'])*0.5+sum(shared_word_weight1)/sum(vector2['tfidf'])*0.5

    WO = sum(shared_word_weight1)/sqrt(sum(vector1['tfidf'])*sum(vector2['tfidf']))

    #print(eucl_word_weight2)

    return D,A1,A2,S,T,sum(alien_word_weight1),sum(alien_word_weight2),E,ED,AD,O,K,ADw,WK,WO





temp = train.apply(word_match,axis=1)

train_wm = pd.DataFrame(temp.tolist(), columns=['match', 'alien1','alien2','matchS','cosin','difw1','difw2','Eucl','Eudiff','AntiDice','Ochia','Kulz','WeightADice','WeightKulz','WeightOchia'])

end = time.clock()

print('wm:',len(train_wm)/end-start)

train[['match','alien1','alien2','matchS','cosin','difw1','difw2','Eucl','Eudiff','AntiDice','Ochia','Kulz','WeightADice','WeightKulz','WeightOchia']]=train_wm

end = time.clock()

print('____different similarities:',end-start)

def plotter(kolom):

    plt.figure(figsize=(15,5))

    plt.hist(train[train['is_duplicate'] == 0][kolom].fillna(0), bins=20, normed=False, label='Not Duplicate')

    plt.hist(train[train['is_duplicate'] == 1][kolom].fillna(0), bins=20, normed=False, alpha=0.7, label='Duplicate')

    plt.legend()

    plt.title('Label distribution over', fontsize=15)

    plt.xlabel(kolom, fontsize=15)



plotter('match')    

plotter('alien1')    

plotter('alien2')    

plotter('matchS')

plotter('cosin')

plotter('difw1')

plotter('difw2')

plotter('Eucl')

plotter('Eudiff')

plotter('AntiDice')

plotter('Ochia')

plotter('Kulz')

plotter('WeightADice')

plotter('WeightKulz')

plotter('WeightOchia')

end = time.clock()

print('clean and make freq word dict:',end-start)
print(train[train['Ochia']>0.98])



from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss

import scipy

import xgboost as xgb

import difflib



y=train['is_duplicate']     

feats = train.columns.values.tolist()

feats=[x for x in feats if x not in ['question1','question2','id','qid1','qid2','is_duplicate']]

print("features",feats)



x_train, x_valid, y_train, y_valid = train_test_split(train[feats], y, test_size=0.1, random_state=0)

#XGBoost model

params = {"objective":"binary:logistic",'eval_metric':'logloss',"max_depth":7}



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=200,verbose_eval=25) #change to higher #s

print('training done')



end = time.clock()

print('____________trained:',end-start)

print("log loss for training data set",log_loss(y, bst.predict(xgb.DMatrix(train[feats]))))

#Predicting for test data set

sub = pd.DataFrame() # Submission data frame

sub['test_id'] = []

sub['is_duplicate'] = []

header=['test_id','question1','question2','id','qid1','qid2','is_duplicate']


def word_match_test(row):

    #print(row)

    q1_word = {}

    q2_word = {}

    q1_word=TreebankWordTokenizer().tokenize(row['question1'].lower())

    q2_word=TreebankWordTokenizer().tokenize(row['question2'].lower())        

    q1_word = [w for w in q1_word if w not in stops]

    q2_word = [w for w in q2_word if w not in stops]



    if len(q1_word) == 0 or len(q2_word)==0:

        return 0,0,0,0,0,0,0,0,0,0,0,0,0,0

    shared_word_in_q1 = [w for w in q1_word if w in q2_word]

    shared_word_in_q2 = [w for w in q2_word if w in q1_word]

    alien_word_in_q1 = [w for w in q1_word if w not in q2_word]

    alien_word_in_q2 = [w for w in q2_word if w not in q1_word] 

    stop_word_in_q1 = [w for w in q1_word if w in stops]

    stop_word_in_q2 = [w for w in q2_word if w in stops] 



    nr_rij=row['test_id']+nr_tr*2

    #T= (tfidfall[nr_rij:nr_rij+1]*tfidfall[nr_rij+nr_tr:nr_rij+nr_tr+1].T).toarray()[0][0]      

    AD = len(shared_word_in_q2)/(len(shared_word_in_q2)+len(alien_word_in_q1)*2+len(alien_word_in_q2)*2)

    D = (len(shared_word_in_q1)+len(shared_word_in_q2))/(len(q1_word)+len(q2_word))

    K = len(shared_word_in_q1)/len(q1_word)*0.5+len(shared_word_in_q2)/len(q2_word)*0.5

    O = len(shared_word_in_q2)/sqrt(len(q1_word)*len(q2_word))

    S = (len(shared_word_in_q1)-len(stop_word_in_q1)+len(shared_word_in_q2)-len(stop_word_in_q2))/(len(q1_word)-len(stop_word_in_q1)+len(q2_word)-len(stop_word_in_q2))

    A1 = (len(alien_word_in_q1)-len(stop_word_in_q1))/(len(q1_word)-len(stop_word_in_q1))    

    A2 = (len(alien_word_in_q2)-len(stop_word_in_q2))/(len(q2_word)-len(stop_word_in_q2))

    coo1 = tfidfall.getrow(nr_rij).tocoo(copy=False)

    coo2 = tfidfall.getrow(nr_rij+nr_te).tocoo(copy=False)

    vector1=pd.DataFrame(data=coo1.data,index=coo1.col,columns=['tfidf']) 

    vector2=pd.DataFrame(data=coo2.data,index=coo2.col,columns=['tfidf']) 

    #print('v1',vector1.T)

    cosin= vector1*vector2

    vector1['tfidf2']=vector2['tfidf']

    eucl = (vector1['tfidf']-vector1['tfidf2'])**2    

    T=(cosin.sum())[0]

    E=sqrt(eucl.sum())

    vector1['woord']=wordnr['woord']

    vector1=vector1.fillna(0)

    vector2['woord']=wordnr['woord']

    shared_word_weight1 = [list(vector1[vector1['woord']==w]['tfidf'])[0] for w in shared_word_in_q1]

    alien_word_weight1 = [list(vector1[vector1['woord']==w]['tfidf'])[0] for w in alien_word_in_q1]

    #print('aliens',alien_word_in_q2,'vector2',vector2.T)

    alien_word_weight2 = [list(vector2[vector2['woord']==w]['tfidf'])[0] for w in alien_word_in_q2]

    eucl_word_weight2 = [list( (vector2[vector2['woord']==w]['tfidf']-0)**2 ) [0] for w in alien_word_in_q2]

    ADw = sum(shared_word_weight1)/(sum(shared_word_weight1)+sum(alien_word_weight1*2)+sum(alien_word_weight2*2))

    ED=sqrt(sum(eucl_word_weight2))

    WK = sum(shared_word_weight1)/sum(vector1['tfidf'])*0.5+sum(shared_word_weight1)/sum(vector2['tfidf'])*0.5

    WO = sum(shared_word_weight1)/sqrt(sum(vector1['tfidf'])*sum(vector2['tfidf']))

    #print(eucl_word_weight2)

    return D,A1,A2,S,T,sum(alien_word_weight1),sum(alien_word_weight2),E,ED,AD,O,K,ADw,WK,WO





temp = test.apply(word_match_test,axis=1)

train_wm = pd.DataFrame(temp.tolist(), columns=['match', 'alien1','alien2','matchS','cosin','difw1','difw2','Eucl','Eudiff','AntiDice','Ochia','Kulz','WeightADice','WeightKulz','WeightOchia'])

end = time.clock()

print('wm:',len(train_wm)/end-start)

test[['match','alien1','alien2','matchS','cosin','difw1','difw2','Eucl','Eudiff','AntiDice','Ochia','Kulz','WeightADice','WeightKulz','WeightOchia']]=train_wm

end = time.clock()



def plottest(kolom):

    plt.figure(figsize=(15,5))

    plt.hist(test[kolom].fillna(0), bins=20, normed=False)

    plt.legend()

    plt.title('Label distribution over', fontsize=15)

    plt.xlabel(kolom, fontsize=15)

    

plottest('match')    

plottest('alien1')    

plottest('alien2')    

plottest('matchS')

plottest('cosin')

plottest('difw1')

plottest('difw2')

plottest('Eucl')

plottest('Eudiff')

plottest('AntiDice')

plottest('Ochia')

plottest('Kulz')

plottest('WeightADice')

plottest('WeightKulz')

plottest('WeightOchia')



sub=pd.DataFrame({'test_id':test['test_id'], 'is_duplicate':bst.predict(xgb.DMatrix(test[feats]))})

print(sub.head())

sub.to_csv('quora_xgb.csv', index=False)