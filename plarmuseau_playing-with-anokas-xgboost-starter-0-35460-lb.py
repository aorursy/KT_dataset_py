import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import math

from textblob import TextBlob as tb

import time

start = time.clock()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


df_train = pd.read_csv('../input/train.csv',encoding='utf8')[:10000]

df_train = df_train.fillna('leeg')

df_test = pd.read_csv('../input/test.csv',encoding='utf8')[:50000]

df_test = df_test.fillna('leeg')

df_train.head(2)

df_test.head(2)

end = time.clock()

print('open:',end-start)
import rake

import operator

stop = set(stopwords.words('english'))

rake_object = rake.Rake(stop, 3, 3, 1)

 

text = "Natural language processing (NLP) deals with the application of computational models to text or speech data. Application areas within NLP include automatic (machine) translation between languages; dialogue systems, which allow a human to interact with a machine using natural language; and information extraction, where the goal is to transform unstructured text into structured (database) representations that can be searched and browsed in flexible ways. NLP technologies are having a dramatic impact on the way people interact with computers, on the way people interact with each other through the use of language, and on the way people access the vast amount of linguistic data now in electronic form. From a scientific viewpoint, NLP involves fundamental questions of how to structure formal models (for example statistical models) of natural language phenomena, and of how to design algorithms that implement these models. In this course you will study mathematical and computational models of language, and the application of these models to key problems in natural language processing. The course has a focus on machine learning methods, which are widely used in modern NLP systems: we will cover formalisms such as hidden Markov models, probabilistic context-free grammars, log-linear models, and statistical models for machine translation. The curriculum closely follows a course currently taught by Professor Collins at Columbia University, and previously taught at MIT."

 

keywords = rake_object.run(text)

print( "keywords: ", keywords )

def cleantxt(x):    

    x = str(x)

    #x = x.replace(r'[^\x00-\x7f]',r' ') 

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        #x = x.replace(char, ' '+char+' ')

        x = x.replace(char, ' ')

    return x



def cleantxtsplit(x):

    x=cleantxt(x)

    return x.split()



def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in cleantxtsplit(row['question1']):

            q1words[word] = 1

    for word in cleantxtsplit(row['question2']):

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    nonshared_words_in_q1 = [w for w in q1words.keys() if w not in q2words]

    nonshared_words_in_q2 = [w for w in q2words.keys() if w not in q1words]

    

    #X = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    #Y = (len(nonshared_words_in_q1) + len(nonshared_words_in_q2))/(len(q1words) + len(q2words))

    X1 = (len(shared_words_in_q1))/(len(q1words) )

    Y1 = (len(nonshared_words_in_q1))/(len(q1words) )

    X2 = (len(shared_words_in_q2))/(len(q2words))

    Y2 = (len(nonshared_words_in_q2))/(len(q2words))

    #R1= math.atan(X1/(Y1+0.0001))

    #R2= math.atan(X2/(Y2+0.0001))

    R3 = (Y1+Y2)/2

    #R= math.atan(X/(Y+0.0001))

    return R3 #R1-R2



train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

train_qs = pd.Series(df_train['question1'].map(cleantxt).tolist() + df_train['question2'].map(cleantxt).tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].map(cleantxt).tolist() + df_test['question2'].map(cleantxt).tolist()).astype(str)



end = time.clock()

print('clean:',end-start)
from collections import Counter



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(train_qs)).split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}

end = time.clock()

print('def:',end-start)
# remove start words, so make all startwords stopwords

format = lambda x: x.split(' ', 1)[0]

merknamen=df_train['question1'].map(format)

merknamen=list(set(merknamen))



txt1=df_train['question1']

txt2=df_train['question2']

txt=txt1.append(pd.DataFrame(list(txt2)))

txt.columns=['q']

print(txt.head())



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(4,4),min_df=1,stop_words=merknamen)

tfidf.fit(txt.q)

tfidf_m = tfidf.transform(txt.q) #print('list of words', tfidf.vocabulary_.keys()) #print('list of nummers', tfidf.vocabulary_.values())

words = pd.DataFrame(pd.Series(list(tfidf.vocabulary_.keys()), index=tfidf.vocabulary_.values()),columns=['woord']) # #print(words)  #woordenboek !



def sort_coo(m):

    tuples = izip(m.row, m.col, m.data)

    return sorted(tuples, key=lambda x: (x[2]))



txt['core']=''

for xi in range(0,len(txt)):

    rij1=tfidf_m[xi][0:]

    print(sort_coo(coo_matrix(rij1)) )



    rij1['woord']=words['woord']

    toprij=rij1[rij1['plaats']>0]

    formathw = lambda x: len(x)

    toprij['lengte']=toprij['woord'].map(formathw)

    toprij=toprij[toprij['lengte']==toprij['lengte'].max()]['woord']

    if len(toprij)>0:

        print(list(toprij)[0])

        txt.set_value(xi, 'klas', list(toprij)[0])

    



print(txt)

end = time.clock()

print('tf:',end-start)
print('Most common words and weights: \n')

print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

print('\nLeast common words and weights: ')

(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
def tf(word,blob):

    return blob.words.count(word)/len(blob.words)

def n_containing(word,bloblist):

    return blob.words.count(word)/len(blob.words)

def idf(word,bloblist):

    return sum(1 for blob in bloblist if word in blob)

def tfidf(word,blob,bloblist):

    return tf(word,blob)*idf(word,bloblist)



def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).split():

            q1words[word] = 1

    for word in str(row['question2']).split():

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights_q1=[weights.get(w, 0) for w in q1words.keys() if w in q2words] 

    shared_weights_q2=[weights.get(w, 0) for w in q2words.keys() if w in q1words]

    nonshared_weights_q1 = [weights.get(w, 0) for w in q1words.keys() if w not in q2words]

    nonshared_weights_q2 =  [weights.get(w, 0) for w in q2words.keys() if w not in q1words]

    total_weights_q1 = [weights.get(w, 0) for w in q1words] 

    total_weights_q2 =[weights.get(w, 0) for w in q2words]

    X1 = np.sum(shared_weights_q1) / np.sum(total_weights_q1)

    Y1 = np.sum(nonshared_weights_q1) / np.sum(total_weights_q1)

    X2 = np.sum(shared_weights_q2) / np.sum(total_weights_q2)

    Y2 = np.sum(nonshared_weights_q2) / np.sum(total_weights_q2)

    #R1= math.atan(X1/(Y1+0.0001))

    #R2= math.atan(X2/(Y2+0.0001))

    R3 = (Y1+Y2)/2

    #X = np.sum(shared_weights) / np.sum(total_weights)

    #Y = np.sum(nonshared_weights) / np.sum(shared_weights)

    #R= math.atan(X/(Y+0.0001))

    return R3 #R1-R2



end = time.clock()

print('wordmatch:',end-start)
tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)



end = time.clock()

print('tfidf:',end-start)
from sklearn.metrics import roc_auc_score

print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))

print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))
# First we create our training and testing data

x_train = pd.DataFrame()

x_test = pd.DataFrame()

x_train['word_match'] = train_word_match

x_train['tfidf_word_match'] = tfidf_train_word_match

x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)



y_train = df_train['is_duplicate'].values

end = time.clock()

print('createtestdata:',end-start)
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

#params['eta'] = 0.12

#params['max_depth'] = 5



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

#bst = xgb.train(params, d_train, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)