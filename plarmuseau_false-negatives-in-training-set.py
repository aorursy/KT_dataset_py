import time

start = time.clock()



#open data

import codecs

import nltk #language functions

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



datas = pd.read_csv('../input/train.csv')[:100000]

datas = datas.fillna('leeg')

#print(datas.head())



def cleantxt(x):    # aangeven sentence

    x = x.lower()

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x



datas['question1']=datas['question1'].map(cleantxt)

datas['question2']=datas['question2'].map(cleantxt)





end = time.clock()

print('open:',end-start)



#datas
 # Import the stop word list

#print stopwords.words("english") 



for xyz in range(0,1000):

    q1=datas.iloc[xyz].question1

    q2=datas.iloc[xyz].question2

    sent1=q1.split()

    sent2=q2.split()

    equq1 = [w for w in sent1 if w in sent2]

    difq1 = [w for w in sent1 if w not in sent2]

    difq2 = [w for w in sent2 if w not in sent1]

    diftot = difq1+difq2

    difton = [w for w in diftot if not w in stopwords.words("english")]

    if len(difton)==0 and datas.iloc[xyz].is_duplicate==0:

        print('false negative ?',q1,q2,datas.iloc[xyz].is_duplicate)

    

end = time.clock()

print('all dubious:',end-start)
df_train = pd.read_csv('../input/train.csv',encoding='utf8')[:10000]

df_train = df_train.fillna('leeg')

df_test = pd.read_csv('../input/test.csv',encoding='utf8')[:50000]

df_test = df_test.fillna('leeg')

df_train.head(2)

df_test.head(2)

end = time.clock()

print('open:',end-start)



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

    X1 = (len(shared_words_in_q1))/(len(q1words) )

    Y1 = (len(nonshared_words_in_q1))/(len(q1words) )

    X2 = (len(shared_words_in_q2))/(len(q2words))

    Y2 = (len(nonshared_words_in_q2))/(len(q2words))

    R3 = (Y1+Y2)/2  

    diftot=nonshared_words_in_q1+nonshared_words_in_q2

    difton = [w for w in diftot if not w in stopwords.words("english")]  #if the difference is only stopwords

    

    if len(difton)==0:

        print(row['id'],row['is_duplicate'],row['question1'],row['question2'])

        if row['is_duplicate']==0:

            df_train.set_value(row['id'], 'is_duplicate', 1)  #replace 0 with 1

        R3=1        



    return R3 #R1-R2



def word_match_share2(row):

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

    X1 = (len(shared_words_in_q1))/(len(q1words) )

    Y1 = (len(nonshared_words_in_q1))/(len(q1words) )

    X2 = (len(shared_words_in_q2))/(len(q2words))

    Y2 = (len(nonshared_words_in_q2))/(len(q2words))

    R3 = (Y1+Y2)/2  

    diftot=nonshared_words_in_q1+nonshared_words_in_q2

    difton = [w for w in diftot if not w in stopwords.words("english")]  #if the difference is only stopwords

    

    if len(difton)==0:

        R3=1        



    return R3 #R1-R2



train_qs = pd.Series(df_train['question1'].map(cleantxt).tolist() + df_train['question2'].map(cleantxt).tolist()).astype(str)

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)  #has to be after cleanign so that the splitting works better

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

        #if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).split():

        #if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights_q1=[weights.get(w, 0) for w in q1words.keys() if w in q2words] 

    shared_weights_q2=[weights.get(w, 0) for w in q2words.keys() if w in q1words]

    nonshared_weights_q1 = [weights.get(w, 0) for w in q1words.keys() if w not in q2words]

    nonshared_weights_q2 =  [weights.get(w, 0) for w in q2words.keys() if w not in q1words]

    nonshared_words_in_q1 = [w for w in q1words.keys() if w not in q2words]

    nonshared_words_in_q2 = [w for w in q2words.keys() if w not in q1words]    

    total_weights_q1 = [weights.get(w, 0) for w in q1words] 

    total_weights_q2 =[weights.get(w, 0) for w in q2words]

    X1 = np.sum(shared_weights_q1) / np.sum(total_weights_q1)

    Y1 = np.sum(nonshared_weights_q1) / np.sum(total_weights_q1)

    X2 = np.sum(shared_weights_q2) / np.sum(total_weights_q2)

    Y2 = np.sum(nonshared_weights_q2) / np.sum(total_weights_q2)

    R3 = (Y1+Y2)/2

    diftot=nonshared_words_in_q1+nonshared_words_in_q2

    difton = [w for w in diftot if not w in stopwords.words("english")]  #if the difference is only stopwords



    if len(difton)==0:

         R3=1    

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

x_test['word_match'] = df_test.apply(word_match_share2, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)



y_train = df_train['is_duplicate'].values

end = time.clock()

print('createtestdata:',end-start)
y_train = df_train['is_duplicate'].values

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

params['eta'] = 0.05

params['max_depth'] = 5



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=100, verbose_eval=25)
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)