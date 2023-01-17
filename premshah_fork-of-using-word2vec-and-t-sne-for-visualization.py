# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data.head()
##data_test = pd.read_csv("../input/test.csv")

#data_test.head()
STOP_WORDS = nltk.corpus.stopwords.words()



def clean_sentence(val):

    "remove chars that are not letters or numbers, downcase, then remove stop words"

    regex = re.compile('([^\s\w]|_)+')

    sentence = regex.sub('', val).lower()

    sentence = sentence.split(" ")

    

    for word in list(sentence):

        if word in STOP_WORDS:

            sentence.remove(word)  

            

    sentence = " ".join(sentence)

    return sentence



def clean_dataframe(data):

    "drop nans, then apply 'clean_sentence' function to question1 and 2"

    data = data.dropna(how="any")

    

    for col in ['question1', 'question2']:

        data[col] = data[col].apply(clean_sentence)

    

    return data



data = clean_dataframe(data)

data.head(5)
##data_test = clean_dataframe(data_test)

#data_test.head()
def build_corpus(data):

    corpus = []

    

    for col in ['question1','question2']:

         for sentence in data[col].iteritems():

            word_list = sentence[1].split(" ")

            corpus.append(word_list)

     

    return corpus



corpus = build_corpus(data)        

corpus[0:2]
#corpus_test = build_corpus(data_test)

#corpus_test[0:2]
def build_corpus_q(data):

    corpus = []

    for sentence in data.iteritems():

            #word_list = sentence[1].split(" ")

            #corpus.append(word_list)

            corpus.append(sentence[1])

     

    return corpus



corpus_q1 = build_corpus_q(data['question1'])

corpus_q2 = build_corpus_q(data['question2'])
#corpus_test_q1 = build_corpus_q(data_test['question1'])

#corpus_test_q2 = build_corpus_q(data_test['question2'])
from gensim.models import word2vec



from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline



model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)

model.wv['trump']
##model_test = word2vec.Word2Vec(corpus_test, size=100, window=20, min_count=200, workers=4)

#model_test.wv['trump']
def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
tsne_plot(model)
#tsne_plot(model_test)
# A more selective model

#model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=1000, workers=4)

#tsne_plot(model)
model.most_similar('india')
def get_tsne_vector(model):

    "Creates a TSNE model and use it with Word2Vec to find how similar both questions are"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

    

    return labels, x, y
words_set, X, Y = get_tsne_vector(model)
#words_test, X_test, Y_test = get_tsne_vector(model_test)
train_qs = pd.Series(data['question1'].tolist() + data['question2'].tolist()).astype(str)

from collections import Counter



def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 



words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
#train_qs_test = pd.Series(data_test['question1'].tolist() + data_test['question2'].tolist()).astype(str)

#from collections import Counter



#def get_weight(count, eps=10000, min_count=2):

#    if count < min_count:

#        return 0

#    else:

#        return 1 / (count + eps)



#eps = 5000 



#words_test = (" ".join(train_qs_test)).lower().split()

#counts_test = Counter(words_test)

#weights_test = {word: get_weight(count) for word, count in counts_test.items()}
def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in STOP_WORDS:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in STOP_WORDS:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R
##def tfidf_word_match_share_test(row):

#    q1words = {}

#    q2words = {}

#    for word in str(row['question1']).lower().split():

#        if word not in STOP_WORDS:

#            q1words[word] = 1

#    for word in str(row['question2']).lower().split():

#        if word not in stoSTOP_WORDSps:

#           q2words[word] = 1

#    if len(q1words) == 0 or len(q2words) == 0:

#        # The computer-generated chaff includes a few questions that are nothing but stopwords

#        return 0

    

#    shared_weights = [weights_test.get(w, 0) for w in q1words.keys() if w in q2words] + [weights_test.get(w, 0) for w in q2words.keys() if w in q1words]

#    total_weights = [weights_test.get(w, 0) for w in q1words] + [weights_test.get(w, 0) for w in q2words]

    

#    R = np.sum(shared_weights) / np.sum(total_weights)

#    return R
tfidf_train_word_match = data.apply(tfidf_word_match_share, axis=1, raw=True)
#tfidf_train_word_match_test = data_test.apply(tfidf_word_match_share_test, axis=1, raw=True)
def word2vec_and_tfidf(data,words,X,Y,tfidf_train_word_match):

    val = []

    for index, row in data.iterrows():

        val1x, val1y = findVal(row['question1'],words,X,Y,tfidf_train_word_match,index)

        val2x, val2y = findVal(row['question2'],words,X,Y,tfidf_train_word_match,index)

        temp = np.square(val1x-val2x) + np.square(val1y-val2y)

        val.append(temp)

      

    return val
def findVal(ques,words,X,Y,tfidf,i):

    valx = 0

    valy = 0

    for wrd in list(ques.split(" ")):

        if wrd in words:

            index = words.index(wrd)

            valx = valx + tfidf[i]*X[index]

            valy = valy + tfidf[i]*Y[index]

    

    return valx, valy
data['val'] = word2vec_and_tfidf(data,words_set,X,Y,tfidf_train_word_match)

data.head()
data[data['is_duplicate']==1]
#data.to_csv("data.csv", sep='|', encoding='utf-8')
#temp = pd.read_csv('data.csv')

#temp.head()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))

data['len_q2'] = data.question2.apply(lambda x: len(str(x)))

data['diff_len'] = data.len_q1 - data.len_q2

data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ','')))))

data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ','')))))

data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))

data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))

data['common_words'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)
from fuzzywuzzy import fuzz



data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_wratio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_partial_token_set_ratio'] = data.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_partial_token_sort_ratio'] = data.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']),str(x['question2'])),axis=1)

data.head()
data.fillna(0, inplace=True)
train, valid = train_test_split(data, test_size = 0.2)

x_valid = valid

x_train = train

y_valid = valid['is_duplicate']

y_train = train['is_duplicate']

#x_train.drop(['id','qid1','qid2','question1','question2','is_duplicate'],axis=1,inplace=True)

#x_test.drop(['id','qid1','qid2','question1','question2','is_duplicate'],axis=1,inplace=True)
#x_train.head()

#x_test.head()
x_train.drop(['question1','question2','is_duplicate'],axis=1,inplace=True)

x_valid.drop(['question1','question2','is_duplicate'],axis=1,inplace=True)
model = RandomForestClassifier(100,oob_score=True)

model.fit(x_train,y_train)

model.score(x_valid,y_valid)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
