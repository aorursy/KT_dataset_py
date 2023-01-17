import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline
df_train = pd.read_csv("../input/train.csv").fillna("")

df_test = pd.read_csv("../input/test.csv").fillna("")
from nltk.metrics import jaccard_distance
def build_dict(sentences):

#    from collections import OrderedDict



    '''

    Build dictionary of train words

    Outputs: 

     - Dictionary of word --> word index

     - Dictionary of word --> word count freq

    '''

    print('Building dictionary..'),

    wordcount = dict()

    #For each worn in each sentence, cummulate frequency

    for ss in sentences:

        for w in ss:

            if w not in wordcount:

                wordcount[w] = 1

            else:

                wordcount[w] += 1

    

    worddict = dict()

    for idx, w in enumerate(sorted(wordcount.items(), key = lambda x: x[1], reverse=True)):

        worddict[w[0]] = idx+2  # leave 0 and 1 (UNK)



    return worddict, wordcount
def generate_sequence(sentences, dictionary):

    '''

    Convert tokenized text in sequences of integers

    '''

    seqs = [None] * len(sentences)

    for idx, ss in enumerate(sentences):

        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in ss]



    return seqs
questions = list(df_train['question1']) + list(df_train['question2'])
def normalize(x):

    return x.lower().split()
tok_questions = [normalize(s) for s in questions]

worddict, wordcount = build_dict(tok_questions)
print(np.sum(list(wordcount.values())), ' total words ', len(worddict), ' unique words')
df_train['s_question1'] = generate_sequence(df_train['question1'].apply(normalize),worddict)

df_train['s_question2'] = generate_sequence(df_train['question2'].apply(normalize),worddict)
df_train.head()
def jc(x):

    return jaccard_distance(set(x['s_question1']),set(x['s_question2']))
df_train['jaccard'] = df_train.apply(jc,axis = 1)
def cosine_d(x):

    a = set(x['s_question1'])

    b = set(x['s_question2'])

    d = len(a)*len(b)

    if (d == 0):

        return 0

    else: 

        return len(a.intersection(b))/d
df_train['cosine'] = df_train.apply(cosine_d,axis = 1)
df_train.head()
df_test['s_question1'] = generate_sequence(df_test['question1'].apply(normalize),worddict)

df_test['s_question2'] = generate_sequence(df_test['question2'].apply(normalize),worddict)
df_test['jaccard'] = df_test.apply(jc,axis = 1)

df_test['cosine'] = df_test.apply(cosine_d,axis = 1)
df_test.head()
sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': 1 - df_test['jaccard']})

sub.to_csv('jaccard_submission.csv', index=False)

sub.head()
from sklearn.metrics import log_loss

log_loss(df_train['is_duplicate'], 1 - df_train['jaccard'])
log_loss(df_train['is_duplicate'], df_train['cosine'])
sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': df_test['cosine']})

sub.to_csv('cosine_submission.csv', index=False)

sub.head()
x_train = df_train[['jaccard','cosine']]
y_train = df_train['is_duplicate']
x_train.head()
from sklearn.cross_validation import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(df_train[['jaccard','cosine']], df_train['is_duplicate'], test_size=0.2, random_state=4242)
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
d_test = xgb.DMatrix(df_test[['jaccard','cosine']])

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)