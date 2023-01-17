!pip install ../input/sacremoses/sacremoses-master/ > /dev/null
import sys

import glob

import torch



sys.path.insert(0, "../input/transformers/transformers-master/")

import transformers

import math



import os

import re

import gc

import pickle  

import random

import string



import numpy as np

import pandas as pd

from scipy import stats



from scipy.stats import spearmanr, rankdata

from os.path import join as path_join

from numpy.random import seed

from urllib.parse import urlparse

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold



seed(42)

random.seed(42)



import nltk

from nltk.corpus import stopwords





# import category_encoders as ce



from sklearn.base import clone

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, RobustScaler, KBinsDiscretizer, QuantileTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold, GroupKFold

from sklearn.multioutput import MultiOutputRegressor

from sklearn.impute import SimpleImputer

from sklearn.metrics import make_scorer



from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, RANSACRegressor

from sklearn.svm import LinearSVR, SVR

from sklearn.ensemble import ExtraTreesRegressor



eng_stopwords = set(stopwords.words("english"))



import tensorflow as tf

import tensorflow_hub as hub
# settings

data_dir = '../input/google-quest-challenge/'

metas_dir = ''

sub_dir = ''



# data_dir = '../input/'

# metas_dir = '../metafeatures/'

# sub_dir = '../submissions/'



RANDOM_STATE = 42



import datetime

todate = datetime.date.today().strftime("%m%d")





nfolds = 5
# count words

def word_count(xstring):

    return xstring.split().str.len()
def spearman_corr(y_true, y_pred):

        if np.ndim(y_pred) == 2:

            corr = np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])

        else:

            corr = stats.spearmanr(y_true, y_pred)[0]

        return corr

    

custom_scorer = make_scorer(spearman_corr, greater_is_better=True)
def chunks(l, n):

    """Yield successive n-sized chunks from l."""

    for i in range(0, len(l), n):

        yield l[i:i + n]
def fetch_vectors(string_list, batch_size=64):

    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/

    DEVICE = torch.device("cuda")

    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")

    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")

    model.to(DEVICE)



    fin_features = []

    for data in chunks(string_list, batch_size):

        tokenized = []

        for x in data:

            x = " ".join(x.strip().split()[:300])

            tok = tokenizer.encode(x, add_special_tokens=True)

            tokenized.append(tok[:512])



        max_len = 512

        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])

        attention_mask = np.where(padded != 0, 1, 0)

        input_ids = torch.tensor(padded).to(DEVICE)

        attention_mask = torch.tensor(attention_mask).to(DEVICE)



        with torch.no_grad():

            last_hidden_states = model(input_ids, attention_mask=attention_mask)



        features = last_hidden_states[0][:, 0, :].cpu().numpy()

        fin_features.append(features)



    fin_features = np.vstack(fin_features)

    return fin_features
# load the data



xtrain = pd.read_csv(data_dir + 'train.csv')

xtest = pd.read_csv(data_dir + 'test.csv')

target_cols = ['question_asker_intent_understanding', 'question_body_critical', 

               'question_conversational', 'question_expect_short_answer', 

               'question_fact_seeking', 'question_has_commonly_accepted_answer', 

               'question_interestingness_others', 'question_interestingness_self', 

               'question_multi_intent', 'question_not_really_a_question', 

               'question_opinion_seeking', 'question_type_choice', 

               'question_type_compare', 'question_type_consequence', 

               'question_type_definition', 'question_type_entity', 

               'question_type_instructions', 'question_type_procedure', 

               'question_type_reason_explanation', 'question_type_spelling', 

               'question_well_written', 'answer_helpful', 

               'answer_level_of_information', 'answer_plausible', 

               'answer_relevance', 'answer_satisfaction', 

               'answer_type_instructions', 'answer_type_procedure', 

               'answer_type_reason_explanation', 'answer_well_written']
# word count in title, body and answer

for colname in ['question_title', 'question_body', 'answer']:

    newname = colname + '_word_len'

    

    xtrain[newname] = xtrain[colname].str.split().str.len()

    xtest[newname] = xtest[colname].str.split().str.len()



    

del newname, colname
for colname in ['question', 'answer']:



    # check for nonames, i.e. users with logins like user12389

    xtrain['is_'+colname+'_no_name_user'] = xtrain[colname +'_user_name'].str.contains('^user\d+$') + 0

    xtest['is_'+colname+'_no_name_user'] = xtest[colname +'_user_name'].str.contains('^user\d+$') + 0

    



colname = 'answer'

# check lexical diversity (unique words count vs total )

xtrain[colname+'_div'] = xtrain[colname].apply(lambda s: len(set(s.split())) / len(s.split()) )

xtest[colname+'_div'] = xtest[colname].apply(lambda s: len(set(s.split())) / len(s.split()) )

## domain components

xtrain['domcom'] = xtrain['question_user_page'].apply(lambda s: s.split('://')[1].split('/')[0].split('.'))

xtest['domcom'] = xtest['question_user_page'].apply(lambda s: s.split('://')[1].split('/')[0].split('.'))



# count components

xtrain['dom_cnt'] = xtrain['domcom'].apply(lambda s: len(s))

xtest['dom_cnt'] = xtest['domcom'].apply(lambda s: len(s))



# extend length

xtrain['domcom'] = xtrain['domcom'].apply(lambda s: s + ['none', 'none'])

xtest['domcom'] = xtest['domcom'].apply(lambda s: s + ['none', 'none'])



# components

for ii in range(0,4):

    xtrain['dom_'+str(ii)] = xtrain['domcom'].apply(lambda s: s[ii])

    xtest['dom_'+str(ii)] = xtest['domcom'].apply(lambda s: s[ii])

    

# clean up

xtrain.drop('domcom', axis = 1, inplace = True)

xtest.drop('domcom', axis = 1, inplace = True)
# shared elements

xtrain['q_words'] = xtrain['question_body'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )

xtrain['a_words'] = xtrain['answer'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )

xtrain['qa_word_overlap'] = xtrain.apply(lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis = 1)

xtrain['qa_word_overlap_norm1'] = xtrain.apply(lambda s: s['qa_word_overlap']/(1 + len(s['a_words'])), axis = 1)

xtrain['qa_word_overlap_norm2'] = xtrain.apply(lambda s: s['qa_word_overlap']/(1 + len(s['q_words'])), axis = 1)

xtrain.drop(['q_words', 'a_words'], axis = 1, inplace = True)



xtest['q_words'] = xtest['question_body'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )

xtest['a_words'] = xtest['answer'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )

xtest['qa_word_overlap'] = xtest.apply(lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis = 1)

xtest['qa_word_overlap_norm1'] = xtest.apply(lambda s: s['qa_word_overlap']/(1 + len(s['a_words'])), axis = 1)

xtest['qa_word_overlap_norm2'] = xtest.apply(lambda s: s['qa_word_overlap']/(1 + len(s['q_words'])), axis = 1)

xtest.drop(['q_words', 'a_words'], axis = 1, inplace = True)

## Number of characters in the text ##

xtrain["question_title_num_chars"] = xtrain["question_title"].apply(lambda x: len(str(x)))

xtest["question_title_num_chars"] = xtest["question_title"].apply(lambda x: len(str(x)))

xtrain["question_body_num_chars"] = xtrain["question_body"].apply(lambda x: len(str(x)))

xtest["question_body_num_chars"] = xtest["question_body"].apply(lambda x: len(str(x)))

xtrain["answer_num_chars"] = xtrain["answer"].apply(lambda x: len(str(x)))

xtest["answer_num_chars"] = xtest["answer"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

xtrain["question_title_num_stopwords"] = xtrain["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

xtest["question_title_num_stopwords"] = xtest["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

xtrain["question_body_num_stopwords"] = xtrain["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

xtest["question_body_num_stopwords"] = xtest["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

xtrain["answer_num_stopwords"] = xtrain["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

xtest["answer_num_stopwords"] = xtest["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

xtrain["question_title_num_punctuations"] =xtrain['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

xtest["question_title_num_punctuations"] =xtest['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

xtrain["question_body_num_punctuations"] =xtrain['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

xtest["question_body_num_punctuations"] =xtest['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

xtrain["answer_num_punctuations"] =xtrain['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

xtest["answer_num_punctuations"] =xtest['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

xtrain["question_title_num_words_upper"] = xtrain["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

xtest["question_title_num_words_upper"] = xtest["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

xtrain["question_body_num_words_upper"] = xtrain["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

xtest["question_body_num_words_upper"] = xtest["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

xtrain["answer_num_words_upper"] = xtrain["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

xtest["answer_num_words_upper"] = xtest["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
# train_question_body_dense = fetch_vectors(xtrain.question_body.values)

# train_answer_dense = fetch_vectors(xtrain.answer.values)



# test_question_body_dense = fetch_vectors(xtest.question_body.values)

# test_answer_dense = fetch_vectors(xtest.answer.values)
module_url = "../input/universalsentenceencoderlarge4/"

embed = hub.load(module_url)
embeddings_train = {}

embeddings_test = {}

for text in ['question_title', 'question_body', 'answer']:

    train_text = xtrain[text].str.replace('?', '.').str.replace('!', '.').tolist()

    test_text = xtest[text].str.replace('?', '.').str.replace('!', '.').tolist()

    

    curr_train_emb = []

    curr_test_emb = []

    batch_size = 4

    ind = 0

    while ind*batch_size < len(train_text):

        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())

        ind += 1

        

    ind = 0

    while ind*batch_size < len(test_text):

        curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())

        ind += 1    

        

    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)

    embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)



    print(text)

    

del embed
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)



cos_dist = lambda x, y: (x*y).sum(axis=1)



dist_features_train = np.array([

    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),

    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),

    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),

    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),

    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),

    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])

]).T



dist_features_test = np.array([

    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),

    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),

    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),

    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),

    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),

    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])

]).T



del embeddings_train, embeddings_test
for ii in range(0,6):

    xtrain['dist'+str(ii)] = dist_features_train[:,ii]

    xtest['dist'+str(ii)] = dist_features_test[:,ii]

    
limit_char = 5000

limit_word = 25000
title_col = 'question_title'

title_transformer = Pipeline([

    ('tfidf', TfidfVectorizer(lowercase = False, max_df = 0.3, min_df = 1,

                             binary = False, use_idf = True, smooth_idf = False,

                             ngram_range = (1,2), stop_words = 'english', 

                             token_pattern = '(?u)\\b\\w+\\b' , max_features = limit_word ))

])



        

title_transformer2 = Pipeline([

 ('tfidf2',  TfidfVectorizer( sublinear_tf=True,

    strip_accents='unicode', analyzer='char',

    stop_words='english', ngram_range=(1, 4), max_features= limit_char))   

])





body_col = 'question_body'

body_transformer = Pipeline([

    ('tfidf',TfidfVectorizer(lowercase = False, max_df = 0.3, min_df = 1,

                             binary = False, use_idf = True, smooth_idf = False,

                             ngram_range = (1,2), stop_words = 'english', 

                             token_pattern = '(?u)\\b\\w+\\b' , max_features = limit_word ))

])





body_transformer2 = Pipeline([

 ('tfidf2',  TfidfVectorizer( sublinear_tf=True,

    strip_accents='unicode', analyzer='char',

    stop_words='english', ngram_range=(1, 4), max_features= limit_char))   

])



answer_col = 'answer'



answer_transformer = Pipeline([

    ('tfidf', TfidfVectorizer(lowercase = False, max_df = 0.3, min_df = 1,

                             binary = False, use_idf = True, smooth_idf = False,

                             ngram_range = (1,2), stop_words = 'english', 

                             token_pattern = '(?u)\\b\\w+\\b' , max_features = limit_word ))

])



answer_transformer2 = Pipeline([

 ('tfidf2',  TfidfVectorizer( sublinear_tf=True,

    strip_accents='unicode', analyzer='char',

    stop_words='english', ngram_range=(1, 4), max_features= limit_char))   

])



num_cols = [

    'question_title_word_len', 'question_body_word_len', 'answer_word_len', 'answer_div',

    'question_title_num_chars','question_body_num_chars','answer_num_chars',

    'question_title_num_stopwords','question_body_num_stopwords','answer_num_stopwords',

    'question_title_num_punctuations','question_body_num_punctuations','answer_num_punctuations',

    'question_title_num_words_upper','question_body_num_words_upper','answer_num_words_upper',

    'dist0', 'dist1', 'dist2', 'dist3', 'dist4',       'dist5'

]



num_transformer = Pipeline([

    ('impute', SimpleImputer(strategy='constant', fill_value=0)),

    ('scale', PowerTransformer(method='yeo-johnson'))

])





cat_cols = [

    'dom_0', 

    'dom_1', 

    'dom_2', 

    'dom_3',     

    'category', 

    'is_question_no_name_user',

    'is_answer_no_name_user',

    'dom_cnt'

]



cat_transformer = Pipeline([

    ('impute', SimpleImputer(strategy='constant', fill_value='')),

    ('encode', OneHotEncoder(handle_unknown='ignore'))

])





preprocessor = ColumnTransformer(

    transformers = [

        ('title', title_transformer, title_col),

        ('title2', title_transformer2, title_col),

        ('body', body_transformer, body_col),

        ('body2', body_transformer2, body_col),

        ('answer', answer_transformer, answer_col),

        ('answer2', answer_transformer2, answer_col),

        ('num', num_transformer, num_cols),

        ('cat', cat_transformer, cat_cols)

    ]

)



pipeline = Pipeline([

    ('preprocessor', preprocessor),

    ('estimator',Ridge(random_state=RANDOM_STATE))

])
# created in previous version - just uploaded here 

vector_as = pd.read_csv('../input/alphas-vector/alphas_vector.csv')

# prep

id_train = xtrain['qa_id']

ytrain = xtrain[target_cols]

xtrain.drop(target_cols + ['qa_id'], axis = 1, inplace = True)





id_test = xtest['qa_id'] 

xtest.drop('qa_id', axis = 1, inplace = True)
dropcols = ['question_user_name', 'question_user_page',

 'answer_user_name', 'answer_user_page','url','host']



xtrain.drop(dropcols, axis = 1, inplace = True)

xtest.drop(dropcols, axis = 1, inplace = True)

mvalid = np.zeros((xtrain.shape[0], len(target_cols)))

mfull = np.zeros((xtest.shape[0], len(target_cols)))

kf = KFold(n_splits = nfolds)



kf = GroupKFold(n_splits=5).split(X=xtrain.question_body, groups=xtrain.question_body)

# for train_index, test_index in kf.split(xtrain):

    

for ind, (train_index, test_index) in enumerate(kf):

    

    print('---')

    # split

    x0, x1 = xtrain.loc[train_index], xtrain.loc[test_index]

    y0, y1 = ytrain.loc[train_index], ytrain.loc[test_index]



    for ii in range(0, ytrain.shape[1]):



        # fit model

        be = clone(pipeline)

        be.steps[1][1].alpha = vector_as.loc[ii]

        be.fit(x0, np.array(y0)[:,ii])



        filename = 'ridge_f'+str(ii)+'.pkl'

        pickle.dump(be, open(filename, 'wb'))

        

        # park forecast

        mvalid[test_index, ii] = be.predict(x1)

        mfull[:,ii] += be.predict(xtest)/5

 
corvec = np.zeros((ytrain.shape[1],1))

for ii in range(0, ytrain.shape[1]):

    mvalid[:,ii] = rankdata(mvalid[:,ii])/mvalid.shape[0]

    mfull[:,ii] = rankdata(mfull[:,ii])/mfull.shape[0]

    

    corvec[ii] = stats.spearmanr(ytrain[ytrain.columns[ii]], mvalid[:,ii])[0]

    

print(corvec.mean())
prval = pd.DataFrame(mvalid)

prval.columns = ytrain.columns

prval['qa_id'] = id_train

prval = prval[['qa_id'] + list(prval.columns[:-1])]

prval.to_csv(metas_dir + 'prval_ridge_'+todate+ '.csv', index = False)





prfull = pd.DataFrame(mfull)

prfull.columns = ytrain.columns

prfull['qa_id'] = id_test

prfull = prfull[['qa_id'] + list(prfull.columns[:-1])]

prfull.to_csv(metas_dir + 'prfull_ridge_'+todate+ '.csv', index = False)

prfull.to_csv(sub_dir + 'submission.csv', index = False)