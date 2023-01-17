import os

import re

import gc

import pickle  

import random

import keras



import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import keras.backend as K



from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda

from keras.optimizers import Adam

from keras.callbacks import Callback

from scipy.stats import spearmanr, rankdata

from os.path import join as path_join

from numpy.random import seed

from urllib.parse import urlparse

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold

from sklearn.linear_model import MultiTaskElasticNet

import lightgbm as lgb

from sklearn import metrics



seed(42)

tf.random.set_seed(42)

random.seed(42)
def read_data():

    data_dir = '../input/google-quest-challenge/'

    print('Reading train set')

    train = pd.read_csv(path_join(data_dir, 'train.csv'))

    print('Reading test set')

    test = pd.read_csv(path_join(data_dir, 'test.csv'))

    print('Our training data have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Our testing data have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test
# 30 target variables where the range is [0, 1]

targets = ['question_asker_intent_understanding', 'question_body_critical', 'question_conversational', 

           'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer', 

           'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent', 

           'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 

           'question_type_compare', 'question_type_consequence', 'question_type_definition', 

           'question_type_entity', 'question_type_instructions', 'question_type_procedure', 

           'question_type_reason_explanation', 'question_type_spelling', 'question_well_written', 

           'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance', 

           'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure', 

           'answer_type_reason_explanation', 'answer_well_written']



# text features, using embedding trained model to preprocess them

input_columns = ['question_title', 'question_body', 'answer']
# get categorical features and ohe

def get_cat_features(train, test):

    find = re.compile(r"^[^.]*")



    train['netloc'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

    test['netloc'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])



    features = ['netloc', 'category']

    merged = pd.concat([train[features], test[features]])

    ohe = OneHotEncoder()

    ohe.fit(merged)

    

    # one hot encode netloc and category

    features_train = ohe.transform(train[features]).toarray()

    features_test = ohe.transform(test[features]).toarray()

    print('Our categorical features have {} rows and {} columns'.format(features_train.shape[0], features_train.shape[1]))

    return features_train, features_test
def get_embedding_features(train, test, input_columns):

    

    # load universal sentence encoder model to get sentence ambeddings

    module_url = "../input/universalsentenceencoderlarge4/"

    embed = hub.load(module_url)

    

    # create empty dictionaries to store final results

    embedding_train = {}

    embedding_test = {}



    # iterate over text columns to get senteces embeddings with the previous loaded model

    for text in input_columns:

    

        print(text)

        train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()

        test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()

    

        # create empy list to save each batch

        curr_train_emb = []

        curr_test_emb = []

    

        # define a batch to transform senteces to their correspinding embedding (1 X 512 for each sentece)

        batch_size = 4

        ind = 0

        while ind * batch_size < len(train_text):

            curr_train_emb.append(embed(train_text[ind * batch_size: (ind + 1) * batch_size])['outputs'].numpy())

            ind += 1

        

        ind = 0

        while ind * batch_size < len(test_text):

            curr_test_emb.append(embed(test_text[ind * batch_size: (ind + 1) * batch_size])['outputs'].numpy())

            ind += 1



        # stack arrays to get a 2D array (dataframe) corresponding with all the sentences and dim 512 for columns (sentence encoder output)

        embedding_train[text + '_embedding'] = np.vstack(curr_train_emb)

        embedding_test[text + '_embedding'] = np.vstack(curr_test_emb)

    

    del embed

    K.clear_session()

    gc.collect()

    

    return embedding_train, embedding_test
def get_dist_features(embedding_train, embedding_test, features_train, features_test):

    

    # define a square dist lambda function were (x1 - y1) ^ 2 + (x2 - y2) ^ 2 + (x3 - y3) ^ 2 + ... + (xn - yn) ^ 2

    # with this we get one vector of dimension 6079

    l2_dist = lambda x, y: np.power(x - y, 2).sum(axis = 1)

    

    # define a cosine dist lambda function were (x1 * y1) ^ 2 + (x2 * y2) + (x3 * y3) + ... + (xn * yn)

    cos_dist = lambda x, y: (x * y).sum(axis = 1)

    

    # transpose it because we have 6 vector of dimension 6079, need 6079 x 6

    dist_features_train = np.array([

        l2_dist(embedding_train['question_title_embedding'], embedding_train['answer_embedding']),

        l2_dist(embedding_train['question_body_embedding'], embedding_train['answer_embedding']),

        l2_dist(embedding_train['question_body_embedding'], embedding_train['question_title_embedding']),

        cos_dist(embedding_train['question_title_embedding'], embedding_train['answer_embedding']),

        cos_dist(embedding_train['question_body_embedding'], embedding_train['answer_embedding']),

        cos_dist(embedding_train['question_body_embedding'], embedding_train['question_title_embedding'])]).T

    

    # transpose it because we have 6 vector of dimension 6079, need 6079 x 6

    dist_features_test = np.array([

        l2_dist(embedding_test['question_title_embedding'], embedding_test['answer_embedding']),

        l2_dist(embedding_test['question_body_embedding'], embedding_test['answer_embedding']),

        l2_dist(embedding_test['question_body_embedding'], embedding_test['question_title_embedding']),

        cos_dist(embedding_test['question_title_embedding'], embedding_test['answer_embedding']),

        cos_dist(embedding_test['question_body_embedding'], embedding_test['answer_embedding']),

        cos_dist(embedding_test['question_body_embedding'], embedding_test['question_title_embedding'])]).T

    

    # get the values of each key, therefore we have 3 arrays of dim 6079 x 512 (hstack to get 6079 x 1536)

    x_train = np.hstack([values for key, values in embedding_train.items()] + [features_train, dist_features_train])

    x_test = np.hstack([values for key, values in embedding_test.items()] + [features_test, dist_features_test])

    print('Our preprocess training set have {} rows and {} columns'.format(x_train.shape[0], x_train.shape[1]))

    print('Our preprocess testing set have {} rows and {} columns'.format(x_test.shape[0], x_test.shape[1]))

    

    return pd.DataFrame(x_train), pd.DataFrame(x_test)
def run_lgb(x_train, x_test, train, target):

    

    # 5 random KFold

    kf = KFold(n_splits = 10, random_state = 42)

    

    # get empty vectors (0) to store out of fold predictions and test predictions

    oof_pred = np.zeros(len(x_train))

    y_pred = np.zeros(len(x_test))

    

    # for each fold train a model and then get the mean to predict the test

    for fold, (tr_ind, val_ind) in enumerate(kf.split(x_train)):

        print('Fold {}'.format(fold + 1))

        x_trn, x_val = x_train.iloc[tr_ind], x_train.iloc[val_ind]

        y_trn, y_val = train[target].iloc[tr_ind], train[target].iloc[val_ind]

        train_set = lgb.Dataset(x_trn, y_trn)

        val_set = lgb.Dataset(x_val, y_val)



        params = {

            'learning_rate': 0.1,

            'metric': 'rmse',

            'objective': 'regression',

            'feature_fraction': 0.85,

            'subsample': 0.85,

            'n_jobs': -1,

            'seed': 42,

            'max_depth': -1

        }

        

        # train the model with early stoping

        model = lgb.train(params, train_set, num_boost_round = 1000000, early_stopping_rounds = 10, 

                          valid_sets=[train_set, val_set], verbose_eval = 10)

        oof_pred[val_ind] = model.predict(x_val)

        y_pred += model.predict(x_test) / kf.n_splits

    loss_score = np.sqrt(metrics.mean_squared_error(train[target], oof_pred))

    print('Our oof rmse score is: ', loss_score)

    return y_pred



def predict(x_train, x_test, train, targets):

    predictions = []

    for num, target in enumerate(targets):

        print('Train model {}'.format(num + 1))

        print('Predicting target {}'.format(target))

        predictions.append(np.clip(run_lgb(x_train, x_test, train, target), a_min = 0, a_max = 1))

    data_dir = '../input/google-quest-challenge/'

    submission = pd.read_csv(path_join(data_dir, 'sample_submission.csv'))

    submission[targets] = np.array(predictions).T

    submission.to_csv('submission.csv', index = False)
# run all

# read training and test data

train, test = read_data()

# get one hot encoded categorical data

features_train, features_test = get_cat_features(train, test)

# get embedding features

embedding_train, embedding_test = get_embedding_features(train, test, input_columns)

# get dist features from the embedding features and concatenate with embedding and categorical features

x_train, x_test = get_dist_features(embedding_train, embedding_test, features_train, features_test)

# trian and predict

predict(x_train, x_test, train, targets)