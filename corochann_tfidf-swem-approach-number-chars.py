import numpy as np

import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import gensim

from nltk.corpus import brown

import random

from sklearn.model_selection import KFold

import lightgbm as lgb

import gc

from keras.callbacks.callbacks import EarlyStopping

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.callbacks.callbacks import EarlyStopping

from scipy.stats import spearmanr

from nltk.corpus import wordnet as wn

import tqdm

from sklearn.model_selection import StratifiedKFold
train = pd.read_csv("../input/google-quest-challenge/train.csv")

test = pd.read_csv("../input/google-quest-challenge/test.csv")
sample_sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
sample_sub 
target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
train.head()
def char_count(s):

    return len(s)



def word_count(s):

    return s.count(' ')



train['question_title_n_chars'] = train['question_title'].apply(char_count)

train['question_title_n_words'] = train['question_title'].apply(word_count)

train['question_body_n_chars'] = train['question_body'].apply(char_count)

train['question_body_n_words'] = train['question_body'].apply(word_count)

train['answer_n_chars'] = train['answer'].apply(char_count)

train['answer_n_words'] = train['answer'].apply(word_count)



test['question_title_n_chars'] = test['question_title'].apply(char_count)

test['question_title_n_words'] = test['question_title'].apply(word_count)

test['question_body_n_chars'] = test['question_body'].apply(char_count)

test['question_body_n_words'] = test['question_body'].apply(word_count)

test['answer_n_chars'] = test['answer'].apply(char_count)

test['answer_n_words'] = test['answer'].apply(word_count)



train['question_body_n_chars'].clip(0, 5000, inplace=True)

test['question_body_n_chars'].clip(0, 5000, inplace=True)

train['question_body_n_words'].clip(0, 1000, inplace=True)

test['question_body_n_words'].clip(0, 1000, inplace=True)



train['answer_n_chars'].clip(0, 5000, inplace=True)

test['answer_n_chars'].clip(0, 5000, inplace=True)

train['answer_n_words'].clip(0, 1000, inplace=True)

test['answer_n_words'].clip(0, 1000, inplace=True)

num_question = train['question_user_name'].value_counts()

num_answer = train['answer_user_name'].value_counts()



train['num_answer_user'] = train['answer_user_name'].map(num_answer)

train['num_question_user'] = train['question_user_name'].map(num_question)

test['num_answer_user'] = test['answer_user_name'].map(num_answer)

test['num_question_user'] = test['question_user_name'].map(num_question)



# map is done by train data, we need to fill value for user which does not appear in train data...

test['num_answer_user'].fillna(1, inplace=True)

test['num_question_user'].fillna(1, inplace=True)
simple_feature_cols = [

    'question_title_n_chars', 'question_title_n_words', 'question_body_n_chars', 'question_body_n_words',

    'answer_n_chars', 'answer_n_words', 'num_answer_user', 'num_question_user'

]

simple_engineered_feature = train[simple_feature_cols].values

simple_engineered_feature_test = test[simple_feature_cols].values
from sklearn.preprocessing import MaxAbsScaler



scaler = MaxAbsScaler()

simple_engineered_feature = scaler.fit_transform(simple_engineered_feature)

simple_engineered_feature_test = scaler.transform(simple_engineered_feature_test)
def simple_prepro(s):

    return [w for w in s.replace("\n"," ").replace(","," , ").replace("("," ( ").replace(")"," ) ").

            replace("."," . ").replace("?"," ? ").replace(":"," : ").replace("n't"," not").

            replace("'ve"," have").replace("'re"," are").replace("'s"," is").split(" ") if w != ""]
def simple_prepro_tfidf(s):

    return " ".join([w for w in s.lower().replace("\n"," ").replace(","," , ").replace("("," ( ").replace(")"," ) ").

            replace("."," . ").replace("?"," ? ").replace(":"," : ").replace("n't"," not").

            replace("'ve"," have").replace("'re"," are").replace("'s"," is").split(" ") if w != ""])
qt_max = max([len(simple_prepro(l)) for l in list(train["question_title"].values)])

qb_max = max([len(simple_prepro(l))  for l in list(train["question_body"].values)])

an_max = max([len(simple_prepro(l))  for l in list(train["answer"].values)])

print("max lenght of question_title is",qt_max)

print("max lenght of question_body is",qb_max)

print("max lenght of question_answer is",an_max)
w2v_model = gensim.models.Word2Vec(brown.sents())
def get_word_embeddings(text):

    np.random.seed(abs(hash(text)) % (10 ** 8))

    words = simple_prepro(text)

    vectors = np.zeros((len(words),100))

    if len(words)==0:

        vectors = np.zeros((1,100))

    for i,word in enumerate(simple_prepro(text)):

        try:

            vectors[i]=w2v_model[word]

        except:

            vectors[i]=np.random.uniform(-0.01, 0.01,100)

            #np.array([len(text)/5000,len(words)/1000,text.count("\n")/10])]

    return np.max(np.array(vectors), axis=0)

                           
question_title = [get_word_embeddings(l) for l in tqdm.tqdm(train["question_title"].values)]

question_title_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["question_title"].values)]



question_body = [get_word_embeddings(l) for l in tqdm.tqdm(train["question_body"].values)]

question_body_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["question_body"].values)]



answer = [get_word_embeddings(l) for l in tqdm.tqdm(train["answer"].values)]

answer_test = [get_word_embeddings(l) for l in tqdm.tqdm(test["answer"].values)]
gc.collect()

tfidf = TfidfVectorizer(ngram_range=(1, 3))

tsvd = TruncatedSVD(n_components = 50)

tfidf_question_title = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_title"].values)])

tfidf_question_title_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_title"].values)])

tfidf_question_title = tsvd.fit_transform(tfidf_question_title)

tfidf_question_title_test = tsvd.transform(tfidf_question_title_test)



tfidf_question_body = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["question_body"].values)])

tfidf_question_body_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["question_body"].values)])

tfidf_question_body = tsvd.fit_transform(tfidf_question_body)

tfidf_question_body_test = tsvd.transform(tfidf_question_body_test)



tfidf_answer = tfidf.fit_transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(train["answer"].values)])

tfidf_answer_test = tfidf.transform([simple_prepro_tfidf(l) for l in tqdm.tqdm(test["answer"].values)])

tfidf_answer = tsvd.fit_transform(tfidf_answer)

tfidf_answer_test = tsvd.transform(tfidf_answer_test)
type2int = {type:i for i,type in enumerate(list(set(train["category"])))}

cate = np.identity(5)[np.array(train["category"].apply(lambda x:type2int[x]))].astype(np.float64)

cate_test = np.identity(5)[np.array(test["category"].apply(lambda x:type2int[x]))].astype(np.float64)
train_features = np.concatenate([question_title, question_body, answer,

                                 tfidf_question_title, tfidf_question_body, tfidf_answer, 

                                 cate, simple_engineered_feature

                                ], axis=1)

test_features = np.concatenate([question_title_test, question_body_test, answer_test, 

                               tfidf_question_title_test, tfidf_question_body_test, tfidf_answer_test,

                                cate_test, simple_engineered_feature_test

                                ], axis=1)
num_folds = 10

fold_scores = []

kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)

test_preds = np.zeros((len(test_features), len(target_cols)))

for train_index, val_index in kf.split(train_features):

    gc.collect()

    train_X = train_features[train_index, :]

    train_y = train[target_cols].iloc[train_index]

    

    val_X = train_features[val_index, :]

    val_y = train[target_cols].iloc[val_index]

    

    model = Sequential([

        Dense(512, input_shape=(train_features.shape[1],)),

        Activation('relu'),

        Dense(128),

        Activation('relu'),

        Dense(len(target_cols)),

        Activation('sigmoid'),

    ])

    

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model.compile(optimizer='adam',

                  loss='binary_crossentropy')

    

    model.fit(train_X, train_y, epochs = 100, validation_data=(val_X, val_y), callbacks = [es])

    preds = model.predict(val_X)

    overall_score = 0

    for col_index, col in enumerate(target_cols):

        overall_score += spearmanr(preds[:, col_index], val_y[col].values).correlation/len(target_cols)

        print(col, spearmanr(preds[:, col_index], val_y[col].values).correlation)

    fold_scores.append(overall_score)

    print(overall_score)



    test_preds += model.predict(test_features)/num_folds

    

print(fold_scores)
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

for col_index, col in enumerate(target_cols):

    sub[col] = test_preds[:, col_index]

sub.to_csv("submission.csv", index = False)
test_preds
sub.isna().sum()
import seaborn as sns



fig, axes = plt.subplots(6, 5, figsize=(18, 15))

axes = axes.ravel()

bins = np.linspace(0, 1, 20)



for i, col in enumerate(target_cols):

    ax = axes[i]

    sns.distplot(train[col], label=col, bins=bins, ax=ax, color='blue')

    sns.distplot(sub[col], label=col, bins=bins, ax=ax, color='orange')

    # ax.set_title(col)

    ax.set_xlim([0, 1])

plt.tight_layout()

plt.show()

plt.close()