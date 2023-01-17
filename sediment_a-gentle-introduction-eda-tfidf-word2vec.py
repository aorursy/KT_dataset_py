import numpy as np

import pandas as pd

import gensim

import matplotlib.pyplot as plt

import random

import gc

import seaborn as sns

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

random.seed(1234)
train_df = pd.read_csv('../input/google-quest-challenge/train.csv')

test_df = pd.read_csv('../input/google-quest-challenge/test.csv')



# the dataset size

print('train set size is %d' % len(train_df))

print('test set size is %d' % len(test_df))
feat_cols = [

    'question_title', 

    'question_body', 

    'question_user_name', 

    'question_user_page', 

    'answer', 

    'answer_user_name', 

    'answer_user_page', 

    'url', 

    'category', 

    'host']

target_cols = [

    'question_asker_intent_understanding', 

    'question_body_critical', 

    'question_conversational', 

    'question_expect_short_answer', 

    'question_fact_seeking', 

    'question_has_commonly_accepted_answer',

    'question_interestingness_others', 

    'question_interestingness_self', 

    'question_multi_intent', 

    'question_not_really_a_question', 

    'question_opinion_seeking', 

    'question_type_choice',

    'question_type_compare', 

    'question_type_consequence', 

    'question_type_definition', 

    'question_type_entity', 

    'question_type_instructions', 

    'question_type_procedure', 

    'question_type_reason_explanation', 

    'question_type_spelling',

    'question_well_written', 

    'answer_helpful', 

    'answer_level_of_information', 

    'answer_plausible', 

    'answer_relevance', 

    'answer_satisfaction', 

    'answer_type_instructions', 

    'answer_type_procedure', 

    'answer_type_reason_explanation',

    'answer_well_written'

]



print('we have %d feature columns and %d target columns' % (len(feat_cols) , len(target_cols)))
peek = train_df.sample()

text_cols = [

    'question_title',

    'question_body',

    'answer'

]



for col_name in feat_cols + target_cols:

    print(col_name)

    print('='* 10)

    print(str(peek[col_name].values[0]) + '\n')
train_df.groupby('question_title').count()['qa_id'].sort_values(ascending=False)
# take a closer look to the most popular question

train_df[train_df['question_title'] == 'What is the best introductory Bayesian statistics textbook?'][feat_cols]
train_df.groupby('category').count()['qa_id'].sort_values(ascending=False).plot(kind='bar', alpha=0.5)
train_df.groupby('host').count()['qa_id'].sort_values(ascending=False).plot(kind='bar', figsize=(16, 6), fontsize=15, alpha=0.5)
train_df.groupby('answer_user_name').count()['qa_id'].sort_values(ascending=False)[:5]
train_df[train_df['answer_user_name'] == 'Scott'][['answer_user_name', 'question_title', 'category', 'host']].sort_values(by='category')
train_df.groupby(['question_user_name']).count()['qa_id'].sort_values(ascending=False).iloc[:5]
train_df[train_df['question_user_name'] == 'Mike'][['question_user_name', 'question_title', 'category', 'host']].sort_values(by=['category', 'question_title'])
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords



TOKENIZER = RegexpTokenizer(r'\w+')

STOPWORDS = set(stopwords.words('english'))
train_df['question_title_len'] = train_df['question_title'].map(lambda x: len(TOKENIZER.tokenize(x)))

train_df['question_body_len'] = train_df['question_body'].map(lambda x: len(TOKENIZER.tokenize(x)))

train_df['answer_len'] = train_df['answer'].map(lambda x: len(TOKENIZER.tokenize(x)))
train_df[['answer_len', 'question_body_len', 'question_title_len']].plot(kind='box', showfliers=False)
def gen_word_cloud(col):

    rows = train_df[col].map(lambda x: TOKENIZER.tokenize(x)).values.tolist()

    words = []

    for row in rows:

        for w in row:

            if w not in STOPWORDS:

                words.append(w.lower())

    

    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(words))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()



gen_word_cloud('question_title')
gen_word_cloud('question_body')
gen_word_cloud('answer')
import re

question_related_target_cols = [ col for col in target_cols if re.search('^question_', col)]

answer_related_target_cols = [ col for col in target_cols if re.search('^answer_', col)]
train_df[answer_related_target_cols[:5]].plot(kind='hist', figsize=(12, 6), alpha=0.5)
train_df[question_related_target_cols[:5]].plot(kind='hist', figsize=(12, 6), alpha=0.5)
plt.figure(figsize=(12, 12))

sns.heatmap(data=train_df[answer_related_target_cols].corr(), 

            square=True, 

            annot=True,

            linewidths=1, 

            cmap=sns.color_palette("Blues"))
plt.figure(figsize=(12, 12))

sns.heatmap(data=train_df[question_related_target_cols].corr(), 

            square=True, 

            linewidths=1, 

            cmap=sns.color_palette("Blues"))
std = train_df.groupby('category')[question_related_target_cols[:8]].std()

train_df.groupby('category')[question_related_target_cols[:8]].mean().plot(kind='bar', figsize=(16, 8), 

                                                                           yerr=std)
std = train_df.groupby('category')[answer_related_target_cols].std()

train_df.groupby('category')[answer_related_target_cols].mean().plot(kind='bar', figsize=(16, 8), 

                                                                           yerr=std)
frequent_hosts = set(train_df.groupby('host').count()['qa_id'].sort_values(ascending=False)[:10].index)

idx = train_df['host'].map(lambda x: x in frequent_hosts)

train_subset = train_df[idx]



std = train_subset.groupby('host')[answer_related_target_cols].std()

train_subset.groupby('host')[answer_related_target_cols].mean().plot(kind='bar', figsize=(16, 8), yerr=std)
std = train_subset.groupby('host')[question_related_target_cols[:8]].std()

train_subset.groupby('host')[question_related_target_cols[:8]].mean().plot(kind='bar', figsize=(16, 8), yerr=std)
plt.figure(figsize=(8, 6))

text_len_cols = ['question_title_len', 'question_body_len', 'answer_len']

corr_with_text_len = train_df[answer_related_target_cols + text_len_cols].corr().loc[text_len_cols, answer_related_target_cols]

sns.heatmap(data=corr_with_text_len.T, 

            square=True, 

            linewidths=1, 

            annot=True,

            cmap=sns.color_palette("Blues"))
plt.figure(figsize=(12, 12))

corr_with_text_len = train_df[question_related_target_cols + text_len_cols].corr().loc[text_len_cols, question_related_target_cols]

sns.heatmap(data=corr_with_text_len.T, 

            square=True, 

            linewidths=1, 

            annot=True,

            cmap=sns.color_palette("Blues"))
# load w2v model, this might take a few moments, grab a coffee and relax

w2v_model = gensim.models.KeyedVectors.load_word2vec_format('../input/word2vec-google/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore')
TFIDF_SVD_WORDVEC_DIM = 300



def get_text_feats(df, col):



    def tokenize_downcase_filtering(x):

        words = TOKENIZER.tokenize(x)

        lower_case = map(lambda w: w.lower(), words)

        content_words = filter(lambda w: w not in STOPWORDS, lower_case)

        return ' '.join(content_words)



    rows = df[col].map(tokenize_downcase_filtering).values.tolist()

    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))  # dont use sklearn default tokenization tool 

    tfidf_weights = tfidf.fit_transform(rows)

    svd = TruncatedSVD(n_components=TFIDF_SVD_WORDVEC_DIM, n_iter=10)  # reduce dimensionality

    dense_tfidf_repr_mat = svd.fit_transform(tfidf_weights)

    

    word2vec_repr_mat = np.zeros((len(df), w2v_model.vector_size))

    for i, row in enumerate(rows):

        word2vec_accum = np.zeros((w2v_model.vector_size, ))

        word_cnt = 0

        for w in row.split(' '):

            if w in w2v_model.wv:

                word2vec_accum += w2v_model.wv[w]

                word_cnt += 1



        # compute the average for the wordvec of each non-sptop word

        if word_cnt != 0:

            word2vec_repr_mat[i] = word2vec_accum / word_cnt



    return  np.concatenate([word2vec_repr_mat, dense_tfidf_repr_mat], axis=1)  # word2vec + tfidf





def one_hot_feats(df, col):

    return pd.get_dummies(df['host'], prefix='host', drop_first=True).values





# let's build features

df_all = pd.concat((train_df, test_df))

df_all['question_title_len'] = df_all['question_title'].map(lambda x: len(TOKENIZER.tokenize(x)))

df_all['question_body_len'] = df_all['question_body'].map(lambda x: len(TOKENIZER.tokenize(x)))

df_all['answer_len'] = df_all['answer'].map(lambda x: len(TOKENIZER.tokenize(x)))



data = []

for col in text_cols:

    data.append(get_text_feats(df_all, col))



for col in ['category', 'host']:

    data.append(one_hot_feats(df_all, col))



data.append(df_all[text_len_cols].values)

data = np.concatenate(data, axis=1)



train_feats = data[:len(train_df)]

test_feats = data[len(train_df):]



# del w2v_model

# gc.collect()

print(train_feats.shape)
# code from https://www.kaggle.com/ryches/tfidf-benchmark

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.model_selection import KFold

from keras.callbacks.callbacks import EarlyStopping

from scipy.stats import spearmanr



num_folds = 5

fold_scores = []

kf = KFold(n_splits=num_folds, shuffle=True, random_state=9102)



test_preds = np.zeros((len(test_feats), len(target_cols)))

for train_index, val_index in kf.split(train_feats):

    train_X = train_feats[train_index, :]

    train_y = train_df[target_cols].iloc[train_index]

    

    val_X = train_feats[val_index, :]

    val_y = train_df[target_cols].iloc[val_index]

    

    model = Sequential([

        Dense(512, input_shape=(train_feats.shape[1],)),

        Activation('tanh'),

        Dense(256),

        Activation('tanh'),

        Dense(len(target_cols)),

        Activation('sigmoid'),

    ])

    

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

    model.compile(optimizer='adam',

                  loss='binary_crossentropy')

    

    model.fit(train_X, train_y, epochs=50, validation_data=(val_X, val_y), callbacks = [es])

    preds = model.predict(val_X)

    overall_score = 0

    print('-'* 10)

    for i, col in enumerate(target_cols):

        overall_score += spearmanr(preds[:, i], val_y[col].values).correlation / len(target_cols)

        print('%s\t%.5f' % (col, spearmanr(preds[:, i], val_y[col].values).correlation))



    fold_scores.append(overall_score)

    test_preds += model.predict(test_feats) / num_folds

    

print(fold_scores)
sub = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

for i, col in enumerate(target_cols):

    sub[col] = test_preds[:, i]

sub.to_csv("submission.csv", index = False)
sub