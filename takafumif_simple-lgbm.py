import pandas as pd

import numpy as np

from collections import defaultdict

import lightgbm as lgb

from scipy.sparse import vstack, hstack, csr_matrix, spmatrix

from scipy.stats import binom

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer as CV

import datetime

import gc

import re
#文字列中の記号を除去する関数

#The function to remove noisy marks in text

def arrange_words(text):

    text = text.replace('!', '')

    text = text.replace('?', '')

    text = text.replace(',', '')

    text = text.replace('.', '')

    text = text.replace('“', '')

    text = text.replace('”', '')

    text = text.replace('‘', '')

    text = text.replace('’', '')

    text = text.replace('•', '')

    text = text.replace('・', '')

    text = text.replace('…', '')

    text = text.replace(':', '')

    text = text.replace(';', '')

    text = text.replace('(', '')

    text = text.replace(')', '')

    text = text.replace('{', '')

    text = text.replace('}', '')

    text = text.replace('[', '')

    text = text.replace(']', '')

    text = text.replace('<', '')

    text = text.replace('>', '')

    text = text.replace('\'', '')

    text = text.replace('\/', '')

    text = text.replace('"', '')

    text = text.replace('-', ' ')

    text = text.replace('_', ' ')

    text = text.replace('\n', ' ')

    text = text.replace('\r', ' ')

    text = text.replace('#', '')

    text = re.sub(r'[0-9]+', "0", text)

    text = ' ' + text + ' '

    return text
#Series全体に含まれる全ての単語とその個数を集計する関数

#The function to count all words and the number of those through overall texts

def get_word_counts(texts):

    word_counts = defaultdict(int)

    for text in texts.values:

        for word in text.split(' '):

            word_counts[word.lower()] += 1

    return word_counts
#二項分布を用いて有効な単語を抽出する関数

#The function to select useful words based on probability of binomial distribution

def extract_useful_column(dataset, all_words, significance_level):

    texts = dataset.copy()

    useful_words = []

    p = texts['target'].sum() / texts['comment_text_arranged'].count()

    for word in all_words:

        texts['comment_text_arranged'] = dataset['comment_text_arranged'].map(lambda x: 1 if ' ' + word + ' ' in x else 0)

        k = texts[texts['comment_text_arranged']==1]['target'].sum()

        N = texts[texts['comment_text_arranged']==1]['comment_text_arranged'].count()

#        print(binom.cdf(k, N, p))

#        if (binom.cdf(k, N, p)<(significance_level/2)) or (binom.cdf(k, N, p)>(1-significance_level/2)):

        p_value = binom.cdf(k, N, p)

        if (p_value<(significance_level/2)) or (p_value>(1-significance_level/2)):

            print(word)

            useful_words.append(word)

    return useful_words
#データ型を定義

#Define data types

dtypes = {

        'id':                                             'category',

        'target':                                       'float16', 

        'comment_text':                           'category', 

        'severe_toxicity':                           'float16', 

        'obscene':                                    'float16', 

        'identity_attack':                           'float16', 

        'insult':                                         'float16', 

        'threat':                                        'float16', 

        'asian':                                         'float16', 

        'atheist':                                       'float16', 

        'bisexual':                                     'float16', 

        'black':                                         'float16', 

        'buddhist':                                    'float16', 

        'christian':                                    'float16', 

        'female':                                       'float16', 

        'heterosexual':                              'float16', 

        'hindu':                                         'float16', 

        'homosexual_gay_or_lesbian':        'float16', 

        'intellectual_or_learning_disability': 'float16', 

        'jewish':                                        'float16', 

        'latino':                                         'float16', 

        'male':                                          'float16', 

        'muslim':                                       'float16', 

        'other_disability':                           'float16', 

        'other_gender':                             'float16', 

        'other_race_or_ethnicity':              'float16', 

        'other_religion':                             'float16', 

        'other_sexual_orientation':             'float16', 

        'physical_disability':                       'float16', 

        'psychiatric_or_mental_illness':       'float16', 

        'transgender':                                'float16', 

        'white':                                          'float16', 

        'created_date':                              'category', 

        'publication_id':                             'category', 

        'parent_id':                                    'category', 

        'article_id':                                     'category', 

        'rating':                                         'category', 

        'funny':                                         'int8', 

        'wow':                                           'int8', 

        'sad':                                             'int8', 

        'likes':                                            'int8', 

        'disagree':                                     'int8', 

        'sexual_explicit':                             'float16', 

        'identity_annotator_count':             'int8', 

        'toxicity_annotator_count':             'int8', 

        }
#訓練データ・テストデータをロード

#Load DataSet

train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', dtype=dtypes)

test  = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv',  dtype=dtypes)

train_ids = train.index

test_ids  = test.index

train_y = train['target'].apply(lambda x: 1 if x>=0.5 else 0)

train_X = train.drop('target', axis=1)

test_X = test

gc.collect()
#comment_textで使われている全ての単語の個数を取得

#get the all words used in the column 'comment_text'

train_X['comment_text_arranged'] = train_X['comment_text'].map(arrange_words)

test_X['comment_text_arranged'] = test_X['comment_text'].map(arrange_words)

train_word_counts = get_word_counts(train_X['comment_text_arranged'])

test_word_counts = get_word_counts(test_X['comment_text_arranged'])

train_word_counts_df = pd.DataFrame(list(train_word_counts.items()), columns=['word', 'count'])

test_word_counts_df = pd.DataFrame(list(test_word_counts.items()), columns=['word', 'count'])
#訓練データとテストデータ双方に存在する単語のみを抽出

#extract just the words in both train data and test data

word_counts_df = train_word_counts_df.merge(test_word_counts_df, on='word', how='inner')
#訓練データで極端に個数が少ない、又は訓練データ・テストデータで個数が極端に偏っている単語以外を抽出

#drop the words extremely few or biased between train data and test data

word_counts_df['scaled_total'] = word_counts_df['count_x'] + word_counts_df['count_y'] * 18

word_counts_df = word_counts_df[word_counts_df['count_x']>100]

word_counts_df = word_counts_df[(word_counts_df['count_x']/word_counts_df['scaled_total']>0.2) & (word_counts_df['count_x']/word_counts_df['scaled_total']<0.8)]
#それぞれの単語を特徴量とするデータセットを作成

#create dataset which has each words as features;



cv = CV(vocabulary=word_counts_df['word'].tolist())

print(datetime.datetime.now())

train_X_flattened = cv.fit_transform(list(train_X['comment_text_arranged'].values))

test_X_flattened = cv.fit_transform(list(test_X['comment_text_arranged'].values))

print(datetime.datetime.now())
#LightGBMで訓練し、予測値を作成。バリデーションにはStratifiedKFoldを用いる。

#train and predict using LightGBM. Validate using StratifiedKFold.



lgb_test_result  = np.zeros(test_ids.shape[0])

m = 100000

counter = 0



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

skf.get_n_splits(train_ids, np.array(train_y))



for train_index, test_index in skf.split(train_ids, train_y):

    

    print('Fold {}\n'.format(counter + 1))

    X_fit = train_X_flattened[train_index]

    X_val = train_X_flattened[test_index]

    X_fit, X_val = csr_matrix(X_fit, dtype='float32'), csr_matrix(X_val, dtype='float32')

    y_fit, y_val = train_y[train_index], train_y[test_index]

    

    gc.collect()



    lgb_model = lgb.LGBMClassifier(max_depth=-1,

                                   n_estimators=1000,

                                   learning_rate=0.05,

                                   num_leaves=2**9-1,

                                   colsample_bytree=0.28,

                                   objective='binary', 

                                   n_jobs=-1)

                                   

    

                               

    lgb_model.fit(X_fit, y_fit, eval_metric='auc', 

                  eval_set=[(X_val, y_val)], 

                  verbose=100, early_stopping_rounds=100)

                  

    

    del X_fit, X_val, y_fit, y_val, train_index, test_index

    gc.collect()

    

    test = csr_matrix(test_X_flattened, dtype='float32')

    lgb_test_result += lgb_model.predict_proba(test)[:,1]

    counter += 1

    

    del test

    gc.collect()

    



submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

submission['prediction'] = lgb_test_result / counter

submission.to_csv('./submission.csv', index=False)