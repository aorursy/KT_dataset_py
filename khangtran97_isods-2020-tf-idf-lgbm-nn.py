import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Tensorflow

import tensorflow as tf



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Concatenate

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint





# Sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



import lightgbm as lgb



# import re 

# from unicodedata import normalize 



import string

import os
train_df = pd.read_csv('/kaggle/input/isods2020/training_data_sentiment/training_data_sentiment.csv')

test_df = pd.read_csv('/kaggle/input/isods2020/testing_data_sentiment/testing_data_sentiment.csv')
train_df.head()
test_df.head()
train_df['is_unsatisfied'].value_counts()
train_df['is_unsatisfied'].hist()
train_df.shape
test_df.shape
len(train_df.iloc[0,0].split('|||'))
len(train_df.iloc[0,1].split('|||'))
def clean_line(lines):

    table = str.maketrans('', '', string.punctuation)



    lines = lines.split('|||')

    tmp = []

    for line in lines:

        line = line.split()

        line = [word.lower() for word in line]

        line = [word.translate(table) for word in line]

        line = [word for word in line if word.isalpha()]

        line = ' '.join(line)

        tmp.append(line)

    

    return '|||'.join(tmp)



def len_mess(messages):

    _n = []

    max_len = []

    min_len = []

    mean_len = []

    for mess in messages:

        mess = mess.split('|||')

        _len = [len(q.split()) for q in mess]

        _n.append(len(mess))

        max_len.append(max(_len))

        min_len.append(min(_len))

        mean_len.append(sum(_len) / len(_len))

    

    return _n, max_len, min_len, mean_len
train_df['Where'] = 'train'

test_df['Where'] = 'test'



data = train_df.append(test_df)

data
%%time

feature_numeric = []



data['clean_ques'] = [clean_line(line) for line in data.question]

data['concat_ques'] = [line.replace('|||', ' ') for line in data.clean_ques]

data['len_ques'] = [len(ques.split('|||')) for ques in data.clean_ques]

data['max_word_ques'] = [max([len(q.split()) for q in ques.split('|||')]) for ques in data.clean_ques]

data['min_word_ques'] = [min([len(q.split()) for q in ques.split('|||')]) for ques in data.clean_ques]

data['mean_word_ques'] = [sum([len(q.split()) for q in ques.split('|||')])/len([len(q.split()) for q in ques.split('|||')]) 

                            for ques in data.clean_ques]



data['clean_answ'] = [clean_line(line) for line in data.answer]

data['concat_answ'] = [line.replace('|||', ' ') for line in data.clean_answ]

data['len_answ'] = [len(answ.split('|||')) for answ in data.clean_answ]

data['max_word_answ'] = [max([len(a.split()) for a in answ.split('|||')]) for answ in data.clean_answ]

data['min_word_answ'] = [min([len(a.split()) for a in answ.split('|||')]) for answ in data.clean_answ]

data['mean_word_answ'] = [sum([len(a.split()) for a in answ.split('|||')])/len([len(a.split()) for a in answ.split('|||')]) 

                            for answ in data.clean_answ]



data['label'] = data['is_unsatisfied'].apply(lambda x: 1 if x == 'Y' else 0)



feature_numeric += ['len_ques', 'max_word_ques', 'min_word_ques', 'mean_word_ques', 

                    'len_answ', 'max_word_answ', 'min_word_answ', 'mean_word_answ']



cols = ['ques', 'answ']

n_tfidf = 600

for idx, col in enumerate(cols):

    print(f'TF-IDF {col}...')

    tfidf = TfidfVectorizer(analyzer='word',

                            token_pattern=r'\w{1,}',

                            ngram_range=(1, 3),

                            max_features= n_tfidf,

                            norm='l1',

                            sublinear_tf=True

                           )

    tfidf.fit(data['concat_' + col])

    

    data[[col + '_' + str(i) for i in range(n_tfidf)]] = tfidf.transform(data['concat_' + col]).toarray()

    feature_numeric += [col + '_' + str(i) for i in range(n_tfidf)]

    

    del tfidf

    

data.head()
train_df = data[data.Where == 'train']

test_df = data[data.Where == 'test']

del data



print(train_df.shape, test_df.shape)

train_df.head()
N_FOLD = 5

X = train_df[feature_numeric].values

y = train_df['label'].values



X.shape, y.shape
hyper_params = {

    'task': 'train',

    'objective': 'binary',

    'boosting_type': 'gbdt',

    'learning_rate': 0.01,

    "num_leaves": 250,

    "max_depth": 40,

    'feature_fraction': 0.85,

    "max_bin": 512,

    'metric': ['auc'],

    

    'subsample': 0.8,

    'subsample_freq': 2,

    'verbose': 0,

}





skf = StratifiedKFold(n_splits = N_FOLD, shuffle=True, random_state=13)



y_pred = np.zeros([y.shape[0]], dtype=float)

pred_sub = np.zeros([test_df.shape[0]], dtype=float)



for idx, (idx_train, idx_valid) in enumerate(skf.split(X, y)):

    print('Fold', idx, '...')

    

    gbm = lgb.LGBMModel(**hyper_params)

    gbm.fit(X[idx_train], y[idx_train],

            eval_set=[(X[idx_valid], y[idx_valid])],

            eval_metric='auc',

            verbose = 100,

            early_stopping_rounds=300)

    

    train_pred = gbm.predict(X[idx_train])

    valid_pred = gbm.predict(X[idx_valid])

    print('\tTrain:', metrics.accuracy_score(y[idx_train], train_pred.round()), metrics.f1_score(y[idx_train], train_pred.round()))

    print('\tValid:', metrics.accuracy_score(y[idx_valid], valid_pred.round()), metrics.f1_score(y[idx_valid], valid_pred.round()))

    

    y_pred[idx_valid] = valid_pred

    pred_sub += gbm.predict(test_df[feature_numeric].values) / N_FOLD
plt.figure(figsize=(10,10))

plt.plot(y,y)



# tmp = gbm.predict(X)

# fpr, tpr, thresholds = metrics.roc_curve(y, tmp, pos_label=1)

# plt.plot(fpr, tpr)



fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)

plt.plot(fpr, tpr)



metrics.roc_auc_score(y, y_pred)