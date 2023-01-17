# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/encoded_train.csv')
test=pd.read_csv('/kaggle/input/Test_health.csv')
sub = pd.read_csv('/kaggle/input/ss_health.csv')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.
import pandas as pd
import re

# Replace all numeric with 'n'
replace_numbers = re.compile(r'\d+', re.IGNORECASE)

COMMENT_COL = 'text'
ID_COL = 'ID'
input_dir = '../input/'
output_dir = '../input/'

# redundancy words and their right formats
redundancy_rightFormat = {
    'ckckck': 'cock',
    'fuckfuck': 'fuck',
    'lolol': 'lol',
    'lollol': 'lol',
    'pussyfuck':'fuck',
    'gaygay': 'gay',
    'haha': 'ha',
    'sucksuck': 'suck'}

redundancy = set(redundancy_rightFormat.keys())
# all the words below are included in glove dictionary
# combine these toxic indicators with 'CommProcess.revise_triple_and_more_letters'
Depression_indicator_words = ['sad','depression','alone','hopeless','Sadness','hatred','situation','stressed',
    'emotionally', 'psycologically','down','sad','lonely','low', 'bad', 'feeling',
    'challenges','unworthy', 'wasted','Challenging']

Depression_indicator_words_sets = set(Depression_indicator_words)


def _get_DepressionIndicator_transformers():
    DepressionIndicator_transformers = dict()
    for word in Depression_indicator_words:
        tmp_1 = []
        for c in word:
            if len(tmp_1) > 0:
                tmp_2 = []
                for pre in tmp_1:
                    tmp_2.append(pre + c)
                    tmp_2.append(pre + c + c)
                tmp_1 = tmp_2
            else:
                tmp_1.append(c)
                tmp_1.append(c + c)
        DepressionIndicator_transformers[word] = tmp_1
    return DepressionIndicator_transformers


DepressionIndicator_transformers = _get_DepressionIndicator_transformers()

deny_origin = {
    "you're": ['you', 'are'],
    "i'm": ['i', 'am'],
    "he's": ['he', 'is'],
    "she's": ['she', 'is'],
    "it's": ['it', 'is'],
    "they're": ['they', 'are'],
    "can't": ['can', 'not'],
    "couldn't": ['could', 'not'],
    "don't": ['do', 'not'],
    "don;t": ['do', 'not'],
    "didn't": ['did', 'not'],
    "doesn't": ['does', 'not'],
    "isn't": ['is', 'not'],
    "wasn't": ['was', 'not'],
    "aren't": ['are', 'not'],
    "weren't": ['were', 'not'],
    "won't": ['will', 'not'],
    "wouldn't": ['would', 'not'],
    "hasn't": ['has', 'not'],
    "haven't": ['have', 'not'],
    "what's": ['what', 'is'],
    "that's": ['that', 'is'],
}
denies = set(deny_origin.keys())
class CommProcess(object):
    @staticmethod
    def clean_text(t):
        t = re.sub(r"[^A-Za-z0-9,!?*.;’´'\/]", " ", t)
        t = replace_numbers.sub(" ", t)
        t = t.lower()
        t = re.sub(r",", " ", t)
        t = re.sub(r"’", "'", t)
        t = re.sub(r"´", "'", t)
        t = re.sub(r"\.", " ", t)
        t = re.sub(r"!", " ! ", t)
        t = re.sub(r"\?", " ? ", t)
        t = re.sub(r"\/", " ", t)
        return t

    @staticmethod
    def revise_deny(t):
        ret = []
        for word in t.split():
            if word in denies:
                ret.append(deny_origin[word][0])
                ret.append(deny_origin[word][1])
            else:
                ret.append(word)
        ret = ' '.join(ret)
        ret = re.sub("'", " ", ret)
        ret = re.sub(r";", " ", ret)
        return ret

    @staticmethod
    def revise_star(t):
        ret = []
        for word in t.split():
            if ('*' in word) and (re.sub('\*', '', word) in Depression_indicator_words_sets):
                word = re.sub('\*', '', word)
            ret.append(word)
        ret = re.sub('\*', ' ', ' '.join(ret))
        return ret

    @staticmethod
    def revise_triple_and_more_letters(t):
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            reg = letter + "{2,}"
            t = re.sub(reg, letter + letter, t)
        return t

    @staticmethod
    def revise_redundancy_words(t):
        ret = []
        for word in t.split(' '):
            for redu in redundancy:
                if redu in word:
                    word = redundancy_rightFormat[redu]
                    break
            ret.append(word)
        return ' '.join(ret)

    @staticmethod
    def fill_na(t):
        if t.strip() == '':
            return 'NA'
        return t


def execute_comm_process(df):
    comm_process_pipeline = [
        CommProcess.clean_text,
        CommProcess.revise_deny,
        CommProcess.revise_star,
        CommProcess.revise_triple_and_more_letters,
        CommProcess.revise_redundancy_words,
        CommProcess.fill_na,
    ]
    for cp in comm_process_pipeline:
        df[COMMENT_COL] = df[COMMENT_COL].apply(cp)
    return df
# Process whole train data
print('Comm has processed whole train data')
df_train = pd.read_csv('/kaggle/input/encoded_train.csv')
df_train = execute_comm_process(df_train)
df_train.to_csv('train_processed.csv', index=False)

# Process test data
print('Comm has processed whole test data')
df_test = pd.read_csv('/kaggle/input/Test_health.csv')
df_test = execute_comm_process(df_test)
df_test.to_csv('test_processed.csv', index=False)
import numpy as np
np.random.seed(42)
import pandas as pd
import string
import re

import gensim
from collections import Counter
import pickle

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, LSTM,Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.preprocessing import text, sequence

from keras.callbacks import Callback
from keras import optimizers
from keras.layers import Lambda

import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords

import os
os.environ['OMP_NUM_THREADS'] = '4'

import gc
from keras import backend as K
from sklearn.model_selection import KFold

from unidecode import unidecode

import time

eng_stopwords = set(stopwords.words("english"))
# 1. preprocessing
train = pd.read_csv("train_processed.csv")
test = pd.read_csv("test_processed.csv")
#2.  remove non-ascii

special_character_removal = re.compile(r'[^A-Za-z\.\-\?\!\,\#\@\% ]',re.IGNORECASE)
def clean_text(x):
    x_ascii = unidecode(x)
    x_clean = special_character_removal.sub('',x_ascii)
    return x_clean

train['clean_text'] = train['text'].apply(lambda x: clean_text(str(x)))
test['clean_text'] = test['text'].apply(lambda x: clean_text(str(x)))

X_train = train['clean_text'].fillna("something").values
y_train = train[["Depression", "Alcohol", "Suicide", "Drugs"]].values
X_test = test['clean_text'].fillna("something").values

def add_features(df):
    
    df['text'] = df['text'].apply(lambda x:str(x))
    df['total_length'] = df['text'].apply(len)
    df['capitals'] = df['text'].apply(lambda text: sum(1 for c in text if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.text.str.count('\S+')
    df['num_unique_words'] = df['text'].apply(lambda text: len(set(w for w in text.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']  

    return df

train = add_features(train)
test = add_features(test)

features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)

ss = StandardScaler()
ss.fit(np.vstack((features, test_features)))
features = ss.transform(features)
test_features = ss.transform(test_features)
# For best score (Public: 9869, Private: 9865), change to max_features = 283759, maxlen = 900
max_features = 1050
maxlen = 128

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_test_sequence = tokenizer.texts_to_sequences(X_test)

x_train = sequence.pad_sequences(X_train_sequence, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test_sequence, maxlen=maxlen)
print(len(tokenizer.word_index))
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                model.save_weights("best_weights.h5")
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True
def get_model(features,clipvalue=1.,num_filters=40,dropout=0.5,embed_size=501):
    features_input = Input(shape=(features.shape[1],))
    inp = Input(shape=(maxlen, ))
    
    # Layer 1: concatenated fasttext and glove twitter embeddings.
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    
    # Uncomment for best result
    # Layer 2: SpatialDropout1D(0.5)
    #x = SpatialDropout1D(dropout)(x)
    
    # Uncomment for best result
    # Layer 3: Bidirectional CuDNNLSTM
    #x = Bidirectional(LSTM(num_filters, return_sequences=True))(x)


    # Layer 4: Bidirectional CuDNNGRU
    x, x_h, x_c = Bidirectional(GRU(num_filters, return_sequences=True, return_state = True))(x)  
    
    # Layer 5: A concatenation of the last state, maximum pool, average pool and 
    # two features: "Unique words rate" and "Rate of all-caps words"
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    x = concatenate([avg_pool, x_h, max_pool,features_input])
    
    # Layer 6: output dense layer.
    outp = Dense(4, activation="sigmoid")(x)

    model = Model(inputs=[inp,features_input], outputs=outp)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
model = get_model(features)

batch_size = 32

# Used epochs=100 with early exiting for best score.
epochs = 100
gc.collect()
K.clear_session()

# Change to 10
num_folds = 10 #number of folds

predict = np.zeros((test.shape[0],4))

# Uncomment for out-of-fold predictions
#scores = []
#oof_predict = np.zeros((train.shape[0],6))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=239)

for train_index, test_index in kf.split(x_train):
    
    kfold_y_train,kfold_y_test = y_train[train_index], y_train[test_index]
    kfold_X_train = x_train[train_index]
    kfold_X_features = features[train_index]
    kfold_X_valid = x_train[test_index]
    kfold_X_valid_features = features[test_index] 
    
    gc.collect()
    K.clear_session()
    
    model = get_model(features)
    
    ra_val = RocAucEvaluation(validation_data=([kfold_X_valid,kfold_X_valid_features], kfold_y_test), interval = 1)
    
    model.fit([kfold_X_train,kfold_X_features], kfold_y_train, batch_size=batch_size, epochs=epochs, verbose=1,
             callbacks = [ra_val])
    gc.collect()
    
    #model.load_weights(bst_model_path)
    model.load_weights("best_weights.h5")
    
    predict += model.predict([x_test,test_features], batch_size=batch_size,verbose=1) / num_folds
    
    #gc.collect()
    # uncomment for out of fold predictions
    #oof_predict[test_index] = model.predict([kfold_X_valid, kfold_X_valid_features],batch_size=batch_size, verbose=1)
    #cv_score = roc_auc_score(kfold_y_test, oof_predict[test_index])
    
    #scores.append(cv_score)
    #print('score: ',cv_score)

print("Done")
#print('Total CV score is {}'.format(np.mean(scores)))    


sub = pd.read_csv('/kaggle/input/ss_health.csv')
class_names = ['Depression', 'Alcohol', 'Suicide', 'Drugs']
sub[class_names] = predict
sub.to_csv('model_9872_baseline_submission.csv',index=False)

# uncomment for out of fold predictions
#oof = pd.DataFrame.from_dict({'id': train['id']})
#for c in class_names:
#    oof[c] = np.zeros(len(train))
#    
#oof[class_names] = oof_predict
#for c in class_names:
#    oof['prediction_' +c] = oof[c]
#oof.to_csv('oof-model_9872_baseline_submission.csv', index=False)