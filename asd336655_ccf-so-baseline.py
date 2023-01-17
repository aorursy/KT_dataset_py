import pandas as pd

import numpy as np

import os

import re
df_train_x = pd.read_csv("/kaggle/input/ccf2019news-so/Train_DataSet.csv")

df_train_y = pd.read_csv("/kaggle/input/ccf2019news-so/Train_DataSet_Label.csv")

df_test_x = pd.read_csv("/kaggle/input/ccf2019news-so/Test_DataSet.csv")

# train_x: id, title, content

# train_y: id, label

print(df_train_x.isnull().sum())

df_train_x.dropna(axis = 0, subset=['title'], inplace=True)

df_train_x.fillna(method='ffill', axis=1, inplace=True)

df_test_x.fillna(method='ffill', axis=1, inplace=True)



# dataframe: id, title, content, label, documnet

df_train = pd.merge(df_train_x, df_train_y, on = 'id')

df_train['document'] = df_train['title'] + df_train['content']

df_test_x['document'] = df_test_x['title'] + df_test_x['content']

print('train shape:', df_train.shape)

print('test shape:', df_test_x.shape)
# 空格

re1 = re.compile(r'\s')

# 代码

re2 = re.compile(r'[a-zA-Z0-9\'=\\/:\/?;()<>"\._-]{5,}')

# 数字大于4个

re3 = re.compile(r'[\b]{5,}')

# 聚合标点

re4 = re.compile(r'[\'=\\/:\/?;()<>"\._-]{2,}')



def re_process(text):

    res1 = re1.sub('', text)

    res2 = re2.sub('', res1)

    res3 = re3.sub('', res2)

    res4 = re4.sub('', res3)

    return res4

    

# 正则表达式 清洗

df_train['document'] = df_train['document'].apply(re_process)

df_test_x['document'] = df_test_x['document'].apply(re_process)



from keras.utils import to_categorical

df_train['label'] = df_train['label'].apply(lambda x:to_categorical(x, 3))

print(df_train.loc[0]['label'])


#! -*- coding:utf-8 -*-

import re, os, json, codecs, gc

import numpy as np

import pandas as pd

from random import choice

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

! pip install keras-bert

from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import *

from keras.callbacks import *

from keras.models import Model

import keras.backend as K

from keras.optimizers import Adam
maxlen = 256

config_path = '/kaggle/input/bertch-l12-h768-a12/chinese_L-12_H-768_A-12/bert_config.json'

checkpoint_path = '/kaggle/input/bertch-l12-h768-a12/chinese_L-12_H-768_A-12/bert_model.ckpt'

dict_path = '/kaggle/input/bertch-l12-h768-a12/chinese_L-12_H-768_A-12/vocab.txt'



token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:

    for line in reader:

        token = line.strip()

        token_dict[token] = len(token_dict)



tokenizer = Tokenizer(token_dict)



from keras.metrics import top_k_categorical_accuracy

def acc_top2(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=2)

                    

def build_bert(nclass):

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)



    for l in bert_model.layers:

        l.trainable = True



    x1_in = Input(shape=(None,))

    x2_in = Input(shape=(None,))



    x = bert_model([x1_in, x2_in])

    x = Lambda(lambda x: x[:, 0])(x)

    p = Dense(nclass, activation='softmax')(x)



    model = Model([x1_in, x2_in], p)

    model.compile(loss='categorical_crossentropy', 

                  optimizer=Adam(1e-5),

                  metrics=['accuracy', acc_top2])

    return model
#%%

# data generator

max_length = 256

encode = lambda x: tokenizer.encode(x)[0]

df_train[['title','content','document']] = df_train[['title','content','document']].applymap(encode)

df_test_x[['title','content','document']] = df_test_x[['title','content','document']].applymap(encode)



def padding(df, max_length, sl=0):

    

    def padding_0(row):

        x = row.document

        if len(x)<max_length:

            x = x+[0]*(max_length-len(x))

            row.document = x

        return row

        

    def padding_repeat_itself(row):

        x = row.document

        if len(x)<max_length:

            sub_length = max_length - len(x)

            fold = sub_length // len(x) 

            x += x*fold

            x += x[:max_length-len(x)]

            row.document = x

        return row

        

    def padding_repeat_title(row):

        x = row.document

        if len(x)<max_length:

            sub_length = max_length - len(x)

            fold = sub_length // len(x) 

            x += row.title * fold

            x += row.title[:max_length-len(x)]

            row.document = x

            return row

        

        

    df['document'] = df['document'].apply(lambda x: x[:max_length])

    if sl == 0:

        df.apply(padding_0, axis=1)

        #map(padding_0, df)

    if sl == 1:

        df.apply(padding_repeat_itself, axis=1)

    if sl == 2:

        df.apply(padding_repeat_title, axis=1)

    return df

        

df_train = padding(df_train,max_length, sl=0)

df_test_x = padding(df_test_x,max_length, sl=0)
! mkdir bert_dump

def run_cv(nfold, data, data_test):

    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)

    train_model_pred = np.zeros((len(data), 3))

    test_model_pred = np.zeros((len(data_test), 3))

    data['set'] = np.ones(len(data))

    X_test = data_test

    X_test['set'] = np.ones(len(data_test))

    X_test.loc[:]['set'] = np.zeros([len(X_test), max_length])

    test_x = [np.array(X_test['document'].tolist()), np.array(X_test['set'].tolist())]

    

    for i, (train_fold, test_fold) in enumerate(kf):

        X_train, X_valid = data.iloc[train_fold], data.iloc[test_fold]

    

        model = build_bert(3)

        if i==0:

            print(model.summary())

        early_stopping = EarlyStopping(monitor='val_acc', patience=3)

        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)

        checkpoint = ModelCheckpoint('bert_dump/' + str(i) + '.hdf5', monitor='val_acc', 

                                         verbose=2, save_best_only=True, mode='max',save_weights_only=True)

        """

        train_D = data_generator(X_train, shuffle=True)

        valid_D = data_generator(X_valid, shuffle=True)

        test_D = data_generator(data_test, shuffle=False)

        model.fit_generator(

            train_D.__iter__(),

            steps_per_epoch=len(train_D),

            epochs=5,

            validation_data=valid_D.__iter__(),

            validation_steps=len(valid_D),

            callbacks=[early_stopping, plateau, checkpoint])

        # return model

        train_model_pred[test_fold, :] =  model.predict_generator(valid_D.__iter__(), steps=len(valid_D),verbose=1)

        test_model_pred += model.predict_generator(test_D.__iter__(), steps=len(test_D),verbose=1)

        """

        X_train.iloc[:]['set'] = np.zeros([len(X_train), max_length])

        X_valid.iloc[:]['set'] = np.zeros([len(X_valid), max_length])

        

        train_y = np.array(X_train['label'].tolist())

        train_x = [np.array(X_train['document'].tolist()), np.array(X_train['set'].tolist())]

        valid_y = np.array(X_valid['label'].tolist())

        valid_x = [np.array(X_valid['document'].tolist()), np.array(X_valid['set'].tolist())]

        model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=5, batch_size=8, callbacks=[early_stopping, plateau, checkpoint],verbose=2)

        

        test_model_pred += model.predict(test_x)

        del model; gc.collect()

        K.clear_session()

    return test_model_pred
Y_pre = run_cv(10, df_train, df_test_x)
test_pred = [np.argmax(x) for x in Y_pre]

df_test_x['label'] = test_pred

df_test_x[['id', 'label']].to_csv('baseline.csv', index=None)