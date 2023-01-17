!pip install transformers

import transformers
import os

import gc

import pandas as pd

import numpy as np

import pickle

import random

import re

import time

import warnings

import string



import tensorflow_hub as hub

from tqdm.notebook import tqdm

import tensorflow as tf

from keras import backend as K

import tensorflow.keras.layers as layers

from tensorflow.keras import callbacks

from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalMaxPooling1D, Dropout, BatchNormalization, Average

from tensorflow.keras.activations import sigmoid

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import AUC

from tensorflow.keras.regularizers import l1, l2

from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler



from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score

import lightgbm as lgb

import xgboost as xgb

from sklearn.svm import SVC



pd.set_option('max_colwidth', 500)

pd.set_option('display.float_format', lambda x: '%.4f' % x)





tf.random.set_seed(42)

random.seed(42)



warnings.filterwarnings("ignore")
import pandas as pd

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')



ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train.loc[train['id'].isin(ids_with_target_error),'target'] = 0
%%time



from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)





def greed_encode(data, max_len) :

    input_ids = []

    attention_masks = []

  

    for i in range(len(data.text)):

        

        encoded = tokenizer.encode_plus(data.text[i], add_special_tokens=True, max_length=max_len, pad_to_max_length=True)

         

        tok_len = sum(encoded['attention_mask'])

        if tok_len > max_len*.8:

            all_encode = tokenizer.encode_plus(data.text[i], add_special_tokens=True)

            all_ids = all_encode['input_ids']

            all_attention = all_encode['attention_mask']  

            max_len_half = int(max_len/2)

            input_ids.append(all_ids[:max_len_half] + all_ids[-max_len_half:])

            attention_masks.append(all_attention[:max_len_half] + all_attention[-max_len_half:])

            

        else:  

            input_ids.append(encoded['input_ids'])

            attention_masks.append(encoded['attention_mask'])

    

    return np.array(input_ids),np.array(attention_masks)





train_input_ids,train_attention_masks = greed_encode(train,50)

test_input_ids,test_attention_masks = greed_encode(test,50)

y_train = train.target
N_SAMPLES = 8

def create_model(bert_model, MAX_LEN=50):

    input_ids = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')

    attention_mask = layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')



    last_hidden_state, _ = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})

    last_hidden_state = Dropout(0.1)(last_hidden_state)

    x_avg = layers.GlobalAveragePooling1D()(last_hidden_state)

    x_max = layers.GlobalMaxPooling1D()(last_hidden_state)

    x = layers.Concatenate()([x_avg, x_max])

    

    samples = []    

    for n in range(N_SAMPLES):

        sample_mask = layers.Dense(64, activation='relu', name = f'dense_{n}')

        sample = layers.Dropout(.5)(x)

        sample = sample_mask(sample)

        sample = layers.Dense(1, activation='sigmoid', name=f'sample_{n}')(sample)

        samples.append(sample)

    

    output = layers.Average(name='output')(samples)

    

    model = Model(inputs=[input_ids, attention_mask], outputs=output)

    model.compile(Adam(lr=1e-5), loss = BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

    #model.compile(tfa.optimizers.RectifiedAdam(learning_rate=1e-5,min_lr=6e-6,total_steps=2000), loss = BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

    return model
from transformers import TFBertModel

bert_model = TFBertModel.from_pretrained('bert-large-uncased')



model = create_model(bert_model)

model.summary()
def lgb_svc_cv(model):

    cls_layer_model = Model(model.input, outputs=model.get_layer(f'dense_1').output)

    X_train = cls_layer_model.predict([train_input_ids,train_attention_masks])

    X_test = cls_layer_model.predict([test_input_ids,test_attention_masks])

    y_train = train.target.values

    

    N_FOLDS = 5

    print(f'LGBM')

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof = np.zeros(len(X_train))

    sub = np.zeros(len(X_test))

    params = {'boosting_type': 'dart'}

    for fold_, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

        X_train_cv, y_train_cv = pd.DataFrame(X_train).loc[train_idx], pd.DataFrame(y_train).loc[train_idx]

        X_val, y_val = pd.DataFrame(X_train).loc[val_idx], pd.DataFrame(y_train).loc[val_idx]

        train_data = lgb.Dataset(X_train_cv, label=y_train_cv)

        val_data = lgb.Dataset(X_val, label=y_val)

        watchlist = [train_data, val_data]

        clf = lgb.train(params, train_set = train_data, valid_sets=watchlist)

        oof[val_idx] = clf.predict(X_val)

        sub += clf.predict(X_test)/folds.n_splits

        

    sub_all_lgb = sub

    print(accuracy_score(y_train, np.round(oof).astype(int)),'\n')        

        



    print(f'SVC')

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof = np.zeros(len(X_train))

    sub = np.zeros(len(X_test))

    scores = [0 for _ in range(folds.n_splits)]

    for fold_, (train_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

        X_train_cv, y_train_cv = pd.DataFrame(X_train).loc[train_idx], pd.DataFrame(y_train).loc[train_idx]

        X_val, y_val = pd.DataFrame(X_train).loc[val_idx], pd.DataFrame(y_train).loc[val_idx]

        clf = SVC(kernel='rbf', C=1.75, gamma = 0.1, probability = True).fit(X_train_cv, y_train_cv)

        oof[val_idx] = clf.predict_proba(X_val)[:,1]

        sub += clf.predict_proba(X_test)[:,1]/folds.n_splits



    sub_all_svc = sub    

    print(accuracy_score(y_train, np.round(oof).astype(int)),'\n')

    return sub_all_lgb, sub_all_svc
oof_preds = np.zeros(train_input_ids.shape[0])

test_preds = np.zeros(test.shape[0])

all_preds = pd.DataFrame()



fold_hist = {}

n_splits = 2

folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)





for i, (trn_idx, val_idx) in enumerate(folds.split(train_input_ids)):

    modelstart = time.time()

    bert_model = TFBertModel.from_pretrained('bert-large-uncased')

    model = create_model(bert_model)

    

    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1,

                                 mode='min', baseline=None, restore_best_weights=True)

    def lr_sc(epoch):

        return 1.5e-5/(epoch + 1)

    scheduler = LearningRateScheduler(lr_sc)

    

    

    history = model.fit(

        x=[train_input_ids[trn_idx], train_attention_masks[trn_idx]], y=y_train[trn_idx],

        validation_data=([train_input_ids[val_idx], train_attention_masks[val_idx]], y_train[val_idx]),

        epochs=3,

        batch_size=16,

        callbacks=[scheduler, es]

    )



    best_index = np.argmin(history.history['val_loss'])

    fold_hist[i] = history

    

    oof_preds[val_idx] = model.predict([train_input_ids[val_idx], train_attention_masks[val_idx]]).ravel()

    all_preds[str(i) + '_fold_NN'] = model.predict([test_input_ids,test_attention_masks]).reshape(-1)

    sc = accuracy_score(y_train[val_idx], (oof_preds[val_idx] > 0.5).astype(int))

    print("\nFOLFD {} in {:.1f} min - Avg Acc NN {:.5f} - Best Epoch {}".format(i, (time.time() - modelstart)/60, sc, best_index + 1),'\n')

    

    

    # grab last layer and use LGBM and SVC

    all_preds[str(i) + '_fold_lgb'], all_preds[str(i) + '_fold_svc'] = lgb_svc_cv(model)

    

    

    del model

    K.clear_session()

    gc.collect()
test_pred = all_preds[['0_fold_NN','1_fold_NN']].mean(axis = 1)*.8 + all_preds[['0_fold_lgb','1_fold_lgb']].mean(axis = 1)*.1 + all_preds[['0_fold_svc','1_fold_svc']].mean(axis = 1)*.1
submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submission['target'] = np.round(test_pred).astype(int)

submission.head(10)
submission.to_csv('submission.csv', index=False)