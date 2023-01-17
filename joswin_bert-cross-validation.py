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
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#version 6

convert_abbrev = False

model_type = 'default'

Dropout_num = 0.1

learning_rate = 1e-5

max_len = 160

layers = [] #not including final layer

activation = 'relu' #for the non-final layers

model_name = 'model_bert_cv_default_{}_shape_{}'.format(Dropout_num,'_'.join([str(i) for i in layers]))
# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input,Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization

from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,log_loss,accuracy_score
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ["[CLS]"] + text + ["[SEP]"]

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    

    if model_type=='default':

        # Without Dropout

        for layer in layers:

            clf_output = Dense(layer, activation=activation)(clf_output)

        out = Dense(1, activation='sigmoid')(clf_output)

    elif model_type=='dropout':

        # With Dropout(Dropout_num), Dropout_num > 0

        for layer in layers:

            x = Dropout(Dropout_num)(clf_output)

            clf_output = Dense(layer, activation=activation)(x)

        x = Dropout(Dropout_num)(clf_output)

        out = Dense(1, activation='sigmoid')(x)

    elif model_type=='GlobalAveragePooling1D':

        for layer in layers:

            if Dropout_num>0:

                clf_output = Dropout(Dropout_num)(clf_output)

            clf_output = Dense(layer, activation=activation)(clf_output)

        x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)

        out = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"#/2 is the updated version. need to try with that

# module_url = "https://tfhub.dev/tensorflow/albert_en_base/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(train.text.values, tokenizer, max_len=max_len)

test_input = bert_encode(test.text.values, tokenizer, max_len=max_len)

train_labels = train.target.values



print(train_input[0].shape,test_input[0].shape)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
train_labels = train.target.values
import gc
infold_pred,outfold_pred = np.zeros(train.shape[0]),np.zeros(train.shape[0])



test_pred = np.zeros(test.shape[0])



for dev_index, val_index in skf.split(train.text.values,train_labels):

    print("TRAIN:", dev_index, "TEST:", val_index)

    dev = train.text.values[dev_index]

    dev_dv = train_labels[dev_index]

    val = train.text.values[val_index]

    val_dv = train_labels[val_index]

    # 

    dev_input = bert_encode(dev, tokenizer, max_len=max_len)

    val_input = bert_encode(val, tokenizer, max_len=max_len)

    test_input = bert_encode(test.text.values, tokenizer, max_len=max_len)

    

    model = build_model(bert_layer, max_len=max_len)

    model.summary()

    checkpoint = ModelCheckpoint('{}.h5'.format(model_name), monitor='val_loss', save_best_only=True)



    dev_history = model.fit(

        dev_input, dev_dv,

        validation_split=0.2,

        epochs=3,

        callbacks=[checkpoint],

        batch_size=16

    )



    model.load_weights('{}.h5'.format(model_name))



    infold_pred[dev_index] += model.predict(dev_input)[:,0]

    outfold_pred[val_index] += model.predict(val_input)[:,0]

    test_pred += model.predict(test_input)[:,0]

    os.remove('{}.h5'.format(model_name))

    gc.collect()
infold_pred = infold_pred/4

test_pred = test_pred/5
import pickle
with open('predictions.pkl','wb') as f:

    pickle.dump({'infold_pred':infold_pred,'outfold_pred':outfold_pred,'test_pred':test_pred},f)
print('Train score:{}, test score:{}'.format(f1_score(train_labels,infold_pred.round().astype(int)),

                                             f1_score(train_labels,outfold_pred.round().astype(int))))
submission['target'] = test_pred.round().astype(int)

submission.to_csv('{}.csv'.format(model_name), index=False)