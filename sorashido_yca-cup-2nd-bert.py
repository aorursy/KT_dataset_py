# official tokenizer

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import plot_model

import tensorflow_hub as hub

from keras.utils import np_utils

import matplotlib.pyplot as plt

import tokenization

import seaborn as sns
train = pd.read_csv("../input/ykc-2nd/train.csv")

test = pd.read_csv("../input/ykc-2nd/test.csv")

sub = pd.read_csv("../input/ykc-2nd/sample_submission.csv")

train.shape, test.shape, sub.shape
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
from keras import backend as K



def recall_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall



def precision_m(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub

def build_model(bert_layer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:, 0, :]

    out = Dense(21, activation='softmax')(clf_output)

    

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])

    

    return model
%%time

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
# train_input = bert_encode(train["product_name"].head(10).values, tokenizer, max_len=160)

# test_input = bert_encode(test["product_name"].head(10).values, tokenizer, max_len=160)

# train_labels = np_utils.to_categorical(train["department_id"].head(10))



train_input = bert_encode(train["product_name"].values, tokenizer, max_len=120)

test_input = bert_encode(test["product_name"].values, tokenizer, max_len=120)

train_labels = np_utils.to_categorical(train["department_id"])
model = build_model(bert_layer, max_len=120)

model.summary()
plot_model(

    model,

    show_shapes=True,

)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)



train_history = model.fit(

    train_input, train_labels,

    validation_split=0.1,

    epochs=5,

    callbacks=[checkpoint],

    batch_size=16

)
loss = train_history.history['loss']

val_loss = train_history.history['val_loss']



nb_epoch = len(loss)

plt.plot(range(nb_epoch), loss,     marker='.', label='loss')

plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')

plt.legend(loc='best', fontsize=10)

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
f1_m = train_history.history['f1_m']

val_f1_m = train_history.history['val_f1_m']



nb_epoch = len(loss)

plt.plot(range(nb_epoch), f1_m,     marker='.', label='f1_m')

plt.plot(range(nb_epoch), val_f1_m, marker='.', label='val_f1_m')

plt.legend(loc='best', fontsize=10)

plt.grid()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.show()
model.load_weights('model.h5')

test_pred = np.argmax(model.predict(test_input), axis=1)
sns.countplot(test_pred, color='blue') # 予測
sns.countplot(train["department_id"], color='red') # target
sub["department_id"] = test_pred

sub.to_csv('sub.csv', index=False)