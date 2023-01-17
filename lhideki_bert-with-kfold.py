# We will use the official tokenization script created by the Google team

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow_hub as hub



import tokenization
def bert_encode(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []

    

    for text in texts:

        text = tokenizer.tokenize(text)

            

        text = text[:max_len-2]

        input_sequence = ['[CLS]'] + text + ['[SEP]']

        pad_len = max_len - len(input_sequence)

        

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0] * pad_len

        pad_masks = [1] * len(input_sequence) + [0] * pad_len

        segment_ids = [0] * max_len

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

    

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
def build_model(bert_layer, max_len=512):

    def inner_build_model():

        input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')

        input_mask = Input(shape=(max_len,), dtype=tf.int32, name='input_mask')

        segment_ids = Input(shape=(max_len,), dtype=tf.int32, name='segment_ids')



        _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

#         clf_output = Bidirectional(LSTM(128))(sequence_output)

        clf_output = sequence_output[:, 0, :]

        out = Dense(1, activation='sigmoid')(clf_output)



        model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy', 'mse'])

    

        return model

    return inner_build_model
%%time

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'

# module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1'



bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train['token_len'] = train['text'].apply(lambda x: len(tokenizer.tokenize(x)))

test['token_len'] = test['text'].apply(lambda x: len(tokenizer.tokenize(x)))
token_max_len = max(train['token_len'].max(), test['token_len'].max()) + 2

display(token_max_len)
from sklearn.model_selection import train_test_split, KFold, cross_val_score



def get_kfold_sets(train, k=5):

    kf = KFold(n_splits=k, shuffle=True)

    for train_texts, train_labels in kf.split(train.text.values, train.target.values):

        train_texts, valid_texts, train_labels, valid_labels = train_test_split(train.text.values, train.target.values, test_size=0.2)

        train_input = bert_encode(train_texts, tokenizer, max_len=token_max_len)

        valid_input = bert_encode(valid_texts, tokenizer, max_len=token_max_len)

        

        yield train_input, train_labels, valid_input, valid_labels



def get_train_sets(train):

    train_input = bert_encode(train.text.values, tokenizer, max_len=token_max_len)

    train_labels = train.target.values

    

    return train_input, train_labels
test_input = bert_encode(test.text.values, tokenizer, max_len=token_max_len)
from sklearn.metrics import f1_score

from keras.callbacks import Callback



class F1Callback(Callback):

    def __init__(self, model, X_val, y_val):

        self.model = model

        self.X_val = X_val

        self.y_val = y_val



    def on_epoch_end(self, epoch, logs):

        pred = self.model.predict(self.X_val)

        f1_val = f1_score(self.y_val, np.round(pred))

        print('\n f1_val = ', f1_val)
model_template = build_model(bert_layer, max_len=token_max_len)()

model_template.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.models import clone_model



def cross_val_score(train, k=3, epochs=10, batch_size=16):

    f1_vals = []

    models = []

    i = 0

    for train_input, train_labels, valid_input, valid_labels in get_kfold_sets(train, k=k):

        model = clone_model(model_template)

        model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy', 'mse'])

        train_history = model.fit(

            train_input, train_labels,

            validation_data=(valid_input, valid_labels),

            epochs=epochs,

            batch_size=batch_size,

            callbacks=[EarlyStopping(patience=1, monitor='val_mse', mode='min', verbose=True)]

        )

        pred = model.predict(valid_input)

        f1_val = f1_score(valid_labels, np.round(pred))

        print(f'f1-val: {f1_val}')

        f1_vals.append(f1_val)

        models.append(model)

        df = pd.DataFrame(train_history.history)

        df['f1-val'] = f1_val

        df.to_csv(f'history_{i}.csv')

        i += 1

    return np.array(f1_vals).mean(), models



k = 5

f1_val, models = cross_val_score(train, k=k)

print(f'f1-mean: {f1_val}')
train_input, train_labels = get_train_sets(train)
def calc_best_threshold(pred, labels):

    f1_vals = []

    ts = []

    for t in np.arange(0.1, 1, 0.1):

        f1_val = f1_score(train_labels, [1 if p >= t else 0 for p in train_pred])

        f1_vals.append(f1_val)

        ts.append(t)

    return ts[np.argmax(f1_vals)]



best_ts = []

for model in models:

    train_pred = model.predict(train_input)

    tmp = calc_best_threshold(train_pred, train_labels)

    best_ts.append(tmp)



print(f'best ts: {best_ts}')
test_preds = []

for model in models:

    test_pred = model.predict(test_input)

    test_preds.append(test_pred)



test_preds = np.array(test_preds)

print(test_preds.shape)
test_size = test_preds.shape[1]

mean_pred = []

for s in range(test_size):

    tmp = []

    for i in range(k):

#         tmp.append(test_preds[i][s][0].round())

        tmp.append(1 if test_preds[i][s][0] >= best_ts[i] else 0)

    mean = np.mean(tmp)

    mean_pred.append(mean)



mean_pred = np.array(mean_pred)

print(mean_pred.shape)

print(mean_pred[20:])

print(mean_pred[:20])
submission['target'] = mean_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)