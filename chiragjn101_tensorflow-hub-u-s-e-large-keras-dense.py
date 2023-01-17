import numpy as np
import pandas as pd
import os
import math
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.layers import (Input, Dense, 
                          LeakyReLU, BatchNormalization,
                          Activation, Dropout)
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
g = tf.Graph()
with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    use = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3", trainable=False)
    text_use_embedded = use(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)

_ = session.run(text_use_embedded, feed_dict={text_input: ["Hello world"]})
print(len(train), len(test))
def embed(texts):
    use_batch_size = 1024
    if len(texts) <= use_batch_size:
        return session.run(text_use_embedded, feed_dict={text_input: texts})
    # we query in small sizes to avoid blowing up
    n_batches = math.ceil(len(texts) / use_batch_size)
    np_arrs = []
    for i in tqdm.tqdm_notebook(range(n_batches), total=n_batches):
        start = i * use_batch_size
        end = (i + 1) * use_batch_size
        texts_batch = texts[start:end]
        np_arrs.append(session.run(text_use_embedded, feed_dict={text_input: texts_batch}))
    return np.concatenate(np_arrs, axis=0)
# Speed test

train_vecs_tmp = embed(train['question_text'].iloc[:2 ** 12].tolist())
print(train_vecs_tmp.shape)
batch_size = 256

def batch_gen(_train_df):
    _train_neg_df = _train_df[_train_df['target'] == 0]
    _train_pos_df = _train_df[_train_df['target'] == 1]
    _batch_size = batch_size // 2
    # Since positive samples are way too less
    n_batches = math.ceil(len(_train_pos_df) / _batch_size)
    while True: 
        _train_neg_df = _train_neg_df.sample(frac=1.)  # Shuffle the data.
        _train_pos_df = _train_pos_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            start = i * _batch_size
            end = (i + 1) * _batch_size
            batch_df = pd.concat([_train_neg_df.iloc[start:end], _train_pos_df.iloc[start:end]]).sample(frac=1.)
            # print('-- DEBUG --', len(batch_df))
            texts_vectors_batch = embed(batch_df['question_text'].tolist())
            texts_targets_batch = np.array(batch_df['target'])
            yield texts_vectors_batch, texts_targets_batch
# # From https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
# #         self.val_f1s = []
# #         self.val_recalls = []
# #         self.val_precisions = []
#         pass

#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = np.asarray(self.model.predict(self.validation_data[0]))
#         val_targ = self.validation_data[1]
#         for thresh in np.arange(0.1, 1.0, 0.05):
#             thresh = np.round(thresh, 2)
#             print('F1 score at threshold {0} is {1}'.format(thresh, f1_score(val_targ, (val_predict > thresh).astype(int))))
# #         _val_f1 = f1_score(val_targ, val_predict)
# #         _val_recall = recall_score(val_targ, val_predict)
# #         _val_precision = precision_score(val_targ, val_predict)
# #         self.val_f1s.append(_val_f1)
# #         self.val_recalls.append(_val_recall)
# #         self.val_precisions.append(_val_precision)
# #         print('— val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))

# metrics = Metrics()
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
# def run_model(_train_df, _val_df, epochs=5, steps_per_epoch=5000):
#     model = Sequential([
#             Dropout(0.3, input_shape=(512,)),
#             Dense(1),
#             BatchNormalization(),
#             Activation('sigmoid'),
#         ])
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     val_vectors = embed(_val_df['question_text'].tolist())
#     val_y = _val_df['target'].values
#     train_batch_generator = batch_gen(_train_df)
#     # val_batch_generator = batch_gen(_val_df)
#     # steps_per_epoch = min(5000, len(_train_df) // batch_size)
#     model.fit_generator(generator=train_batch_generator,
#                         steps_per_epoch=steps_per_epoch,
#                         validation_data=(val_vectors, val_y),
#                         # validation_data=val_batch_generator,
#                         # validation_steps=len(_val_df) // batch_size,
#                         epochs=epochs,
#                         callbacks=[reduce_lr, metrics],
#                         verbose=True)
# # Test the setup

# sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.01, random_state=42)
# for dev_index, val_index in sss.split(train['question_text'], train['target']):    
#     _train = train.iloc[dev_index, :]
#     _val = train.iloc[val_index, :]
#     print('------')
#     print(len(_train), len(_val))
#     run_model(_train, _val, 10, 50)
# final_model = Sequential([
#                 Dropout(0.1, input_shape=(512,)),
#                 Dense(32),
#                 BatchNormalization(),
#                 Activation('relu'),
#                 Dropout(0.3),
#                 Dense(1, input_shape=(512,)),
#                 BatchNormalization(),
#                 Activation('sigmoid'),
#               ])


final_model = Sequential([
                Dropout(0.3, input_shape=(512,)),
                Dense(1),
                BatchNormalization(),
                Activation('sigmoid'),
              ])
sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
final_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
batch_generator = batch_gen(train)
final_model.fit_generator(batch_generator,
                          epochs=20,
                          steps_per_epoch=50,
                          verbose=True)
submission = pd.DataFrame.from_dict({'qid': test['qid']})
predictions = final_model.predict(embed(test['question_text'].tolist()))
predictions = (predictions > 0.65).astype(int)
submission['prediction'] = predictions
submission.to_csv('submission.csv', index=False)
pd.value_counts(submission['prediction'])
