from __future__ import print_function

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



%matplotlib inline

import numpy as np

import pandas as pd

import datetime, time, json

from keras.models import Model

from keras.layers import Input, Bidirectional, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization, concatenate

from keras.layers.embeddings import Embedding

from keras.regularizers import l2

from keras.callbacks import Callback, ModelCheckpoint

from keras import backend as K

from sklearn.model_selection import train_test_split

from math import log

from sklearn import metrics

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 你改这里的配置就行了 组合出一个过程

# 比如分出:    2的值 分别为 150, 200, 250, 300, 400的时候的解, 确定最优   也可以逐层加, 比如4层Dense dimension为[150,200, 300, 500]

# 5的值分别为 3,4,5,7  确定最优

# 其他的module可以置成 False 得到attention batch_normalization, BI LSTM的效果





ATTENTION_ENABLED = True # 1

HIDDEN_LAYER_DIMENSION = 200 # 2

BATCH_NORM_ENABLED = True # 3

DROP_OUT_ENABLED = True # 4

DENSE_LAYER_NUMS = 4 # 5

BI_DIRECT_ENABLED = True # 6



conf_lists = [

#     [False, 300, True, True, 4, True],

#     [True, 300, True, True, 4, True],

#     [True, 300, False, True, 4, True],

#     [True, 300, True, False, 4, True],

#     [True, 300, True, True, 4, False],

    

#     [True, 250, True, True, 4, True],

    [True, 300, True, True, 4, True],

#     [True, 400, True, True, 4, True],

#     [True, 500, True, True, 4, True],

    

#     [True, 300, True, True, 2, True],

#     [True, 300, True, True, 3, True],

#     [True, 300, True, True, 4, True],

#     [True, 300, True, True, 5, True],

#     [True, 300, True, True, 6, True]

]

Q1_TRAINING_DATA_FILE = 'q1.npy'

Q2_TRAINING_DATA_FILE = 'q2.npy'

LABEL_TRAINING_DATA_FILE = 'label.npy'

WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'

NB_WORDS_DATA_FILE = 'nb_words.json'

MODEL_WEIGHTS_FILE = 'question_pairs_weights.h5'

MAX_SEQUENCE_LENGTH = 25

WORD_EMBEDDING_DIM = 300

SENT_EMBEDDING_DIM = 128

VALIDATION_SPLIT = 0.1

TEST_SPLIT = 0.1

RNG_SEED = 13371447

NB_EPOCHS = 25

DROPOUT = 0.2

BATCH_SIZE = 512
q1_data = np.load(open('../input/nishizhu/'+Q1_TRAINING_DATA_FILE, 'rb'))

q2_data = np.load(open('../input/nishizhu/'+Q2_TRAINING_DATA_FILE, 'rb'))

labels = np.load(open('../input/nishizhu/'+LABEL_TRAINING_DATA_FILE, 'rb'))

word_embedding_matrix = np.load(open('../input/nishizhu/'+WORD_EMBEDDING_MATRIX_FILE, 'rb'))

with open('../input/nishizhu/'+NB_WORDS_DATA_FILE, 'r') as f:

    nb_json = json.load(f)

nb_words = nb_json['nb_words']
print(nb_words)

print(word_embedding_matrix.shape)
q1_train = q1_data[0:323480]

q1_test = q1_data[323480:]

q2_train = q2_data[0:323480]

q2_test = q2_data[323480:]

label_train = labels[0:323480]

label_test = labels[323480:]



print(len(q1_data))

print(len(q1_train), len(q1_test))

print(len(q2_train), len(q2_test))

print(len(label_train), len(label_test))
def cal_tf_idf(q1, q2, nb_words, q_len):

    tf_dic = {}

    for i in range(1, nb_words+1):

        tf_dic[i] = 0

    idf_dic = {}

    for i in range(1, nb_words+1):

        idf_dic[i] = 0

    

    q = list(q1)+list(q2)

    

    words_num = 0

    for q_idx in range(len(q)):

        query = q[q_idx]

        cur_idf_dic = {}

        for idx in query:

            if idx == 0:

                break

            words_num += 1

            try:

                tf_dic[idx] += 1

            except:

                print(query)

                print(q_idx)

            if idx not in cur_idf_dic:

                cur_idf_dic[idx] = 1

                idf_dic[idx] += 1

    

    tf_idf_dic = {}

    for i in range(1, nb_words+1):

        tf = tf_dic[i]/words_num

        idf = log(q_len/(idf_dic[i]+1))

        tf_idf_dic[i] = tf*idf

    return tf_idf_dic

#     tf = []

#     for query in q:

#         len_count = 0

#         cur_tf = {}

#         for idx in query:

#             if idx == 0:

#                 break

#             len_count += 1

#             if idx not in cur_tf:

#                 cur_tf[idx] = 1

#             else:

#                 cur_tf[idx] += 1

            

#             if cur_tf[idx] == 1:

#                 idf_dic[idx] += 1

                

#         cur_q_tf = [0]*len(query)

#         for i in range(len(len_count)):

#             cur_q_tf[i] = cur_tf[query[i]]/len_count

#         tf.append(cur_q_tf)

    

    

#     tf_idf = []

    





        
# tf_idf_train = cal_tf_idf(q1_train, q2_train, nb_words, len(q1_train))

# for idx in tf_idf_train.keys():

#     word_embedding_matrix[idx] = word_embedding_matrix[idx]*tf_idf_train[idx]

X = np.stack((q1_train, q2_train), axis=1)

y = label_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RNG_SEED)

Q1_train = X_train[:,0]

Q2_train = X_train[:,1]

Q1_test = X_test[:,0]

Q2_test = X_test[:,1]
def build_model(ATTENTION_ENABLED, HIDDEN_LAYER_DIMENSION, BATCH_NORM_ENABLED, DROP_OUT_ENABLED, DENSE_LAYER_NUMS, BI_DIRECT_ENABLED):

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))



    q1 = Embedding(nb_words + 1, 

                     WORD_EMBEDDING_DIM, 

                     weights=[word_embedding_matrix], 

                     input_length=MAX_SEQUENCE_LENGTH, 

                     trainable=False)(question1)



    if BI_DIRECT_ENABLED == True:

        q1 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q1)

    else:

        q1 = LSTM(SENT_EMBEDDING_DIM, return_sequences=True)(q1)



    q2 = Embedding(nb_words + 1, 

                     WORD_EMBEDDING_DIM, 

                     weights=[word_embedding_matrix], 

                     input_length=MAX_SEQUENCE_LENGTH, 

                     trainable=False)(question2)

    if BI_DIRECT_ENABLED == True:

        q2 = Bidirectional(LSTM(SENT_EMBEDDING_DIM, return_sequences=True), merge_mode="sum")(q2)

    else:

        q2 = LSTM(SENT_EMBEDDING_DIM, return_sequences=True)(q2)







    # attention = dot([q1,q2], [1,1])

    # attention = Flatten()(attention)

    # attention = Dense((MAX_SEQUENCE_LENGTH*SENT_EMBEDDING_DIM))(attention)

    # attention = Reshape((MAX_SEQUENCE_LENGTH, SENT_EMBEDDING_DIM))(attention)





    if ATTENTION_ENABLED == True:

        attention = dot([q1,q2], [1,1])

        attention = Flatten()(attention)

        attention = Dense((MAX_SEQUENCE_LENGTH*SENT_EMBEDDING_DIM))(attention)

        attention = Reshape((MAX_SEQUENCE_LENGTH, SENT_EMBEDDING_DIM))(attention)

        merged = add([q1,attention])

    else:

        merged = concatenate([q1,q2])





    merged = Flatten()(merged)



    for i in range(0, DENSE_LAYER_NUMS):

        merged = Dense(HIDDEN_LAYER_DIMENSION, activation='relu')(merged)

        if DROP_OUT_ENABLED == True:

            merged = Dropout(DROPOUT)(merged)

        if BATCH_NORM_ENABLED == True:

            merged = BatchNormalization()(merged)







    is_duplicate = Dense(1, activation='sigmoid')(merged)



    model = Model(inputs=[question1,question2], outputs=is_duplicate)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model.summary()
result_list = []

for conf in conf_lists:



    ATTENTION_ENABLED = conf[0]

    HIDDEN_LAYER_DIMENSION = conf[1]

    BATCH_NORM_ENABLED = conf[2]

    DROP_OUT_ENABLED = conf[3]

    DENSE_LAYER_NUMS = conf[4]

    BI_DIRECT_ENABLED = conf[5]



    model = build_model(ATTENTION_ENABLED, HIDDEN_LAYER_DIMENSION, BATCH_NORM_ENABLED, DROP_OUT_ENABLED, DENSE_LAYER_NUMS, BI_DIRECT_ENABLED)

    print("Starting training at", datetime.datetime.now())

    t0 = time.time()

    callbacks = [ModelCheckpoint('/kaggle/working/'+MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]

    history = model.fit([Q1_train, Q2_train],

                        y_train,

                        epochs=NB_EPOCHS,

                        validation_split=VALIDATION_SPLIT,

                        verbose=2,

                        batch_size=BATCH_SIZE,

                        callbacks=callbacks)

    t1 = time.time()



    print("Training ended at", datetime.datetime.now())

    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

    

    y = model.predict([q1_test,q2_test])

    soft_y = []

    for i in range(len(y)):

        if y[i] <= 0.5:

            soft_y.append(0)

        else:

            soft_y.append(1)

    label_test = np.array(label_test)

    soft_y = np.array(soft_y)

    precision = metrics.precision_score(label_test, soft_y)

    recall = metrics.recall_score(label_test, soft_y)

    f1_score = metrics.f1_score(label_test, soft_y)

    

    print(conf)

    print(precision, recall, f1_score)

    result_list.append([precision, recall, f1_score])

    

    # ATTENTION_ENABLED = True # 1

# HIDDEN_LAYER_DIMENSION = 200 # 2

# BATCH_NORM_ENABLED = True # 3

# DROP_OUT_ENABLED = True # 4

# DENSE_LAYER_NUMS = 4 # 5

# BI_DIRECT_ENABLED = True # 6



# conf_lists = [
print(result_list)
print("Starting training at", datetime.datetime.now())

t0 = time.time()

callbacks = [ModelCheckpoint('/kaggle/working/'+MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]

history = model.fit([Q1_train, Q2_train],

                    y_train,

                    epochs=NB_EPOCHS,

                    validation_split=VALIDATION_SPLIT,

                    verbose=2,

                    batch_size=BATCH_SIZE,

                    callbacks=callbacks)

t1 = time.time()



print("Training ended at", datetime.datetime.now())

print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
y = model.predict([q1_test,q2_test])
soft_y = []

for i in range(len(y)):

    if y[i] <= 0.5:

        soft_y.append(0)

    else:

        soft_y.append(1)
print(len(soft_y))
label_test = np.array(label_test)

soft_y = np.array(soft_y)




print(metrics.precision_score(label_test, soft_y))



print(metrics.recall_score(label_test, soft_y))



print(metrics.f1_score(label_test, soft_y))