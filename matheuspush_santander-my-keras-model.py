import numpy as np, pandas as pd

from keras import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

import keras

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import os



batch = 1024

SEED = 13

np.random.seed(SEED)

print(os.listdir("../input"))
# Load data

data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



y = data['target']

X = data.drop(columns=['ID_code', 'target'])



y = y.astype('int8')

X = X.astype('float16')



print("data loaded")
# Data processing

scaler = StandardScaler()

X = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    stratify=y, 

                                                    test_size=0.2,

                                                    random_state=13)
def auc_roc(y_true, y_pred):

    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)



    # find all variables created for this metric

    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]



    # Add metric variables to GLOBAL_VARIABLES collection.

    # They will be initialized for new session.

    for v in metric_vars:

        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)



    # force to update metric values

    with tf.control_dependencies([update_op]):

        value = tf.identity(value)

        return value
%%time

best_params = ''

best_auc = 0

i = 1



for pat in [10]:

    for w in [8]:

        for neu1 in [256]:

            for neu2 in [0.5]:

                for drop in [0.5]:

                    for opt in ['sgd']:

                        

                        print('Fitting...', i)

                        i += 1



                        model = Sequential()  # Instantiate sequential model

                        model.add(Dense(neu1, activation='relu', input_dim=200)) # Add first layer. Make sure to specify input shape

                        if drop > 0:

                            model.add(Dropout(drop)) # Add second layer                            

                        model.add(Dense(int(neu1 * neu2), activation='relu'))

                        # model.add(Dropout(0.2)) # Add second layer

                        # model.add(Dense(256, activation='relu'))

                        model.add(Dense(1, activation='sigmoid')) # Add third layer



                        model.compile(optimizer=opt,

                                      loss='binary_crossentropy',

                                      metrics=[auc_roc])



                        # change patience

                        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=pat)

                        mc = ModelCheckpoint('best_model.h5', monitor='val_auc_roc', mode='max', verbose=0, save_best_only=True)



                        # change weights

                        history = model.fit(X_train, y_train, 

                                            batch_size=batch, 

                                            validation_data=(X_test, y_test), 

                                            epochs=200, verbose=0, 

                                            callbacks=[es, mc], 

                                            class_weight = {0: 1, 1: w})





                        saved_model = load_model('best_model.h5', custom_objects={'auc_roc': auc_roc})



                        y_pred = saved_model.predict(X_test)

                        auc = metrics.roc_auc_score(y_test, np.rint(y_pred))



                        if auc > best_auc:

                            best_auc = auc

                            best_params = (pat, w, neu1, neu2, drop, opt)



                        print(best_params)

                        print(best_auc)
# Save outputc file

id_code = test.pop('ID_code')

test = test.astype('float16')

test = scaler.transform(test)

targets = saved_model.predict(test)

output = pd.DataFrame({'ID_code': id_code, 'target': np.rint(targets[:,0])})

output.to_csv('output.csv', index=False)