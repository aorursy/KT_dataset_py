import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from zipfile import ZipFile

from time import time

from numpy import empty



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')

df_train.head()
texts = df_train[['question1','question2']]

labels = df_train['is_duplicate']



del df_train
# Model params

MAX_NB_WORDS = 100000

MAX_SEQUENCE_LENGTH = 128

VALIDATION_SPLIT = 0.1

EMBEDDING_DIM = 64



# Train params

NB_EPOCHS = 2

BATCH_SIZE = 1024

VAL_SPLIT = 0.1

WEIGHTS_PATH = 'lstm_weights.h5'

SUBMIT_PATH = 'lstm_submission_1.csv'
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



tk = Tokenizer(nb_words=MAX_NB_WORDS)



tk.fit_on_texts(list(texts.question1.values.astype(str)) + list(texts.question2.values.astype(str)))

x1 = tk.texts_to_sequences(texts.question1.values.astype(str))

x1 = pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)



x2 = tk.texts_to_sequences(texts.question2.values.astype(str))

x2 = pad_sequences(x2, maxlen=MAX_SEQUENCE_LENGTH)



# Preprocessing Test

print("Acquiring Test Data")

t0 = time()

df_test = pd.read_csv('../input/test.csv')

print("Done! Acquisition time:", time()-t0)



# Preprocessing

print("Preprocessing test data")

t0 = time()



i = 0

while True:

    if (i*BATCH_SIZE > df_test.shape[0]):

        break

    t1 = time()

    tk.fit_on_texts(list(df_test.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].question1.values.astype(str))

                    + list(df_test.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].question2.values.astype(str)))

    i += 1

    if (i % 100 == 0):

        print("Preprocessed Batch {0}/{1}, Word index size: {2}, ETC: {3} seconds".format(i,

                                                                int(df_test.shape[0]/BATCH_SIZE+1),

                                                                len(tk.word_index),

                                                                int(int(df_test.shape[0]/BATCH_SIZE+1)-i)*(time()-t1)))



word_index = tk.word_index



print("Done! Preprocessing time:", time()-t0)

print("Word index length:",len(word_index))



print('Shape of data tensor:', x1.shape, x2.shape)

print('Shape of label tensor:', labels.shape)
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Bidirectional, merge, Activation, Embedding

from keras.models import Sequential, load_model, Model ,Input

from keras.layers.normalization import BatchNormalization

from keras.callbacks import ModelCheckpoint

from keras import backend as K



def get_model(p_drop=0.0):

    embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=False)

                            

    shared_lstm = Bidirectional(LSTM(64))

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedded_sequences_1 = embedding_layer(sequence_1_input)

    x1 = shared_lstm(embedded_sequences_1)





    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedded_sequences_2 = embedding_layer(sequence_2_input)

    y1 = shared_lstm(embedded_sequences_2)



    merged = merge([x1,y1], mode='concat')

    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(input=[sequence_1_input,sequence_2_input], output=preds)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    

    return model
'''

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV



print("Searching for optimum hyper parameters.")

t0 = time()

model = KerasClassifier(build_fn=get_model, verbose=0)



# define the grid search parameters

batch_size = [128, 256, 512, 1024, 2048]

p_drop = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

param_grid = dict(batch_size=batch_size, p_drop=p_drop)



grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

grid_result = grid.fit([x1, x2], labels)



# summarize results

print("Best: {0} using {1}".format(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

print("Done! Time elapsed:", time()-t0)

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))

'''

    

# Usually, this is the code for gridsearch a keras model with sklearn, however for the merged model,

# i got this error. As I can't find a solution for this error on the web and don't have the time to dig

# deeper, I'll appreciate to hear your insights about how to do it, if you have some to share!
model = get_model(p_drop=0.2)

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)



model.fit([x1, x2], y=labels, batch_size=1024, nb_epoch=2,

                 verbose=1, validation_split=0.1, shuffle=True, callbacks=[checkpoint])
# Load best model

#print("Loading best trained model")

#model = load_model(WEIGHTS_PATH)



# Predicting

i = 0

predictions = empty([df_test.shape[0],1])

while True:

    t1 = time()

    if (i * BATCH_SIZE > df_test.shape[0]):

        break

    x1 = pad_sequences(tk.texts_to_sequences(

        df_test.question1.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].values.astype(str)), maxlen=MAX_SEQUENCE_LENGTH)

    x2 = pad_sequences(tk.texts_to_sequences(

        df_test.question2.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].values.astype(str)), maxlen=MAX_SEQUENCE_LENGTH)

    try:

        predictions[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = model.predict([x1, x2], batch_size=BATCH_SIZE, verbose=0)

    except ValueError:

        predictions[i*BATCH_SIZE:] = model.predict([x1, x2], batch_size=BATCH_SIZE, verbose=0)[:(df_test.shape[0]-i*BATCH_SIZE)]



    i += 1

    if (i % 1000 == 0):

        print("Predicted Batch {0}/{1}, ETC: {2} seconds".format(i,

                                                                int(df_test.shape[0]/BATCH_SIZE),

                                                                int(int(df_test.shape[0]/BATCH_SIZE+1)-i)*(time()-t1)))





df_test["is_duplicate"] = predictions





df_test[['test_id','is_duplicate']].to_csv(SUBMIT_PATH, header=True, index=False)

print("Done!")

print("Submission file saved to:",check_output(["ls", SUBMIT_PATH]).decode("utf8"))