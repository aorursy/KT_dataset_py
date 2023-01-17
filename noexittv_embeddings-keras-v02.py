import pandas as pd

from collections import Counter

import numpy as np

import time

from sklearn.model_selection import train_test_split

from sklearn import metrics



# We'll have to import all keras stuff here later

from keras.layers import Bidirectional, Dense, Dropout, Embedding, CuDNNLSTM, CuDNNGRU, Input, GlobalMaxPool1D

from keras.callbacks import ModelCheckpoint

from keras.models import Model, load_model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras import optimizers



# Can keras find a gpu?

from keras import backend as K

print(K.tensorflow_backend._get_available_gpus())
# Helper function that will be used in all cells!

def timeSince(t0):

    ''' This function will be used to print the time since t0. 

        Will be called in every cell to give me some measurement. '''

    print('Cell complete in {:.0f}m {:.0f}s'.format((time.time()-t0) // 60, (time.time()-t0) % 60))
# Dataset path

_traindataset = '../input/train.csv'

_testdataset = '../input/test.csv'



# Embeddings path

_glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

_paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

_wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

_google_news = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'



embeddings = [{'name': 'glove', 'path': _glove},

              {'name': 'paragram', 'path': _paragram},

              {'name': 'fasttext', 'path': _wiki_news}]



# Other constants here?
t0 = time.time()

df_train = pd.read_csv(_traindataset)

df_test = pd.read_csv(_testdataset)

timeSince(t0)
t0 = time.time()



hparam = {}

hparam['VOCAB_SIZE'] = 50000

hparam['PAD_LENGTH'] = 100

hparam['MINIBATCH_SIZE'] = 512

hparam['LEARNING_RATE'] = 1e-3

hparam['EPOCHS'] = 5

hparam['LSTM_HIDDEN_SIZE'] = 128

hparam['WORD_EMB_DIM'] = 300 # This will be set per embedding but all should be 300

    

timeSince(t0)
t0 = time.time()

df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=2019)

timeSince(t0)
t0 = time.time()

        

X_train = df_train['question_text'].fillna("_nan_").str.lower().values

X_val = df_val['question_text'].fillna("_nan_").str.lower().values  

X_test = df_test['question_text'].fillna("_nan_").str.lower().values  

y_train = df_train['target'].values

y_val = df_val['target'].values

qid_test = df_test['qid'].values



print("Lenth X_train, y_train = {}, {}".format(len(X_train), len(y_train)))

print("Lenth X_val, y_val = {}, {}".format(len(X_val), len(y_val)))

print("Lenth X_test, qid_test = {}, {}".format(len(X_test), len(qid_test)))



timeSince(t0)
t0 = time.time()



vocab_size = hparam['VOCAB_SIZE']

pad_length = hparam['PAD_LENGTH']



## Tokenize the sentences

tokenizer = Tokenizer(num_words=vocab_size)

tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train_token = tokenizer.texts_to_sequences(X_train)

X_val_token = tokenizer.texts_to_sequences(X_val)

X_test_token = tokenizer.texts_to_sequences(X_test)



## Pad the sentences 

X_train_pad = pad_sequences(X_train_token, maxlen=pad_length)

X_val_pad = pad_sequences(X_val_token, maxlen=pad_length)

X_test_pad = pad_sequences(X_test_token, maxlen=pad_length)



# pack into placeholder

dataset = {'X_train': X_train_pad, 'y_train': y_train,

          'X_val': X_val_pad, 'y_val': y_val,

          'X_test': X_test_pad}



print("X_train_pad.shape: {}".format(X_train_pad.shape))

print("y_train.shape: {}".format(y_train.shape))

print("X_val_pad.shape: {}".format(X_val_pad.shape))

print("y_val.shape: {}".format(y_val.shape))

print("X_test_pad.shape: {}".format(X_test_pad.shape))



print(X_train[3])

print(X_train_pad[3])

print(len(tokenizer.word_counts))

print(len(tokenizer.word_index))



timeSince(t0)
t0 = time.time()



def load_embed(file):

    ''' Load the embedding from file '''

    def get_coefs(word, *arr):

        return word, np.asarray(arr, dtype='float32')

    

    if file.split('/')[-1] == 'wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index



def create_emb_matrix(emb_mean, emb_std, nb_words, embed_size):

    ''' Creates a initial random embedding matrix 

        All words that are not found in the embedding will thus be a random vector

        Can maybe be imrpoved by initializing all words not found to the same random vector '''

    return np.random.normal(emb_mean, emb_std, (nb_words, embed_size))



def fill_emb_matrix(word_idx, emb_matrix, emb_index, vocab_size):

    ''' Created a word2vec format matrix that we can use to embed our words '''

    for word, i in word_idx:

        if i >= vocab_size:

            return emb_matrix

        emb_vector = emb_index.get(word)

        if emb_vector is not None:

            emb_matrix[i] = emb_vector

    return emb_matrix



timeSince(t0)
t0 = time.time()



def train_val_pred(dataset, hparam, embedding_matrix):

    ''' This function will train a model using some embedding matrix.

        The prediction threshold will then be calculated based on the threshold that gives

        the best f1 score on the validation data. The predictions on the test set will then

        be returned along with the best calculated f1 score '''

    

    # Get data from dataset

    X_train = dataset['X_train']

    y_train = dataset['y_train']

    X_val = dataset['X_val']

    y_val = dataset['y_val']

    X_test = dataset['X_test']



    # Get hyperparameters

    VOCAB_SIZE = hparam['VOCAB_SIZE']

    PAD_LENGTH = hparam['PAD_LENGTH']

    MINIBATCH_SIZE = hparam['MINIBATCH_SIZE']

    LEARNING_RATE = hparam['LEARNING_RATE']

    EPOCHS = hparam['EPOCHS']

    LSTM_HIDDEN_SIZE = hparam['LSTM_HIDDEN_SIZE']

    WORD_EMB_DIM = hparam['WORD_EMB_DIM']

    

    # Create the model

    inp = Input(shape=(PAD_LENGTH,))

    x = Embedding(VOCAB_SIZE, WORD_EMB_DIM, weights=[embedding_matrix], trainable=True)(inp)

    x = Bidirectional(CuDNNLSTM(LSTM_HIDDEN_SIZE, return_sequences=True))(x)

    x = GlobalMaxPool1D()(x)

    x = Dense(16, activation="relu")(x)

    x = Dropout(0.1)(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    adam = optimizers.Adam(lr=LEARNING_RATE)

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.summary()

    

    # Train model

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=MINIBATCH_SIZE, 

          validation_data = (X_val, y_val))

    

    # Get threshold that gives best f1 score on validation set

    val_preds = model.predict(X_val, batch_size=MINIBATCH_SIZE, verbose=1)

    best_f1 = -1

    best_thresh = -1

    for thresh in np.arange(0.1, 0.501, 0.01):

        thresh = np.round(thresh, 2)

        f1 = metrics.f1_score(y_val, (val_preds > thresh).astype(int))

        if f1 > best_f1:

            best_f1 = f1

            best_thresh = thresh

    print("Best f1 score = {} at tresh {}".format(best_f1, best_thresh))

    

    # Get predictions on test set

    test_preds = model.predict(X_test)

    

    # Some memory management!

    del embedding_matrix, model, inp, x, adam

    import gc; gc.collect()

    

    # Return predictions and the thresh that gave best f1 score on validation data

    return test_preds, val_preds, best_thresh, best_f1

    

timeSince(t0)
t0 = time.time()



# Non-iterable vars

word_index = tokenizer.word_index

nb_words = min(hparam['VOCAB_SIZE'], len(word_index))



timeSince(t0)
t0 = time.time()



# Define some list where we save the results!

results = []

    

for embedding in embeddings:

    emb_name = embedding['name']

    emb_path = embedding['path']

    print("Running procedure on {}".format(emb_name))

    

    # Load embedding

    emb_index = load_embed(emb_path)

    all_emb = np.stack(list(emb_index.values()))

    emb_mean, emb_std = all_emb.mean(), all_emb.std()

    emb_size = all_emb.shape[1]

    hparam['WORD_EMB_DIM'] = emb_size # Set this! Not really needed tho...

    print("{} mean: {}, std: {}, size: {}".format(emb_name, emb_mean, emb_std, emb_size))

    

    # Convert emb to word2vec format

    emb_matrix = create_emb_matrix(emb_mean, emb_std, nb_words, emb_size)

    emb_matrix = fill_emb_matrix(word_index.items(), emb_matrix, emb_index, hparam['VOCAB_SIZE'])

    

    # Run entire procedure and get predictions, best threshold and best f1 score

    test_preds, val_preds, thresh, f1 = train_val_pred(dataset, hparam, emb_matrix)

    

    print("len(test_preds) = {}, len(val_preds) = {}, thresh = {} at f1 = {}".format(len(test_preds), 

                                                                                     len(val_preds), 

                                                                                     thresh, 

                                                                                     f1))

    # Save into results

    new_result = {'name': emb_name, 

                  'test_preds': test_preds, 

                  'val_preds': val_preds, 

                  'thresh': thresh, 

                  'f1': f1}

    results.append(new_result)

    

    # Memory management!

    del emb_index, all_emb, emb_mean, emb_std, emb_size, emb_matrix

    import gc; gc.collect()

    time.sleep(10)



timeSince(t0)
t0 = time.time()



print("Got {} number of results!".format(len(results)))

avg_thresh = 0

for result in results:

    print("{} gave f1 score {} with thresh {}".format(result['name'], result['f1'], result['thresh']))

    avg_thresh += result['thresh']



avg_thresh = avg_thresh / len(results)

print("Got an average threshold at {}".format(avg_thresh))



timeSince(t0)
t0 = time.time()



pred_glove_val = results[0]['val_preds']

preds_paragram_val = results[1]['val_preds']

preds_fasttext_val = results[2]['val_preds']



pred_val_y = 0.33*pred_glove_val + 0.33*preds_paragram_val + 0.34*preds_fasttext_val 



best_f1_combined = -1

best_thresh_combined = -1

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    f1 = metrics.f1_score(dataset['y_val'], (pred_val_y > thresh).astype(int))

    if f1 > best_f1_combined:

        best_f1_combined = f1

        best_thresh_combined = thresh

    #print("F1 score at threshold {0} is {1}".format(thresh, f1))

    

print("Best f1 score = {} at tresh {}".format(best_f1_combined, best_thresh_combined))



timeSince(t0)
t0 = time.time()



print("Using treshold {}".format(best_thresh_combined))



pred_glove_test = results[0]['test_preds']

preds_paragram_test = results[1]['test_preds']

preds_fasttext_test = results[2]['test_preds']



pred_test_y = 0.33*pred_glove_test + 0.33*preds_paragram_test + 0.34*preds_fasttext_test

pred_test_y_res = (pred_test_y > best_thresh_combined).astype(int)



results_dict = {'qid':qid_test, 'prediction':[]}



for prediction in pred_test_y_res:

    results_dict['prediction'].append(prediction[0])



    

print(results_dict['qid'][:15])

print(results_dict['prediction'][:15])



    

# Save results

df = pd.DataFrame(data=results_dict)

df.to_csv('submission.csv', index=False)

print("Saved csv to disk!")



timeSince(t0)