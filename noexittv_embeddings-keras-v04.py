import pandas as pd

from collections import Counter

import numpy as np

import time

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

import operator

import re

import gc



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
t0 = time.time()

df_train = pd.read_csv(_traindataset)

df_test = pd.read_csv(_testdataset)

timeSince(t0)
t0 = time.time()



df_train['processed_questions'] = df_train['question_text'].fillna("_nan_").apply(lambda x: x.lower())

df_test['processed_questions'] = df_test['question_text'].fillna("_nan_").apply(lambda x: x.lower())



timeSince(t0)
t0 = time.time()



contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 

                       "could've": "could have", "couldn't": "could not", "didn't": "did not",  

                       "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 

                       "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 

                       "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",

                       "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 

                       "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 

                       "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 

                       "mayn't": "may not", "might've": "might have","mightn't": "might not",

                       "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 

                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                       "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 

                       "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 

                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 

                       "she'll've": "she will have", "she's": "she is", "should've": "should have", 

                       "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",

                       "so's": "so as", "this's": "this is","that'd": "that would", 

                       "that'd've": "that would have", "that's": "that is", "there'd": "there would", 

                       "there'd've": "there would have", "there's": "there is", "here's": "here is",

                       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", "they've": "they have", 

                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", 

                       "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 

                       "weren't": "were not", "what'll": "what will", "what'll've": "what will have", 

                       "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", 

                       "when've": "when have", "where'd": "where did", "where's": "where is", 

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", 

                       "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", 

                       "will've": "will have", "won't": "will not", "won't've": "will not have", 

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 

                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",

                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                       "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 

                       "you're": "you are", "you've": "you have"}



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



df_train['processed_questions'] = df_train['processed_questions'].apply(lambda x: 

                                                                    clean_contractions(x, 

                                                                                       contraction_mapping))

df_test['processed_questions'] = df_test['processed_questions'].apply(lambda x: 

                                                                  clean_contractions(x, 

                                                                                     contraction_mapping))

timeSince(t0)
t0 = time.time()



punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'



punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", 

                 "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', 

                 '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 

                 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}



def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text



df_train['processed_questions'] = df_train['processed_questions'].apply(lambda x: clean_special_chars(x, 

                                                                                      punct, 

                                                                                      punct_mapping))

df_test['processed_questions'] = df_test['processed_questions'].apply(lambda x: clean_special_chars(x, 

                                                                                      punct, 

                                                                                      punct_mapping))

timeSince(t0)
t0 = time.time()



# Seems like this lowers the f1 score. Investigate further

#df_train['processed_questions'] = df_train['processed_questions'].apply(lambda x: re.sub(r'[^\x20-\x7e]',r'', x))

#df_test['processed_questions'] = df_test['processed_questions'].apply(lambda x: re.sub(r'[^\x20-\x7e]',r'', x))



timeSince(t0)
t0 = time.time()



def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



def vocab_to_integer(vocab):

    ''' Map each vocab words to an integer.

        Starts at 1 since 0 will be used for padding.'''

    return {word: ii for ii, word in enumerate(vocab, 1)}

    

    

all_questions = pd.concat([df_train['processed_questions'], df_test['processed_questions']])

final_vocab = build_vocab(all_questions)

word_to_idx = vocab_to_integer(final_vocab)



timeSince(t0)
t0 = time.time()



hparam = {}

hparam['VOCAB_SIZE'] = len(final_vocab) + 1

hparam['PAD_LENGTH'] = 77

hparam['MINIBATCH_SIZE'] = 512

hparam['LEARNING_RATE'] = 1e-3

hparam['EPOCHS'] = 4

hparam['LSTM_HIDDEN_SIZE'] = 128

hparam['WORD_EMB_DIM'] = 0 # This will be set when we concatenate the embeddings!

hparam['KFOLDS'] = 10



# To add

# padding = pre

# truncating = pre

    

timeSince(t0)
t0 = time.time()



# Original vocab

vocab_original = build_vocab(pd.concat([df_train['question_text'], df_test['question_text']]))



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



def check_coverage(vocab, embeddings_index):

    ''' Checks the coverate of a vocabulary in a given embedding.

        Returns an array of out of vocab words (oov) '''

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass

    print('{} iv words, {} unique'.format(nb_known_words, len(known_words)))

    print('{} oov words, {} unique'.format(nb_unknown_words, len(unknown_words)))

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words



def create_emb_matrix(nb_words, embed_size):

    ''' Creates a initial random embedding matrix '''

    # This is now zeroes which means that all words that doesn't have an embedding

    # will be a zero vector... Before this was random.normalized! Maybe change back if score gets worse!

    return np.zeros((nb_words, embed_size), dtype=np.float32)



def fill_emb_matrix(word_idx, emb_matrix, emb_index):

    ''' Created a word2vec format matrix that we can use to embed our words '''

    for word, i in word_idx:

        emb_vector = emb_index.get(word)

        if emb_vector is not None:

            emb_matrix[i] = emb_vector

    return emb_matrix



def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")



timeSince(t0)
t0 = time.time()



conc_embedding = None # Concatenated embeddings will be saved as this variable

word_index = word_to_idx

nb_words = min(hparam['VOCAB_SIZE'], len(word_index) + 1) # this step should be unescessary???

hparam['VOCAB_SIZE'] = nb_words

print(hparam['VOCAB_SIZE'], len(word_index) + 1)

print(f"Got a vocab size of {nb_words} number of words")



for embedding in embeddings:

    emb_name = embedding['name']

    emb_path = embedding['path']

    print("Running procedure on {}".format(emb_name))

    

    # Load embedding

    print("Loading {}".format(emb_name))

    emb_index = load_embed(emb_path)

    

    # Add lowercase words to embedding

    print("Adding lowercase to {}".format(emb_name))

    add_lower(emb_index, vocab_original)

    

    # Check OOV score

    _ = check_coverage(final_vocab, emb_index)

    

    emb_size = 300

    hparam['WORD_EMB_DIM'] += emb_size

    

    # Convert emb to word2vec format

    emb_matrix = create_emb_matrix(nb_words, emb_size)

    print(emb_matrix.size)

    print(emb_matrix.shape)

    emb_matrix = fill_emb_matrix(word_index.items(), emb_matrix, emb_index)

    

    # Save or concatenate this embedding with the previous embedding

    if conc_embedding is not None:

        conc_embedding = np.concatenate((conc_embedding, emb_matrix), axis=1)

        print("concatenated! now got shape: {}".format(conc_embedding.shape))

        #conc_embedding += emb_matrix

        #print("Added! now got shape: {}".format(conc_embedding.shape))

    else:

        conc_embedding = emb_matrix

    

    # Memory management!

    del emb_matrix, emb_index, emb_name, emb_path, emb_size

    import gc; gc.collect()

    

timeSince(t0)
t0 = time.time()        



def embed_word_to_int(X, vocab_to_int):

    embedded_X = []

    for q in X:

        tmp_X = []

        for w in q.split():

            tmp_X.append(vocab_to_int[w])

        embedded_X.append(tmp_X)

    return embedded_X



# Embed each word as a unique integer

X_train = embed_word_to_int(df_train['processed_questions'].values, word_to_idx)

X_test = embed_word_to_int(df_test['processed_questions'].values, word_to_idx)



pad_length = hparam['PAD_LENGTH']



# Pad the questions to the same length

X_train_pad = pad_sequences(X_train, maxlen=pad_length, padding='pre', truncating='pre')

X_test_pad = pad_sequences(X_test, maxlen=pad_length, padding='pre', truncating='pre')



#print("y_train.shape: {}".format(y_train.shape))

#print("X_test.shape: {}".format(X_test.shape))



print(df_train['processed_questions'][3333])

print(X_train[3333])

print(X_train_pad[3333])



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

    x = Embedding(VOCAB_SIZE, WORD_EMB_DIM, weights=[embedding_matrix], trainable=False)(inp)

    # x = Bidirectional(CuDNNLSTM(LSTM_HIDDEN_SIZE, return_sequences=True))(x)

    x = CuDNNLSTM(LSTM_HIDDEN_SIZE, return_sequences=True)(x)

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



# We use StratifiedKFold for iterating over the k-folds

kfold = StratifiedKFold(n_splits=hparam['KFOLDS'], shuffle=True, random_state=2019)



timeSince(t0)
t0 = time.time()



X = X_train_pad

y = df_train['target'].values



results = [] # All results are saved in this list!



for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):

    X_train, X_val = X[train_index], X[val_index]

    y_train, y_val = y[train_index], y[val_index]

    

    print(f"Training on {len(X_train)} and validating on {len(X_val)} number of words")

    

    # Create dataset placeholder

    dataset = {'X_train': X_train, 'y_train': y_train,

          'X_val': X_val, 'y_val': y_val,

          'X_test': X_test_pad}

    

    # Run entire procedure and get predictions, best threshold and best f1 score

    # Here we use the concatenated embedding with a dimension of 900!

    test_preds, val_preds, thresh, f1 = train_val_pred(dataset, hparam, conc_embedding)

    

    print("len(test_preds) = {}, len(val_preds) = {}, thresh = {} at f1 = {}".format(len(test_preds), 

                                                                                     len(val_preds), 

                                                                                     thresh, 

                                                                                     f1))

    # Save into results

    new_result = {'name': 'fold-' + str(fold), 

                  'test_preds': test_preds, 

                  'val_preds': val_preds, 

                  'thresh': thresh, 

                  'f1': f1}

    results.append(new_result)

    

    # Memory management!

    import gc; gc.collect()

    

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
'''

t0 = time.time()



pred_val_y = None

factor = int(1.0 / len(results))



for result in results:

    pred_val_y += factor * result['val_preds']



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

'''
t0 = time.time()



print("Using treshold {}".format(avg_thresh))



factor = 1.0 / len(results)

pred_test_y = results[0]['test_preds'] * factor



print("Using factor: ", factor)



for i in range(1, len(results)):

    pred_test_y += factor * results[i]['test_preds']

    



pred_test_y_res = (pred_test_y > avg_thresh).astype(int)



results_dict = {'qid':df_test['qid'].values, 'prediction':[]}



for prediction in pred_test_y_res:

    results_dict['prediction'].append(prediction[0])

    

print(results_dict['qid'][:15])

print(results_dict['prediction'][:15])

    

# Save results

df = pd.DataFrame(data=results_dict)

df.to_csv('submission.csv', index=False)

print("Saved csv to disk!")



timeSince(t0)
'''

t0 = time.time()



def pad_per_batch(X, batch_size):

    X_pad = []

    print(f"Length is {len(X)} and using batch size {batch_size}")

    e = 0

    

    max_found = 0

    max_p_found = 0 # Keep stats for now

    

    for i in range(batch_size,len(X), batch_size):

        # start and end for this minibatch

        s = i-batch_size

        e = i

        batch = X[s:e]

        

        # calculate 98th percentile batch length

        a = np.array([len(x) for x in batch])

        p = int(np.percentile(a, 98)) # Get 98th percentile of all lengths on this batch!

        m = int(np.max(a))

        

        # track stats for now

        if m > max_found:

            max_found = m

        if p > max_p_found:

            max_p_found = p

        

        padded_batch = pad_sequences(batch, maxlen=p)

        for vec in padded_batch:

            X_pad.append(vec)

    

    # Get the last batch as well!

    last_batch = X[e:]

    a = np.array([len(x) for x in last_batch])

    p = int(np.percentile(a, 98)) # Get 98th percentile of all lengths on this batch!

    

    padded_batch = pad_sequences(last_batch, maxlen=p)

    for vec in padded_batch:

        X_pad.append(vec)

        

    print(f"max p = {max_p_found} and max = {max_found}")

    print(f"X_pad length = {len(X_pad)}, X length = {len(X)}")

    return np.asarray(X_pad), max_p_found

        

# Pad the questions per batch size

X_train_pad, max_pad_len1 = pad_per_batch(X_train_token, hparam['MINIBATCH_SIZE'])

X_val_pad, max_pad_len2 = pad_per_batch(X_val_token, hparam['MINIBATCH_SIZE'])

X_test_pad, max_pad_len3 = pad_per_batch(X_test_token, hparam['MINIBATCH_SIZE'])



max_pad_len = max(max_pad_len1, max(max_pad_len2, max_pad_len3))

hparam['PAD_LENGTH'] = max_pad_len



# pack into placeholder

dataset = {'X_train': X_train_pad, 'y_train': y_train,

          'X_val': X_val_pad, 'y_val': y_val,

          'X_test': X_test_pad}



print("X_train_pad.shape: {}".format(X_train_pad.shape))

print("X_train_pad[3333].shape: {}".format(X_train_pad[3333].shape))

print("y_train.shape: {}".format(y_train.shape))

print("X_val_pad.shape: {}".format(X_val_pad.shape))

print("y_val.shape: {}".format(y_val.shape))

print("X_test_pad.shape: {}".format(X_test_pad.shape))



print(X_train[3333])

print(X_train_token[3333])

print(X_train_pad[3333])

print(len(tokenizer.word_counts))

print(len(tokenizer.word_index))



timeSince(t0)

'''