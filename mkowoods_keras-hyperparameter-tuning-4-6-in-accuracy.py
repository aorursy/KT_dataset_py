import os

import re



import numpy as np

import tensorflow as tf



np.random.seed(1)

tf.set_random_seed(2)



import pandas as pd

import keras

# from tqdm import tqdm

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import class_weight

from sklearn.metrics import f1_score, classification_report, log_loss



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical



from keras.models import Sequential

from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Flatten

from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping



# https://www.kaggle.com/inspector/keras-hyperopt-example-sketch



#training constants

MAX_SEQ_LEN = 25 #this is based on a quick analysis of the len of sequences train['text'].apply(lambda x : len(x.split(' '))).quantile(0.95)

DEFAULT_BATCH_SIZE = 128

print(os.listdir('../input'))
data = pd.read_csv('../input/Sentiment.csv')

train, test = train_test_split(data, random_state = 42, test_size=0.1)

print(train.shape)

print(test.shape)
# Mapping of common contractions, could probbaly be done better

CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",

                       "It's": 'It is', "Can't": 'Can not',

                      }



def clean_text(text, mapping):

    replace_white_space = ["\n"]

    for s in replace_white_space:

        text = text.replace(s, " ")

    replace_punctuation = ["’", "‘", "´", "`", "\'", r"\'"]

    for s in replace_punctuation:

        text = text.replace(s, "'")

    

    # Random note: removing the URL's slightly degraded performance, it's possible the model learned that certain URLs were positive/negative

    # And was able to extrapolate that to retweets. Could also explain why re-training the Embeddings improves performance.

    # remove twitter url's

#     text = re.sub(r"http[s]?://t.co/[A-Za-z0-9]*","TWITTERURL",text)

    mapped_string = []

    for t in text.split(" "):

        if t in mapping:

            mapped_string.append(mapping[t])

        elif t.lower() in mapping:

            mapped_string.append(mapping[t.lower()])

        else:

            mapped_string.append(t)

    return ' '.join(mapped_string)
# Get tweets from Data frame and convert to list of "texts" scrubbing based on clean_text function

# CONTRACTION_MAPPING is a map of common contractions(e.g don't => do not)

train_text_vec = [clean_text(text, CONTRACTION_MAPPING) for text in train['text'].values]

test_text_vec = [clean_text(text, CONTRACTION_MAPPING) for text in test['text'].values]





# tokenize the sentences

tokenizer = Tokenizer(lower=False)

tokenizer.fit_on_texts(train_text_vec)

train_text_vec = tokenizer.texts_to_sequences(train_text_vec)

test_text_vec = tokenizer.texts_to_sequences(test_text_vec)



# pad the sequences

train_text_vec = pad_sequences(train_text_vec, maxlen=MAX_SEQ_LEN)

test_text_vec = pad_sequences(test_text_vec, maxlen=MAX_SEQ_LEN)



print('Number of Tokens:', len(tokenizer.word_index))

print("Max Token Index:", train_text_vec.max(), "\n")



print('Sample Tweet Before Processing:', train["text"].values[0])

print('Sample Tweet After Processing:', tokenizer.sequences_to_texts([train_text_vec[0]]), '\n')



print('What the model will interpret:', train_text_vec[0].tolist())
# One Hot Encode Y values:

encoder = LabelEncoder()



y_train = encoder.fit_transform(train['sentiment'].values)

y_train = to_categorical(y_train) 



y_test = encoder.fit_transform(test['sentiment'].values)

y_test = to_categorical(y_test) 
# get an idea of the distribution of the text values

from collections import Counter

ctr = Counter(train['sentiment'].values)

print('Distribution of Classes:', ctr)



# get class weights for the training data, this will be used data

y_train_int = np.argmax(y_train,axis=1)

cws = class_weight.compute_class_weight('balanced', np.unique(y_train_int), y_train_int)

print(cws)
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval



DROPOUT_CHOICES = np.arange(0.0, 1.0, 0.1)

UNIT_CHOICES = np.arange(8, 129, 8, dtype=int)

FILTER_CHOICES = list(range(1, 9, 1))

EMBED_UNITS = np.arange(32, 513, 32, dtype=int)

space = {

    

    'spatial_dropout': hp.choice('spatial_dropout', DROPOUT_CHOICES),

    'embed_units': hp.choice('embed_units', EMBED_UNITS),

    'conv1_units':  hp.choice('conv1_units', UNIT_CHOICES),

    'conv1_filters': hp.choice('conv1_filters', FILTER_CHOICES),

    #nesting the layers ensures they're only un-rolled sequentially

    'conv2': hp.choice('conv2', [False, {

        'conv2_units':  hp.choice('conv2_units', UNIT_CHOICES),

        'conv2_filters': hp.choice('conv2_filters', FILTER_CHOICES),

        #only make the 3rd layer availabile if the 2nd one is

        'conv3': hp.choice('conv3', [False, {

            'conv3_units':  hp.choice('conv3_units', UNIT_CHOICES),

            'conv3_filters': hp.choice('conv3_filters', FILTER_CHOICES),

        }]),

    }]),

    'dense_units':  hp.choice('dense_units', UNIT_CHOICES),

    'batch_size':  hp.choice('batch_size', UNIT_CHOICES),

    'dropout1':  hp.choice('dropout1', DROPOUT_CHOICES),

    'dropout2':  hp.choice('dropout2', DROPOUT_CHOICES)

}
X_train = train_text_vec

X_test = test_text_vec



def objective(params, verbose=0, checkpoint_path = 'model.hdf5'):

    

    if verbose > 0:

        print ('Params testing: ', params)

        print ('\n ')

    

    model = Sequential()

    model.add(Embedding(input_dim = (len(tokenizer.word_counts) + 1), output_dim = params['embed_units'], input_length = MAX_SEQ_LEN))

    model.add(SpatialDropout1D(params['spatial_dropout']))

    model.add(Conv1D(params['conv1_units'], params['conv1_filters']))

    

    #layers are hyperparameters and can be excluded/included dynamically(which is fun)

    if params['conv2']:

        model.add(Conv1D(params['conv2']['conv2_units'], params['conv2']['conv2_filters']))

        

    if params['conv2'] and params['conv2']['conv3']:

        model.add(Conv1D(params['conv2']['conv3']['conv3_units'], params['conv2']['conv3']['conv3_filters']))

    

    model.add(GlobalMaxPool1D())

    model.add(Dropout(params['dropout1']))

    model.add(Dense(params['dense_units'], activation='relu'))

    model.add(Dropout(params['dropout2']))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    model.fit(

        X_train, 

        y_train, 

        validation_data=(X_test, y_test),  

        epochs=8,  #usually train the model for best accuracy, but when dropout is really low, the time to convergence can be excessive

        batch_size=params['batch_size'],

        class_weight=cws,

         #saves the most accurate model, usually you would save the one with the lowest loss

        callbacks= [

            ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=verbose, save_best_only=True),

            EarlyStopping(patience = 2, verbose=verbose,  monitor='val_acc')

        ],

        verbose=verbose

    ) 

    

    model.load_weights(checkpoint_path)

    predictions = model.predict(X_test, verbose=verbose)

    acc = (predictions.argmax(axis = 1) == y_test.argmax(axis = 1)).mean()

    return {'loss': -acc, 'status': STATUS_OK}    
#hidden to clear out the tensorflow warnings



objective({

    'spatial_dropout': 0.25,

    'embed_units': 128,

    'conv1_units':  64,

    'conv1_filters': 4,

    'conv2': {

        'conv2_units': 32,

        'conv2_filters': 4,

        'conv3': {

            'conv3_units': 16,

            'conv3_filters': 4        

        },

    },

    'dense_units':  64,

    'batch_size':  64,

    'dropout1': 0.0,

    'dropout2': 0.0,

}, verbose=2)
objective({

    'spatial_dropout': 0.25,

    'embed_units': 128,

    'conv1_units':  64,

    'conv1_filters': 4,

    'conv2': {

        'conv2_units': 32,

        'conv2_filters': 4,

        'conv3': {

            'conv3_units': 16,

            'conv3_filters': 4        

        },

    },

    'dense_units':  64,

    'batch_size':  64,

    'dropout1': 0.0,

    'dropout2': 0.0,

}, verbose=2)
trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100, rstate=np.random.RandomState(99))
space_eval(space, best)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt



y = np.array([ -t['loss'] for t in trials.results])

x = np.arange(1, len(y) + 1, 1)



y_max = np.max(y)

best_y = y[y == y_max]

best_y_xs = x[y == y_max]



#just to calculate a locally weighted average

reg = KNeighborsRegressor()

reg.fit(x.reshape(-1, 1), y)

preds = reg.predict(x.reshape(-1, 1))



plt.plot(x, y, 'go', alpha=0.5)

plt.plot(best_y_xs, best_y, 'ro')

plt.plot(x, preds, '--')



plt.ylabel('Accuracy')

plt.xlabel('Iteration')

plt.title('Accuracy Per Step')

plt.show()
def extract_params(trials):

    n_trials = len(trials.trials)

    keys = trials.vals.keys()

    data = []

    conv2_idx = -1

    conv3_idx = -1

    for i in range(n_trials):

        vals = {} 

        conv2_layer_active = (trials.vals['conv2'][i] == 1)

        if conv2_layer_active: conv2_idx += 1



        conv3_layer_active = (trials.vals['conv3'][conv2_idx] == 1)    

        if conv2_layer_active and conv3_layer_active: conv3_idx += 1



        for k in keys:

            if k in ('conv2_units', 'conv2_filters', 'conv3'):

                if conv2_layer_active:

                    vals[k] =  trials.vals[k][conv2_idx]  

            elif k in ('conv3_units', 'conv3_filters'):

                if conv3_layer_active:

                    vals[k] =  trials.vals[k][conv3_idx]  

            else:

                vals[k] = trials.vals[k][i]



        data.append(space_eval(space, vals))



    for idx, data_dict in enumerate(data):

        data_dict['accuracy'] = -trials.trials[idx]['result']['loss']

    

    return data



def _flatten(data):

    new_data = {}

    for k in data:

        #there's a more elegant(recursive) way to code this, but not the focus of this project...

        if k == 'conv2':

            new_data['conv2'] = int(bool(data['conv2']))

            conv2_dict = data['conv2'] if data['conv2'] else {}

            for k_c2 in conv2_dict:

                if k_c2.startswith('conv2'):

                     new_data[k_c2] = conv2_dict[k_c2]

                elif k_c2 == 'conv3':

                    new_data['conv3'] = int(bool(conv2_dict['conv3']))

                    conv3_dict = conv2_dict['conv3'] if conv2_dict['conv3'] else {}

                    for k_c3 in conv3_dict:

                        if k_c3.startswith('conv3'):

                             new_data[k_c3] = conv2_dict['conv3'][k_c3]

        else:

            new_data[k] = data[k]

    return new_data

    

import pandas as pd

data = list(map(_flatten, extract_params(trials)))

df = pd.DataFrame(list(data))

df = df.fillna(0) #missing values occur when the object is not populated
corr = df.corr()

corr
matfig = plt.figure(figsize=(8,8))

plt.matshow(corr, fignum=matfig.number)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')

plt.yticks(range(len(corr.columns)), corr.columns)

plt.colorbar()

plt.show()
def get_mean_by_bin(x, y):

    return [y[x == x_bin].mean() for x_bin in x]



for col in df.columns:

#     reg = KNeighborsRegressor()

#     reg.fit(df[col].values.reshape(-1, 1), df.accuracy)

#     preds = reg.predict(np.sort(df[col].values).reshape(-1, 1))

    x_argsort = np.argsort(df[col].values)

    x_sorted = df[col].values[x_argsort]

    y_sorted = df.accuracy.values[x_argsort]

    

    y_max = np.max(y_sorted)

    best_y = y_sorted[y_sorted == y_max]

    best_y_xs = x_sorted[y_sorted == y_max]

    

    plt.plot(df[col], df.accuracy, 'go', alpha=0.5)

    plt.plot(x_sorted, get_mean_by_bin(x_sorted, y_sorted) , '--')

    plt.plot(best_y_xs, best_y, 'ro')

    plt.xlabel(col)

    plt.ylabel('Accuracy')

    plt.title('%s vs Accuracy' % (col, ))

    plt.show()