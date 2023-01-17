## Loading necessary packages

## Basic packages:

import numpy as np 
import pandas as pd 
import string 
from tqdm import tqdm
import operator
import os
import gc
import time
import re

## K fold analysis

from random import shuffle

## Reading embeddings:

from gensim.models import KeyedVectors

## Working with text:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

## Accuracy measure

from sklearn.metrics import f1_score

## Deep learning: 

from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, LSTM, Embedding, GlobalMaxPool1D, Conv1D, MaxPooling1D 
from keras.layers import CuDNNLSTM, Bidirectional, CuDNNGRU, GlobalAvgPool1D, concatenate
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer

## Levenshtein distance

import Levenshtein 
## Reading input

d = pd.read_csv('../input/train.csv')
Y_train = d['target']
X_train = d['question_text'] 

## Reading the test set 

d_test = pd.read_csv('../input/test.csv')
X_test = d_test['question_text']  
def encode_digit(x):
    """
    Encodes a digit in a string
    """
    x = re.sub('[0-9]{5,}', ' ##### ', x)
    x = re.sub('[0-9]{4}', ' #### ', x)
    x = re.sub('[0-9]{3}', ' ### ', x)
    x = re.sub('[0-9]{2}', ' ## ', x)
    return x

def clean_digits(string_vec):
    """
    Removes digits from a string vector
    """
    cleaned_string = [encode_digit(s) for s in string_vec]
    
    return pd.Series(cleaned_string)

def clean_ws(string_vec):
    """
    Cleans whitespaces
    """
    cleaned_string = [re.sub( '\s+', ' ', s).strip() for s in string_vec]
    return pd.Series(cleaned_string)

def clean_word(char, punct):
    """
    A function that removes bad punctuations and splits good ones in a given string
    """
    for p in punct:
        char = char.replace(p, f' {p} ')
    
    return(char)

def clean_punct(string_vec, punct):
    """
    Function that cleans the punctuations
    """
    cleaned_string = []
    for char in tqdm(string_vec):
        char = [clean_word(x, punct) for x in char.split()]
        cleaned_string.append(' '.join(char))
    return pd.Series(cleaned_string)   

def tokenize_text(string_vec, tokenizer, max_len):
    """
    Tokenizes a given string vector
    """
    token = tokenizer.texts_to_sequences(string_vec)
    token = pad_sequences(token, maxlen = max_len)
    
    return token
def load_from_text(path):
    """
    A functions that load embeddings from a txt document
    """
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(path, errors='ignore'))
    return embeddings_index

def read_embedding(path, reading_type, binary = False):
    """
    Reads the embeddings from a .txt or .vec file
    """
    if(reading_type == 'text'):
        model_embed = load_from_text(path)
        
    if(reading_type == 'word2vec'):
        model_embed = KeyedVectors.load_word2vec_format(path, binary = binary)
    
    return model_embed  

def create_embedding_matrix(model_embed, tokenizer, max_features, embed_size):
    """
    Creates the embeding matrix
    """
    embedding_matrix = np.zeros((max_features, embed_size))
    for word, index in tokenizer.word_index.items():
        if index > max_features - 1:
            break
        else:
            try:
                embedding_matrix[index] = model_embed[word]
            except:
                continue
    return embedding_matrix        

def find_most_similar(words, model_embed, window = 1, return_top = 1):
    """
    Finds the most similar words in the given embedding dictionary based on Levenshtein's distance
    """
    all_keys = list(model_embed.keys())
    key_dict = {}
    
    for key in all_keys:
        key_dict.update({key: len(key)})
    
    vocab_mapper = []    
    for word in tqdm(words):
        w_l = len(word)
        sub_key_dict = [k for k, v in key_dict.items() if w_l - window <= v <= w_l + window] # Subseting the search plane
        dist_list = [{word : key, 'dist': Levenshtein.distance(word, key)} for key in sub_key_dict]
        dist_list = sorted(dist_list, key = lambda k: k['dist'])[:return_top] ## Extracting the top matches 
        vocab_mapper.append(dist_list)
    
    ## Creating a pandas dataframe for easier exploration
    
    vocab = pd.DataFrame(columns = ['orig_word', 'suggestion', 'dist'])
    
    for entry in vocab_mapper:
        for j in range(return_top):
            keys = [*entry[j].keys()]
            suggestion = entry[j][keys[0]]
            dist = entry[j][keys[1]]
            vocab = vocab.append({'orig_word' : keys[0], 'suggestion' : suggestion, 'dist' : dist}, ignore_index = True)
    
    vocab = vocab.sort_values('orig_word')
    return vocab

def build_vocab(sentences, verbose =  True):
    """
    A function that creates a vocabulary from the text
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence.split():
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, model_embed):
    """
    A function to count the words that are missing from the embeddings
    """
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = model_embed[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def spell_checker(string_vec, vocab_mapper):
    """
    A function to change the word in a dictionary "x" : "y"
    in the following manner: x -> y
    """
    cleaned_strings = []
    for char in string_vec:
        cleaned_words = []
        for x in char.split():
            if vocab_mapper.get(x) is not None:
                x = vocab_mapper.get(x)
            cleaned_words.append(x)
        cleaned_words = ' '.join(cleaned_words)
        cleaned_strings.append(cleaned_words)
    
    return pd.Series(cleaned_strings)    

def to_binary(p_array, treshold):
    """
    Converts the prediction from probability to 0 or 1
    """
    y_hat = []
    for i in range(len(p_array)):
        if p_array[i] > treshold:
            y_hat.append(1)
        else:
            y_hat.append(0)
    return y_hat    

def optimal_treshold(y, yhat):
    """
    Computes the otpimal treshold for the f1 statistic
    """
    best_threshold = 0
    best_score = 0
    for threshold in [i/100 for i in range(10,90)]:
        score = f1_score(y, yhat > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result

def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def share_unique(string_vec):
    """
    A function that calculates the share of unique words in a given string
    """
    share_list = []
    for char in string_vec:
        sh = len(set(char.split()))/len(char.split())
        share_list.append(sh)
    return share_list
    
def share_capslock(string_vec):
    """
    Calculates the share of caps locked words in a given string
    """
    share_list = []
    for char in string_vec:
        to_upper = char.upper().split()
        sh = len(set(char.split()).intersection(to_upper))/len(char.split())
        share_list.append(sh)
    return share_list    

def ends_with_symbol(string_vec, symbol):
    """
    Returns a list of 1 and 0 indicating whether a string ended with a symbol or not
    """
    return [int(x.endswith(symbol)) for x in string_vec]

def count_words(string_vec):
    """
    Counts the number of words in a given string
    """
    return [len(x.split()) for x in string_vec]
    
def count_occurance(string_vec):
    """
    Counts the number of ? and ! in a given string
    """
    return [x.count('!') + x.count('?') for x in string_vec]
print(os.listdir("../input/embeddings/"))
embed_dict = {'google': {'path' : '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
                         'reading' : 'word2vec', 
                         'binary': True}, 
                         
               'glove': {'path': '../input/embeddings/glove.840B.300d/glove.840B.300d.txt', 
                         'reading' : 'text',
                         'binary': False}, 
               
                'wiki': {'path' : '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec', 
                         'reading' : 'text', 
                         'binary': False}, 
                         
               'paragram': {'path': '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt',
                         'reading' : 'text', 
                          'binary' : False}
}
embeds_to_use = ['glove'] ## name of key in the 'embed_dict' dictionary
index = 0
for embed in embeds_to_use:
    
    print('Reading: ' + embed)
    path = embed_dict.get(embed)['path']
    type_of_file = embed_dict.get(embed)['reading']
    bin = embed_dict.get(embed)['binary']
    embedding = read_embedding(path, type_of_file, bin)

    if embed == embeds_to_use[0]:
        model_embed = embedding
    else: 
        all_keys = [k for k in embedding.keys()]
        index += 1
        for key in tqdm(all_keys):
            try:
                vect = np.concatenate([model_embed[key], embedding[key]])
            except:
                vect = np.concatenate([np.zeros((1, 300 * index), dtype = 'float32')[0], embedding[key]])
                
            model_embed.update({key : vect})    
            del vect
        del all_keys    
            
    del embedding
    gc.collect()
    time.sleep(5)    
## Word fixing:

vocab_mapper = {'Quorans' : 'Qurans', 
                'Blockchain' : 'blockchain', 
                'blockchains' : 'blockchain',
                'demonetisation' : 'demonetization', 
                'ethereum' : 'Ethereum', 
                'Qoura' : 'Quora', 
                'SJWs' : 'SJW', 
                'bhakts' : 'bhakti', 
                'Bhakts' : 'Bhakti', 
                'kotlin' : 'Kotlin', 
                'narcissit' : 'narcissist', 
                'Trumpism' : 'Trump', 
                'Tamilans' : 'Tamilians', 
                'acturial' : 'actuarial', 
                'demonitization' : 'demonetization', 
                'Demonetization' : 'demonetization',
                'demonitisation' : 'demonetization',
                'Demonetisation' : 'demonetization',
                'Whyis' : 'Why is', 
                'AirPods' : 'AirPod', 
                'Drumpf': 'Trumpf', 
                'Zhihu' : 'Zhihua', 
                'Neuralink' : 'Neurolink', 
                'fullform' : 'full-form', 
                'biharis' : 'Biharis', 
                'madheshi' : 'Madheshi', 
                'Xiomi' : 'Xiaomi', 
                'rohingya' : 'Rohingya', 
                'Despacito' : 'Desposito', 
                'schizoids' : 'schizoid', 
                'MHTCET' : 'MHT-CET', 
                'fortnite' : 'Fortnite',
                'Bittrex' : 'Bitrex', 
                'ReactJS' : 'JavaScript', 
                'hyperloop' : 'Hyperloop', 
                'adhaar' : 'Aadhaar', 
                'Adhaar' : 'Aadhaar', 
                'Baahubali' : 'Bahubali', 
                'Cryptocurrency' : 'cryptocurrency', 
                'cryptocurrencies' : 'cryptocurrency',
                'cryptocoins' : 'cryptocurrency',    
                "\u200b":" ",
                "\ufeff" : "",
                "2k17" : '2017',
                "2k18" : '2018',
                "nofap": 'no fap', 
                'Brexiting' : 'Brexit',
                'mastuburation' : 'masturbation',
                'quara' : 'Quora',
                'Quoras' : 'Quora',
                "fiancé" : "fiance", 
                'π' : 'pi', 
                'Pokémon' : 'Pokemon', 
                '€' : 'euro'
}

## Expanding phrases

word_expansion = {"aren't" : 'are not', 
                "I'm" : 'I am',
                "What's" : 'What is',
                "don’t" : "do not", 
                "isn't" : "is not", 
                "I’m" : "I am", 
                'aren’t' : 'are not', 
                "Can't" : "cannot", 
                "can't" : "cannot",
                "don't" : 'do not', 
                "How's" : "how is", 
                "we're" : 'we are', 
                "won't" : 'will not', 
                "they're" : "they are",
                "he's" : "he is", 
                "doesn’t" : "does not", 
                "shouldn't" : "should not", 
                "Shouldn't" : "should not",
                "hasn't" : "has not", 
                "couldn't" : "could not", 
                "I’ve" : "I have", 
                "aren't" : "are not", 
                "weren't" : 'were not'}

## Punctuations to extract

punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
punct = ''.join(punct)

## Defining the preprocesing

def preproc_pipeline(string_df, model_embed, punct, word_expansion = None, vocab_mapper = None):
    """
    The whole pipeline of cleaning
    """
    
    if(word_expansion is not None):
        string_df = spell_checker(string_df, word_expansion)
        
    string_df = clean_punct(string_df, punct)
    
    if(vocab_mapper is not None):
        string_df = spell_checker(string_df, vocab_mapper)
    
    string_df = clean_digits(string_df)
    string_df = clean_ws(string_df)
    return string_df

## Applying the preprocesing

X_tr = preproc_pipeline(X_train, model_embed, punct, vocab_mapper = vocab_mapper)
X_te = preproc_pipeline(X_test, model_embed, punct, vocab_mapper = vocab_mapper)
## The whole vocabulary
vocab = build_vocab(X_tr)

## A dictionary for words that are out of the vocabulary but in the embeddings
oov = check_coverage(vocab, model_embed)

## Subsetting only relevant words
oov = [x[0] for x in oov if x[1] > 60]

## Findint the top 5 synonyms to mannualy add them to either
## word_expansion or vocab_mapper dictionaries
synonyms = find_most_similar(oov, model_embed, return_top = 3)
print(synonyms)
maxlen = 70 ## Number of words in each sentence to use
max_features = 120000 

batch_size = 512 ## batch size
numb_epoch = 2 ## number of epochs

## Size of the embedding
### If we used more than one embedding, then the number of coordinates for each word is doubled, tripled, etc.

embed_size = 300 * len(embeds_to_use)
def RNN_model(maxlen, embed_size, max_features, embedding_matrix,
              loss_f = 'binary_crossentropy', opti = 'adam', metr = f1):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences = True))(x)
    x = Bidirectional(CuDNNLSTM(100, return_sequences = True))(x)
    x = Bidirectional(CuDNNLSTM(72, return_sequences = True))(x)
    
    atten = Attention(maxlen)(x)
    avg_pool = GlobalAvgPool1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    
    conc = concatenate([atten, avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    outp = Dense(1, activation="sigmoid")(conc)   

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss = loss_f, optimizer = opti , metrics=[metr])
    return model
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(list(X_tr))

X_tr = tokenize_text(X_tr, tokenizer, maxlen)
X_te = tokenize_text(X_te, tokenizer, maxlen)   

## Creating a weight matrix for words in training matrix

embedding_matrix = create_embedding_matrix(model_embed, 
                                            tokenizer, 
                                            max_features, 
                                            embed_size)

## Creating the model 

model = RNN_model(X_tr.shape[1], embed_size, max_features, embedding_matrix)

model_fited = model.fit(
X_tr,
Y_train.values, 
batch_size = batch_size, 
nb_epoch = numb_epoch)  

## Predictions

y_hat_probs = model.predict(X_te)
    
## To binary

y_hat = to_binary(y_hat_probs, 0.38)
    
## Creating the upload file

print('The submission file has ' + str(np.sum(y_hat) * 100/len(y_hat)) + ' percent insincere')
d_test = d_test.reset_index()
d_test['prediction'] = y_hat
to_upload = d_test[['qid', 'prediction']]
to_upload.to_csv('submission.csv', index = False)