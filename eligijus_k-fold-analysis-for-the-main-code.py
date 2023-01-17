from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.util import ngrams
import pandas as pd
import string
import re
from gensim.models import KeyedVectors
import numpy as np
from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, LSTM, Embedding, GlobalMaxPool1D, Conv1D, MaxPooling1D 
from keras.layers import CuDNNLSTM, Bidirectional, CuDNNGRU, GlobalAvgPool1D, concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Optimizer
from keras.callbacks import *
from keras import backend as K
from keras.engine.topology import Layer
import gc
import Levenshtein 
from tqdm import tqdm
import operator

## Reading input

d = pd.read_csv('../input/train.csv')
Y_train = d['target']
X_train = d['question_text'] 

## Reading the test set 

d_test = pd.read_csv('../input/test.csv')
X_test = d_test['question_text']
## Preprocesing functions

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

## Functions for embeddings

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
    Creates an embeding matrix
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

def tokenize_text(string_vec, tokenizer, max_len):
    """
    Tokenizes a given string vector
    """
    token = tokenizer.texts_to_sequences(string_vec)
    token = pad_sequences(token, maxlen = max_len)
    
    return token
    
### Functions for predictions

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

## Feature engineering

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
    
## Functions for deep learning
    
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
                '\u200b':" ", 
                "\ufeff" : "",
                'mastuburation' : 'masturbation',
                'quara' : 'Quora',
                'Quoras' : 'Quora',
                "fiancé" : "fiance", 
                'π' : 'pi', 
                'Pokémon' : 'Pokemon' }

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
## Main model 

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

## Hyperparameters

maxlen = 70 ## Number of words in each sentence to use
max_features = 120000 

batch_size = 512 ## batch size
numb_epoch = 2 ## number of epochs

## Size of the embedding
### If we used more than one embedding, then the number of coordinates for each word is doubled, tripled, etc.

embed_size = 300 * len(embeds_to_use)
use_k_fold = False
X_tr = preproc_pipeline(X_train, model_embed, punct, vocab_mapper = vocab_mapper)
X_te = preproc_pipeline(X_test, model_embed, punct, vocab_mapper = vocab_mapper)
## Reshuffling all the data to insure the same distribution in all folds

X_tr = X_tr.iloc[np.random.permutation(len(X_tr))]
Y_tr = Y_train.iloc[X_tr.index.values]

## Reseting the indexes

X_tr = X_tr.reset_index(drop = True)
Y_tr = Y_tr.reset_index(drop = True)

k_fold = 5
t_results = [] ## Object to store the optimal tresholds in

## Looping through all the folds

k_fold_index = np.array_split(X_tr.index.values, k_fold)
for k in range(k_fold):

    print('Fold: ' + str(k + 1))

    ## Creating the training and test sets

    test_index = k_fold_index[k]
    not_k = [x for x in set(range(k_fold)) - set([k])]
    train_index = np.concatenate([k_fold_index[x] for x in not_k])

    XX_tr = X_tr.iloc[train_index]
    YY_tr = Y_tr.iloc[train_index] 

    XX_te = X_tr.iloc[test_index]
    YY_te = Y_tr.iloc[test_index]

    ## Text tokenization

    tokenizer = Tokenizer(num_words = max_features)
    tokenizer.fit_on_texts(list(XX_tr))

    XX_tr = tokenize_text(XX_tr, tokenizer, maxlen)
    XX_te = tokenize_text(XX_te, tokenizer, maxlen)   

    ## Creating a weight matrix for words in training matrix

    embedding_matrix = create_embedding_matrix(model_embed, 
                                                tokenizer, 
                                                max_features, 
                                                embed_size)

    ## Creating the model 

    model = RNN_model(XX_tr.shape[1], embed_size, max_features, embedding_matrix)

    model_fited = model.fit(
    XX_tr,
    YY_tr.values, 
    batch_size = batch_size, 
    nb_epoch = numb_epoch)  

    ## Predictions

    y_hat_probs = model.predict(XX_te)

    ## Optimal treshold

    opti_t = optimal_treshold(YY_te.values, y_hat_probs)
    print(opti_t)
    t_results.append(opti_t['threshold'])

    ### Releasing memory

    del embedding_matrix, model, XX_tr, XX_te, tokenizer, model_fited
    gc.collect()
    time.sleep(5)

## Averaging the optimal tresholds

opti_t = np.mean(t_results)
print("Optimal treshold: " + str(opti_t))