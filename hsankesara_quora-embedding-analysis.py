import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout, CuDNNLSTM
from keras import backend as K
from keras import optimizers
from keras.models import Model
import nltk
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from nltk import tokenize,word_tokenize
import gc
from tqdm import tqdm
tqdm.pandas()
gc.collect()
import seaborn as sns
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
max_features= 200000
max_senten_len = 40
max_senten_num = 3
embed_size = 300
VALIDATION_SPLIT = 0.1
from sklearn.utils import shuffle
df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv("../input/test.csv")
df.head()
len(df.target.unique())
df.head()
df.columns = ['qid', 'text', 'category']
test_df.columns = ['qid', 'text']
df.head()
df = df[['text', 'category']]
df.info()
df['text'] = df['text'].apply(lambda x: x.lower())
test_df['text'] = test_df['text'].apply(lambda x: x.lower())
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'u.s':'america', 'e.g':'for example'}
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
df['text'] = df['text'].apply(lambda x: clean_contractions(x, contraction_mapping))
test_df['text'] = test_df['text'].apply(lambda x: clean_contractions(x, contraction_mapping))
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '!':' '}
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text
df['text'] = df['text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
test_df['text'] = test_df['text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x
df['text'] = df['text'].apply(lambda x: correct_spelling(x, mispell_dict))
test_df['text'] = test_df['text'].apply(lambda x: correct_spelling(x, mispell_dict))
labels = df['category']
text = df['text']
indices = np.arange(text.shape[0])
np.random.shuffle(indices)
text = text[indices]
labels = labels.iloc[indices]
nb_validation_samples = int(VALIDATION_SPLIT * df.shape[0])

train_text = text[:-nb_validation_samples].reset_index().drop('index', axis=1)
y_train = labels[:-nb_validation_samples].reset_index().drop('index', axis=1)
val_text = text[-nb_validation_samples:].reset_index().drop('index', axis=1)
y_val = labels[-nb_validation_samples:].reset_index().drop('index', axis=1)
test = test_df['text']
cates = df.groupby('category')
print("total categories:", cates.ngroups)
print(cates.size())
paras = []
labels = []
texts = []
sent_lens = []
sent_nums = []
for idx in range(train_text.shape[0]):
    text = train_text.text[idx]
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    sent_nums.append(len(sentences))
    for sent in sentences:
        sent_lens.append(len(text_to_word_sequence(sent)))
    paras.append(sentences)
f, ax = plt.subplots(figsize=(12, 6))
sns.distplot(sent_lens, ax=ax)
plt.show()
sns.distplot(sent_nums)
plt.show()
val_paras = []
val_labels = []
for idx in range(val_text.shape[0]):
    text = val_text.text[idx]
    sentences = tokenize.sent_tokenize(text)
    val_paras.append(sentences)
test_paras = []
test_labels = []
for idx in range(test.shape[0]):
    text = test[idx]
    sentences = tokenize.sent_tokenize(text)
    test_paras.append(sentences)
tokenizer = Tokenizer(num_words=max_features, oov_token=True)
tokenizer.fit_on_texts(texts)
x_train = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
for i, sentences in enumerate(paras):
        tokenized_sent = tokenizer.texts_to_sequences(sentences)
        padded_seq = pad_sequences(tokenized_sent, maxlen=max_senten_len, padding='post', truncating='post')
        for j, seq in enumerate(padded_seq):
            if(j < max_senten_num):
                x_train[i,j,:] = seq
            else:
                break
x_train.shape
x_val = np.zeros((val_text.shape[0], max_senten_num, max_senten_len), dtype='int32')
for i, sentences in enumerate(val_paras):
        tokenized_sent = tokenizer.texts_to_sequences(sentences)
        padded_seq = pad_sequences(tokenized_sent, maxlen=max_senten_len, padding='post', truncating='post')
        for j, seq in enumerate(padded_seq):
            if(j < max_senten_num):
                x_val[i,j,:] = seq
            else:
                break
test_data = np.zeros((test.shape[0], max_senten_num, max_senten_len), dtype='int32')
for i, sentences in enumerate(test_paras):
        tokenized_sent = tokenizer.texts_to_sequences(sentences)
        padded_seq = pad_sequences(tokenized_sent, maxlen=max_senten_len, padding='post', truncating='post')
        for j, seq in enumerate(padded_seq):
            if(j < max_senten_num):
                test_data[i,j,:] = seq
            else:
                break
print(test_data.shape, x_val.shape)
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
print('Shape of training tensor:', x_train.shape)
print('Shape of validation tensor:', x_val.shape)
print('Shape of test tensor:', test_data.shape)
import os
gc.collect()
word_index = tokenizer.word_index
max_features = len(word_index)+1
def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word.lower(), np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 
    
def load_fasttext(word_index):    
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100 and o.split(" ")[0] in word_index )

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix

def load_para(word_index):
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word.lower(), np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
Embedding_funs = [load_glove,  load_fasttext, load_para]
import itertools
REG_PARAM = 1e-2
l2_reg = regularizers.l2(REG_PARAM)
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
# https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = metrics.f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
def get_model(embedding_matrix):
    embedding_layer = Embedding(nb_words, embed_size, weights=[embedding_matrix])
    word_input = Input(shape=(max_senten_len,), dtype='float32')
    word_sequences = embedding_layer(word_input)
    word_lstm = Bidirectional(CuDNNLSTM(max_senten_len, return_sequences=True, recurrent_regularizer=l2_reg))(word_sequences)
    word_dense = TimeDistributed(Dense(64))(word_lstm)
    word_att = AttentionWithContext()(word_dense)
    wordEncoder = Model(word_input, word_att)

    sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
    sent_encoder = TimeDistributed(wordEncoder)(sent_input)
    sent_lstm = Bidirectional(CuDNNLSTM(max_senten_num, return_sequences=True, recurrent_regularizer=l2_reg))(sent_encoder)
    sent_dense = TimeDistributed(Dense(32))(sent_lstm)
    sent_att = AttentionWithContext()(sent_dense)
    preds = Dense(1, activation='sigmoid', kernel_regularizer=l2_reg)(sent_att)
    model = Model(sent_input, preds)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1])
    return model
# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def train_pred(model, train_X, train_y, val_X, val_y, epochs=2, callback=None):
    h = model.fit(train_X, train_y, batch_size=512, epochs=epochs, validation_data=(val_X, val_y), callbacks = callback, verbose=1)
    model.load_weights(filepath)
    pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test_data], batch_size=1024, verbose=0)
    print(metrics.f1_score(val_y, (pred_val_y > 0.33).astype(int)))
    print('=' * 60)
    return pred_val_y, pred_test_y
filepath="weights_best.h5"
validation_results = np.zeros((len(Embedding_funs), x_val.shape[0]))
test_results = np.zeros((len(Embedding_funs), test_data.shape[0]))
for indx, fun in enumerate(Embedding_funs):
    checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint]
    embedding_matrix = fun(word_index)
    nb_words = embedding_matrix.shape[0]
    model = get_model(embedding_matrix)
    pred_val_y , pred_test_y= train_pred(model, x_train, y_train, x_val, y_val, epochs = 3, callback = callbacks)
    validation_results[indx] = pred_val_y.reshape(-1)   
    test_results[indx] = pred_test_y.reshape(-1)
    os.remove(filepath)
    del model
    gc.collect()
def check_all_validations(validation_results, val_y, total=3):
    all_combs_f1 = {}
    all_combs_thres = {}
    for i in range(total):
        combinations = list(itertools.combinations(range(total), i+1))
        for indexes in combinations:
            val_res = np.mean(validation_results[list(indexes)], axis=0)
            search_result = threshold_search(val_y, val_res)
            all_combs_f1[indexes] = search_result['f1']
            all_combs_thres[indexes] = search_result['threshold']
    return all_combs_f1, all_combs_thres
all_combinations_f1, all_combinations_thresh = check_all_validations(validation_results, y_val)
for i in all_combinations_f1:
    print(i, ':', all_combinations_f1[i], 'with threshold equals to', all_combinations_thresh[i])
import operator
all_comb_sorted = sorted(all_combinations_f1.items(), key=operator.itemgetter(1), reverse=True)
pred_test_y = np.mean(test_results[list(all_comb_sorted[0][0])], axis=0)
best_thresh = all_combinations_thresh[all_comb_sorted[0][0]]
pred_test_y = (pred_test_y>best_thresh).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)


