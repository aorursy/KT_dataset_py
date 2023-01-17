import pandas as pd
import numpy as np
import operator 
import re
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.layers import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Model
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.initializers import glorot_normal,orthogonal

import gc
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")
train_y = train['target'].values
# test = pd.read_csv("../input/test.csv")
# preprocessing: https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

train['question_text'] = train['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_special_chars(text, punct, mapping):
    for p in mapping:
        if p in text:
            text = text.replace(p, mapping[p])
    
    for p in punct:
        if p in text:
            text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        if s in text:
            text = text.replace(s, specials[s])
    
    return text

train['question_text'] = train['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def correct_spelling(x, dic):
    for word in dic.keys():
        if word in x:
            x = x.replace(word, dic[word])
    return x

train['question_text'] = train['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))
# clean LaTeX tags: https://www.kaggle.com/sunnymarkliu/latex-cannot-be-used-in-a-quora-question
def clean_latex_tag(text):
    corr_t = []
    for t in text.split(" "):
        t = t.strip()
        if t != '':
            corr_t.append(t)
    text = ' '.join(corr_t)
    
    text = re.sub('(\[ math \]).+(\[ / math \])', 'mathematical formula', text)
    return text

train['question_text'] = train['question_text'].map(clean_latex_tag)
train_text, valid_text, y_train, y_valid = train_test_split(train['question_text'], train_y, test_size=0.1, random_state=2018)
EMBED_SIZE = 300
MAX_WORDS_LEN = 70
MAX_VOCAB_FEATURES = 200000
%%time
print('tokenize and padding')
tokenizer = Tokenizer(num_words=MAX_VOCAB_FEATURES, filters='')
tokenizer.fit_on_texts(train_text)

# tokenize
train_X = tokenizer.texts_to_sequences(train_text)
valid_X = tokenizer.texts_to_sequences(valid_text)

# Pad the sentences 
train_X = pad_sequences(train_X, maxlen=MAX_WORDS_LEN)
valid_X = pad_sequences(valid_X, maxlen=MAX_WORDS_LEN)

word_index = tokenizer.word_index
nb_words = min(MAX_VOCAB_FEATURES, len(word_index))

def load_glove():
    print("Extracting Glove embedding")
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    emb_mean, emb_std = -0.005838499, 0.48782197

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]

            if i >= nb_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector

    print('Glove:', embedding_matrix.shape)
    return embedding_matrix

glove_embedding_matrix = load_glove()
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
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings/notebook
def simple_model(embedding_matrix):
    inp = Input(shape=(MAX_WORDS_LEN,))
    x = Embedding(nb_words, EMBED_SIZE, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
    return model
model = simple_model(glove_embedding_matrix)
model.fit(train_X, y_train, batch_size=512, epochs=2, validation_data=(valid_X, y_valid))
import psutil
import multiprocessing
import markovify as mk

class MarkovifyTextAugmentor(object):
    
    def __init__(self, docs, gen_max_len, filter_min_len=10):
        """
        Build Markov models from large corpora of text, and generating random sentences from that.
        """
        self.mk_text_model = mk.Text(docs)
        self.gen_max_len = gen_max_len
        self.filter_min_len = filter_min_len
    
    def genenerate(self, count):
        texts = []
        
        for _ in range(count):
            text = self.mk_text_model.make_short_sentence(self.gen_max_len)
            if text and len(text.split(' ')) > self.filter_min_len:
                texts.append(text)
        
        return texts
insincere_docs = train_text[train['target'] == 1]
ins_doc_median_len = int(insincere_docs.str.len().median())
%%time
mk_augmentor = MarkovifyTextAugmentor(insincere_docs, ins_doc_median_len, filter_min_len=6)
augment_insincere_count = 2000
%%time
def gen_text_process(count):
    return mk_augmentor.genenerate(count)

num_cores = psutil.cpu_count()  # number of cores on your machine

pool = multiprocessing.Pool(processes=num_cores)
apply_results = []
process_counts = augment_insincere_count // num_cores
for _ in range(num_cores):
    apply_results.append(pool.apply_async(gen_text_process, (process_counts,)))

pool.close()
pool.join()

all_gen_texts = []
for res in apply_results:
    all_gen_texts.extend(res.get())
print('generate {} insincere questions'.format(len(all_gen_texts)))
print(all_gen_texts[:10])
%%time
aug_train_text = np.array(train_text.values.tolist() + all_gen_texts)
aug_y_train = np.array(y_train.tolist() + [1] * len(all_gen_texts))

# shuffle
shuf_idx = np.random.permutation(len(aug_train_text))
aug_train_text = aug_train_text[shuf_idx]
aug_y_train = aug_y_train[shuf_idx]
%%time
print('tokenize and padding')
tokenizer = Tokenizer(num_words=MAX_VOCAB_FEATURES, filters='')
tokenizer.fit_on_texts(aug_train_text)

# tokenize
aug_train_X = tokenizer.texts_to_sequences(aug_train_text)
valid_X = tokenizer.texts_to_sequences(valid_text)

# Pad the sentences 
aug_train_X = pad_sequences(aug_train_X, maxlen=MAX_WORDS_LEN)
valid_X = pad_sequences(valid_X, maxlen=MAX_WORDS_LEN)

word_index = tokenizer.word_index
nb_words = min(MAX_VOCAB_FEATURES, len(word_index))

def load_glove():
    print("Extracting Glove embedding")
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    emb_mean, emb_std = -0.005838499, 0.48782197

    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBED_SIZE))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]

            if i >= nb_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector

    print('Glove:', embedding_matrix.shape)
    return embedding_matrix

glove_embedding_matrix = load_glove()
model = simple_model(glove_embedding_matrix)
model.fit(aug_train_X, aug_y_train, batch_size=512, epochs=2, validation_data=(valid_X, y_valid))
