# the obvious
import numpy as np
import pandas as pd

# core utility modules
from os import listdir, path
import string
from collections import Counter
import time
import gc

# for visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud
%matplotlib inline

# for preprocessing and feature extraction
import keras.preprocessing.text as text
import keras.preprocessing.sequence as seq 
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# for logging and early stopping and learning rate scheduling
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# for the metric (F1 Score)
from sklearn.metrics import f1_score

# for model creation and training
from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Input, LSTM, Dropout, Bidirectional, CuDNNLSTM, CuDNNGRU, SimpleRNN, Embedding, GlobalMaxPool1D
import keras.backend as K
from sklearn.svm import SVC
from keras.optimizers import Adam
from keras import initializers

# other imports
import operator 
import re
max_seq_len = 60 # The Max Length Of The Text Sequence
embed_size = 300 # The Number Of Features In The Embedding For A Single Word
max_features = 50000 # The Maximum Number Of Words In The Vocab
EMBEDDING = 'glove.840B.300d' # Learned Embedding To Be Used, Change This For Using Different Embeddings
MODEL = 'attention' # The Model To Use To Classify The Insincere/Sincere Questions, Other Possible Vals Are : 'nb', 'svm', 'rnn', 'gru' and 'lstm'
embedding_matrix = 'None' # The Embeddigns Matrix
embeddings_idx = 'None' # The Mappings From Embedding Index To The Embedding
checkpoint = ModelCheckpoint('./checkpoints/', monitor='val_acc', verbose=0, save_best_only=True)
earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=0)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
reducelr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
thresh = 0.4
listdir('../input')
train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
train_set.head()
test_set.head()
x = train_set['target'].value_counts(dropna=False)
print(x)
sincere_examples = x[0]
insincere_examples = x[1]
plt.hist(train_set['target'], bins=range(0,6), align='left', rwidth=1)
# max and min question lengths
# to remove punctuations : translate(str.maketrans('','',string.punctuation))
lengths_without_puncs = [len(i.translate(str.maketrans('','',string.punctuation)).split()) for i in train_set['question_text']]
lengths = [len(i.split()) for i in train_set['question_text']]
print('With Punctuations: ')
print('Max Length Of Questions: {}'.format(np.max(lengths)))
print('Min Length Of Questions: {}'.format(np.min(lengths)))
print('Without Punctuations: ')
print('Max Length Of Questions: {}'.format(np.max(lengths_without_puncs)))
print('Min Length Of Questions: {}'.format(np.min(lengths_without_puncs)))
# print(len(lengths))
print(len(lengths_without_puncs) - np.count_nonzero(lengths_without_puncs)) # Will remove them or use fillna to overcome this
plt.hist(lengths)
plt.yscale('log')
# Code from https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def clean_contractions(text, mapping=contraction_mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def clean_special_chars(text, punct=punct, mapping=punct_mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text

def correct_spelling(x, dictionary=mispell_dict):
    for word in dictionary.keys():
        x = x.replace(word, dictionary[word])
    return x


def clean(text):
    text = text.lower()
    text = clean_contractions(text)
    text = clean_special_chars(text)
    text = correct_spelling(text)
    return text
sincere_counts = Counter()
insincere_counts = Counter()
word_dict = Counter()
sincere_to_insincere_ratio = Counter()

def prepare_dicts():
    qs = [clean(i) for i in train_set['question_text']]
    lbl = [j for j in train_set['target']]
    for i,j in zip(qs,lbl):
        words = i.split()
        # making the dictionaries
        for word in words:
            word_dict[word] += 1
            if j == 0:
                sincere_counts[word] += 1
            elif j == 1:
                insincere_counts[word] += 1
    
    tst_qs = [clean(i) for i in test_set['question_text']]
    
    for i in tst_qs:
        i = i.split()
        for j in i:
            word_dict[j] += 1
    
    print('Words in sincere Questions: {}'.format(len(sincere_counts)))
    print('Words in insincere Questions: {}'.format(len(insincere_counts)))
    print('Total Words in corpus: {}'.format(len(word_dict)))

    print('Most Common Words in Sincere Questions : ')
    print(sincere_counts.most_common()[:10])
    print('Most Common Words in Insincere Questions : ')
    print(insincere_counts.most_common()[:10])

    for i in sincere_counts:
        if sincere_counts[i] >= 100:
            sincere_to_insincere_ratio[i] = np.log(sincere_counts[i]/(insincere_counts[i] + 1))

    print('The Most Sincere Words : ')
    print(sincere_to_insincere_ratio.most_common()[:10])
    print('The Most Insincere Words : ')
    print(list(reversed(sincere_to_insincere_ratio.most_common()))[:10])
    
#     return sincere_counts, insincere_counts, word_dict
prepare_dicts()
wordCloud = WordCloud().generate(" ".join([key[0] for key in sincere_to_insincere_ratio.most_common()[:10]]))
fig = plt.figure()
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
fig.suptitle('Most Common Words In Sincere Questions', fontsize=14, fontweight='bold')
plt.show()
wordCloud = WordCloud().generate(" ".join([key[0] for key in list(reversed(sincere_to_insincere_ratio.most_common()))[:10]]))
fig = plt.figure()
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
fig.suptitle('Most Common Words In Insincere Questions', fontsize=14, fontweight='bold')
plt.show()
# Creating The Datasets First
train_x = list(train_set['question_text'].fillna("_na_").values)
train_y = train_set['target'].values

test_x = list(test_set['question_text'].fillna("_na_").values)

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)

# Cleaning Up The Data (train + test)
train_x = [clean(i) for i in train_x]
val_x = [clean(i) for i in val_x]
test_x = [clean(i) for i in test_x]

# An Example From Train Set
print('An Example From Train Set: ')
print(train_x[0])

## Tokenize the sentences
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x))
train_X = tokenizer.texts_to_sequences(train_x)
val_X = tokenizer.texts_to_sequences(val_x)
test_X = tokenizer.texts_to_sequences(test_x)

# After Tokenizing
print('After Tokenizing: ')
print(train_X[0])

## Pad the sentences 
train_X = seq.pad_sequences(train_X, maxlen=max_seq_len)
val_X = seq.pad_sequences(val_X, maxlen=max_seq_len)
test_X = seq.pad_sequences(test_X, maxlen=max_seq_len)

# After Padding
print('After Padding: ')
print(train_X[0])

print(np.shape(train_X), np.shape(train_y), np.shape(val_X), np.shape(val_y))
# Thanks to https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def get_embeddings(embedding_name, mode='new'):
    # Getting The File
    filePath = '../input/embeddings/{0}/{0}.txt'.format(embedding_name)
    
    # Creating a Dictionary of format {word : Embedding}
    if mode == 'new':
        embeddings_idx = dict(get_coefs(*i.split(" ")) for i in open(filePath))
        # All Embeddings
        all_embs = np.stack(embeddings_idx.values())

        # Creating The Embedding Matrix with distribution, for if there is a missing word in the embeddings, it'll have
        # the embedding vector with the same distribution
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]

        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        # Filling in the given learned embeddings in the embedding matrix
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_idx.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
                
    return embeddings_idx, embedding_matrix
# Checking OOV words (Out Of Vocab words)
def check_coverage(vocab, embeddings_index):
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

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
embedding_idxs, embedding_mtx = get_embeddings(EMBEDDING, 'new')
unk_wrds = check_coverage(word_dict, embedding_idxs)
print(unk_wrds[:10])
print(np.shape(embedding_mtx))
# Create The Model
def get_model(model_type):
    if model_type == 'nb':
        # create naivebayes model
        model = NaiveBayes()

    elif model_type == 'svm':
        # create svm model
        model = __SVC__()

    elif model_type == 'rnn':
        inp = Input(shape=(max_seq_len,))
        layer = Embedding(max_features, embed_size, weights=[embedding_mtx], trainable=False)(inp)
#         layer = SimpleRNN(128, return_sequences=True)(layer)
        layer = SimpleRNN(32, return_sequences=True)(layer)
        layer = GlobalMaxPool1D()(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dropout(0.1)(layer)
        layer = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs=inp, outputs=layer)

    elif model_type == 'lstm':
        # create lstm model
        inp = Input(shape=(max_seq_len,))
        layer = Embedding(max_features, embed_size, weights=[embedding_mtx], trainable=False)(inp)
#         layer = Bidirectional(CuDNNLSTM(128, return_sequences=True))(layer)
        layer = Bidirectional(CuDNNLSTM(32, return_sequences=True))(layer)
        layer = GlobalMaxPool1D()(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dropout(0.1)(layer)
        layer = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs=inp, outputs=layer)

    elif model_type == 'gru':
        # create attention model
        inp = Input(shape=(max_seq_len,))
        layer = Embedding(max_features, embed_size, weights=[embedding_mtx], trainable=False)(inp)
#         layer = Bidirectional(CuDNNGRU(128, return_sequences=True))(layer)
        layer = Bidirectional(CuDNNGRU(32, return_sequences=True))(layer)
        layer = GlobalMaxPool1D()(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dropout(0.1)(layer)
        layer = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs=inp, outputs=layer)

    elif model_type == 'attention':
        inp = Input(shape=(max_seq_len,))
        layer = Embedding(max_features, embed_size, weights=[embedding_mtx], trainable=False)(inp)
#         layer = Bidirectional(CuDNNLSTM(128, return_sequences=True))(layer)
        layer = Bidirectional(CuDNNLSTM(32, return_sequences=True))(layer)
        layer = Attention(max_seq_len)(layer)
        layer = Dense(16, activation='relu')(layer)
        layer = Dropout(0.1)(layer)
        layer = Dense(1, activation='sigmoid')(layer)
        model = Model(inputs=inp, outputs=layer)

    return model

# Defining The NaiveBayes Class
class NaiveBayes():
    def __init__(self):
        self.sincere_example_count = sincere_examples
        self.insincere_example_count = insincere_examples
        self.total_examples = x[0]+x[1]
        self.sincere_dict = sincere_counts
        self.insincere_dict = insincere_counts
        self.word_dict= word_dict
        self.sincere_word_count = np.sum(list(sincere_counts.values()))
        self.insincere_word_count = np.sum(list(insincere_counts.values()))
        self.sincere_prob = self.sincere_example_count / self.total_examples
        self.insincere_prob = self.insincere_example_count / self.total_examples
    
    def summary(self):
        print('Positive Examples : {}, Negative Examples : {}, Total Examples : {}'.format(self.sincere_example_count, self.insincere_example_count, self.total_examples))
    
    def predict(self, x_test):
        # The NB Prediction with Laplace Smoothing
        print('Predicting...')
        predictions = []
        for example in x_test:
            p_words = np.prod([word_dict[j]/np.sum(list(word_dict.values())) for j in example.split()])
            p_words += 2
            sincere_prob_num = np.prod([sincere_counts[j]/self.sincere_word_count for j in example.split()]) * self.sincere_prob
            insincere_prob_num = np.prod([insincere_counts[j]/self.insincere_word_count for j in example.split()]) * self.insincere_prob

            sincere_prob = sincere_prob_num/p_words
            insincere_prob = insincere_prob_num/p_words

#             print('Sincere_prob: {}, Insincere_prob: {}'.format(sincere_prob, insincere_prob))
            predictions.append(np.argmax([sincere_prob, insincere_prob]))
#             print('predicted Class : {}'.format(np.argmax([sincere_prob, insincere_prob])))
        return predictions

# The SVM Class
class __SVC__(SVC):
    def __init__(self):
        super(__SVC__,self).__init__(verbose=True)
        print('initializing...')
    
    def summary(self):
        print(self.__dict__)
        
    def prepare_data(self):
        self.X_train = [embedding_mtx[i] for example in train_X for i in example]
        self.X_val = [embedding_mtx[i] for example in val_X for i in example]
        self.X_test = [embedding_mtx[i] for example in test_X for i in example]
class Attention(Layer):
    def __init__(self, step_dim, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.features_dim = 0
        self.step_dim = step_dim
        self.bias = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     )
        else:
            self.b = None

        self.built = True

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
def print_f1s(predictions):
    for threshold in np.arange(0.1, 0.501, 0.01):
        threshold = np.round(threshold, 2)
        print("F1 score at threshold {0} is {1}".format(threshold, f1_score(val_y, (predictions>threshold).astype(int))))
# nb = get_model('nb')
# nb.summary()
# predictions = nb.predict(val_x)
# predictions_nb = nb.predict(test_x)
# print('Done!')

# print_f1s(predictions)
    
# predictions_nb = pd.DataFrame({"qid":test_set["qid"].values})
# predictions_nb['prediction'] = predictions_nb
# predictions_nb.to_csv("submission_nb.csv", index=False)

# #freeing up some memory
# del nb, word_dict, sincere_counts, insincere_counts, sincere_to_insincere_ratio

# gc.collect()
# time.sleep(10)

# svm = get_model('svm')
# svm.summary()
# svm.prepare_data()
# svm.fit(svm.X_train, train_y)

# predictions = svm.predict(svm.X_val)

# predictions_svm = svm.predict(svm.X_test)

# print_f1s(predictions)
    
# # predictions_svm = pd.DataFrame({"qid":test_set["qid"].values})
# # predictions_svm['prediction'] = predictions_svm
# # predictions_svm.to_csv("submission_svm.csv", index=False)

# del svm
# gc.collect()
# time.sleep(10)

rnn = get_model('rnn')
rnn.summary()
rnn.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
rnn.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y), callbacks=[earlystop, reducelr])


predictions_rnn_real = rnn.predict(test_X)
predictions_rnn = (predictions_rnn_real >= thresh).astype(int)
predictions_val_rnn = rnn.predict(val_X, batch_size=1024)

print_f1s(predictions_val_rnn)

# prediction_rnn = pd.DataFrame({"qid":test_set["qid"].values})
# prediction_rnn['prediction'] = predictions_rnn
# prediction_rnn.to_csv("submission.csv", index=False)

del rnn
gc.collect()
time.sleep(10)
gru = get_model('gru')
gru.summary()
gru.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
gru.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y), callbacks=[earlystop, reducelr])

predictions_gru_real = gru.predict(test_X)
predictions_gru = (predictions_gru_real >= thresh).astype(int)
predictions_val_gru = gru.predict(val_X, batch_size=1024)

print_f1s(predictions_val_gru)

prediction_gru = pd.DataFrame({"qid":test_set["qid"].values})
prediction_gru['prediction'] = predictions_gru
prediction_gru.to_csv("submission.csv", index=False)

del gru
gc.collect()
time.sleep(10)
lstm = get_model('lstm')
lstm.summary()
lstm.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
lstm.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y), callbacks=[earlystop, reducelr])

predictions_lstm_real = lstm.predict(test_X)
predictions_lstm = (predictions_lstm_real >= thresh).astype(int)
predictions_val_lstm = lstm.predict(val_X, batch_size=1024)

print_f1s(predictions_val_lstm)

# prediction_lstm = pd.DataFrame({"qid":test_set["qid"].values})
# prediction_lstm['prediction'] = predictions_lstm
# prediction_lstm.to_csv("submission.csv", index=False)

del lstm
gc.collect()
time.sleep(10)
attention = get_model('attention')
attention.summary()
attention.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
attention.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y), callbacks=[earlystop, reducelr])

predictions_attention_real = attention.predict(test_X)
predictions_attention = (predictions_attention_real >= thresh).astype(int)
predictions_val_attention = attention.predict(val_X, batch_size=1024)

print_f1s(predictions_val_attention)

# prediction_attention = pd.DataFrame({"qid":test_set["qid"].values})
# prediction_attention['prediction'] = predictions_attention
# prediction_attention.to_csv("submission.csv", index=False)

del attention
gc.collect()
time.sleep(10)

val_preds = 0.50*predictions_val_gru + 0.25*predictions_val_lstm + 0.25*predictions_val_attention
val_preds = (val_preds > thresh).astype(int)
print_f1s(val_preds)
final_preds = 0.50*predictions_gru_real + 0.25*predictions_lstm_real + 0.25*predictions_attention_real
final_preds = (final_preds > thresh).astype(int)

final_prediction = pd.DataFrame({"qid":test_set["qid"].values})
final_prediction['prediction'] = final_preds
final_prediction.to_csv("submission.csv", index=False)
