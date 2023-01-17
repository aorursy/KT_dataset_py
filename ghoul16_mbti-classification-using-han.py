import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

%matplotlib inline

from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Model
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
data=pd.read_csv('../input/mbti-type/mbti_1.csv')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer() 
cachedStopWords = stopwords.words("english")

def pre_process_data(data, remove_stop_words=True):
    list_posts = []
    len_data = len(data)
    i=0   
    for row in data.iterrows():
        i+=1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts) #remove urls
        temp = re.sub("[^a-zA-Z.]", " ", temp) #remove all punctuations except fullstops 
        temp = re.sub(' +', ' ', temp).lower() 
        temp=re.sub(r'\.+', ".", temp) #remove multiple fullstops
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
        list_posts.append(temp)

    text = np.array(list_posts)
    return text
clean_text = pre_process_data(data, remove_stop_words=True)
data['clean_text']=clean_text
data = data[['clean_text', 'type']]
data.head()
types=data['type']
text=data['clean_text']
tps=data.groupby('type')
print("total types:",tps.ngroups)
print(tps.size())
paras = []
labels = []
texts = []
sent_lens = []  # will contain lengths of all sentences in complete dataset
sent_nums = []  # will contain number of sentences in each instance 
for idx in range(data.clean_text.shape[0]):
    temp_text = data.clean_text[idx]
    texts.append(temp_text)
    sentences = tokenize.sent_tokenize(temp_text)
    valid_sentences=[]  
    for sent in sentences:
        temp_len=len(text_to_word_sequence(sent))
        if ((temp_len>8) and (temp_len<16)):
            valid_sentences.append(sent) 
    sent_nums.append(len(valid_sentences))
    for valid_sent in valid_sentences:
        sent_lens.append(len(text_to_word_sequence(valid_sent)))
    paras.append(valid_sentences)
max_features=200000 # maximum number of unique words that should be included in the tokenized word index
max_senten_len=16   # maximum number of words in each sentence
max_senten_num=6  # maximum number of sentences in one document
embed_size=100      # vector size of word embedding
VALIDATION_SPLIT = 0.2
tokenizer = Tokenizer(num_words=max_features, oov_token=True)
tokenizer.fit_on_texts(texts)
using_instances=1000
data_2 = np.zeros((1000, max_senten_num, max_senten_len), dtype='int32')
for i, sentences in enumerate(paras[:1000]):
    for j, sent in enumerate(sentences):
        if j< max_senten_num:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<max_senten_len and tokenizer.word_index[word]<max_features:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1
                else:
                    print(word)
                    pass
    print("Indexing done for Instance "+str(i))
word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
labels = pd.get_dummies(types[:1000])
print('Shape of data tensor:', data_2.shape)
print('Shape of labels tensor:', labels.shape)
#This implementation is taken from Github Repository DeepResearch, Author: Heet Sankesara

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
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

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
                'W_regularizer': self.W_regularizer,
                'u_regularizer': self.u_regularizer,
                'b_regularizer': self.b_regularizer,
                'W_constraint': self.W_constraint,
                'u_constraint': self.u_constraint,
                'b_constraint': self.b_constraint,
                'bias': self.bias,
        })
        return config
indices = np.arange(data_2.shape[0])
np.random.shuffle(indices)
data_2 = data_2[indices]
labels = labels.iloc[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data_2.shape[0])

x_train = data_2[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data_2[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
print('Number of positive and negative reviews in traing and validation set')
print(y_train.columns.tolist())
print(y_train.sum(axis=0).tolist())
print(y_val.sum(axis=0).tolist())
REG_PARAM = 1e-13
l2_reg = regularizers.l2(REG_PARAM)
import os
GLOVE_DIR = "../input/glove-6b-100d/glove.6B.100d.txt"
embeddings_index = {}
f = open(GLOVE_DIR,encoding="utf8")
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        print(word)
        pass
f.close()
print('Total %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
absent_words = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        absent_words += 1
print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')
embedding_layer = Embedding(len(word_index) + 1,embed_size,weights=[embedding_matrix], input_length=max_senten_len, trainable=False)
word_input = Input(shape=(max_senten_len,), dtype='float32')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
word_att = AttentionWithContext()(word_dense)
wordEncoder = Model(word_input, word_att)

sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
preds = Dense(16, activation='softmax')(sent_att)
model = Model(sent_input, preds)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
checkpoint = ModelCheckpoint('1000_instances_model.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='auto') 
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=512, callbacks=[checkpoint])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
model.save('mbti_han.h5')
