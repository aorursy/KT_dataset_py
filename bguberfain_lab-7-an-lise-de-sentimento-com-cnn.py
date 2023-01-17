import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(5)
import os

from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.optimizers import Adam

import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import log_loss, classification_report
dfAvaliacoesAnalisadas = pd.read_csv('../input/all.csv')
dfAvaliacoesAnalisadas.head()
def converteCategoria(df, coluna):
    le = preprocessing.LabelEncoder()
    le.fit(df[coluna])
    df[coluna] = le.transform(df[coluna])
    return le

num_classes = len(dfAvaliacoesAnalisadas.manifest_atendimento.unique())

labelEncoderManifAtendimento = converteCategoria(dfAvaliacoesAnalisadas, 'manifest_atendimento')
print(num_classes)
import re
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^a-zA-Z0-9ÀÁÂÃÄÅÇÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝàáâãäåçèéêëìíîïðòóôõöùúûüýÿ,!?\'\`\.\(\)]", " ", string)
    string = re.sub(r"INC[0-9]{7,}", " <INCIDENTE> ", string)
    string = re.sub(r"[+-]?\d+(?:\.\d+)?", " <NUMERO> ", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() #.lower()

dfAvaliacoesAnalisadas['coment'] = dfAvaliacoesAnalisadas['coment'].apply(clean_str)
dfAvaliacoesAnalisadas['coment'] = dfAvaliacoesAnalisadas['coment'].apply(lambda x : x.split(' '))
%matplotlib inline
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

sequence_length = dfAvaliacoesAnalisadas['coment'].apply(len).values
print(np.percentile(sequence_length, 99.9))
print(np.max(sequence_length))
plt.hist(sequence_length)
def pad_sentence(sentence, sequence_length, padding_word="<PAD/>"):
    if len(sentence) > sequence_length:
        sentence = sentence[:sequence_length]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    return new_sentence

corte = np.max(sequence_length)
dfAvaliacoesAnalisadas['coment'] = dfAvaliacoesAnalisadas['coment'].apply(lambda x : pad_sentence(x, corte))
import itertools
from collections import Counter

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return (vocabulary, vocabulary_inv)

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return (x, y)

comments = dfAvaliacoesAnalisadas['coment'].values
labels = dfAvaliacoesAnalisadas['manifest_atendimento'].values

vocabulary, vocabulary_inv = build_vocab(comments)
X, ylabels = build_input_data(comments, labels, vocabulary)
from gensim.models import word2vec
from os.path import join, exists, split

def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {str:int}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    model_dir = 'word2vec_models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)
    model_name = join(model_dir, model_name)
    if exists(model_name) and False:
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 4       # Number of threads to run in parallel
        downsampling = 1e-3   # Downsample setting for frequent words
        
        # Initialize and train the model
        print("Training Word2Vec model...")
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers, \
                            size=num_features, min_count = min_word_count, \
                            window = context, sample = downsampling)
        
        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)
        
        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)
    
    #  add unknown words
    embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                                        for w in vocabulary_inv])]
    return embedding_weights
embedding_dim = 50
min_word_count = 1
context = 10

embedding_weights = train_word2vec(X, vocabulary_inv, embedding_dim, min_word_count, context)
print("Tamanho do vocabulário: {:d}".format(len(vocabulary)))
print(embedding_weights[0].shape) # número de palavras x tamanho do vetor definido.
filter_sizes = (3, 4, 5) # cada item da lista representa os tamanhos de filtro que usaremos
num_filters = 128 # quantidade de filtro para cada um dos tamanhos acima
dropout_prob = (0.3, 0.5) # probabilidade de cada camada de Dropout
hidden_dims = 64 # número de neurônios na camada densa final
def build_model():

    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1)
        pool = MaxPooling1D(pool_length=2)
        
    model = Sequential()
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=corte, weights=embedding_weights))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))

    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
model = build_model()
model.summary()
plot_model(model, to_file='plot_model.png', show_shapes=True)
from sklearn.model_selection import train_test_split
y = to_categorical(ylabels)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
y = to_categorical(ylabels)
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_valid, y_valid), verbose=1)
preds = model.predict_proba(X_valid) # as previsões são probabilidades para cada uma das 3 classes

#conta o número de acertos, considerando a classe de maior probabilidade
acc_score = np.sum(np.argmax(preds,1)==np.argmax(y_valid,1))/float(len(y_valid))
#calcula o categorical log-loss
log_loss_score= log_loss(y_valid, preds)
print('Accuracy: %.4f Categorical log-loss: %.4f' % (acc_score, log_loss_score))
