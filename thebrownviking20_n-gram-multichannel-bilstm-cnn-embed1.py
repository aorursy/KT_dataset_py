# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import os
print(os.listdir("../input"))
print(os.listdir("../input/embeddings"))

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Keras based imports
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
quora_train = pd.read_csv("../input/train.csv")
quora_test = pd.read_csv("../input/test.csv")
quora_train.head()
punctuations = string.punctuation

def punct_remover(my_str):
    my_str = my_str.lower()
    no_punct = ""
    for char in my_str:
       if char not in punctuations:
           no_punct = no_punct + char
    return no_punct

punctuations
tqdm.pandas()
questions = quora_train["question_text"].progress_apply(punct_remover)
test = quora_test["question_text"].progress_apply(punct_remover)
# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])
 
# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded
embeddings_index = dict()
f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt',encoding='utf8')
for line in f:
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embed_token = create_tokenizer(questions)
vocabulary_size = 90000

embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in embed_token.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
# define the model
def define_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs1)
    conv1 = Conv1D(filters=16, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    lstm1 = Bidirectional(CuDNNLSTM(10, return_sequences = True))(drop1)
    gru1 = Bidirectional(CuDNNGRU(10, return_sequences = True))(lstm1)
    pool1 = MaxPooling1D(pool_size=2)(gru1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs2)
    conv2 = Conv1D(filters=16, kernel_size=6, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    lstm2 = Bidirectional(CuDNNLSTM(10, return_sequences = True))(drop2)
    gru2 = Bidirectional(CuDNNLSTM(10, return_sequences = True))(lstm2)
    pool2 = MaxPooling1D(pool_size=2)(gru2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocabulary_size, 300, weights=[embedding_matrix])(inputs3)
    conv3 = Conv1D(filters=16, kernel_size=8, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    lstm3 = Bidirectional(CuDNNLSTM(10, return_sequences = True))(drop3)
    gru3 = Bidirectional(CuDNNGRU(10, return_sequences = True))(lstm3)
    pool3 = MaxPooling1D(pool_size=2)(gru3)
    flat3 = Flatten()(pool3)
    # merge
    merged = concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model
%%time
# Preprocess data

# create tokenizer
tokenizer = create_tokenizer(questions)
# calculate max document length
length = max_length(questions)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, questions, length)
testX = encode_text(tokenizer, test, length)
print(trainX.shape, testX.shape)
# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], quora_train["target"].values, epochs=5, batch_size=4096)

# save the model
model.save('model.h5')
preds = model.predict([testX,testX,testX])
preds = (preds[:,0] > 0.5).astype(np.int)
submission = pd.DataFrame.from_dict({'qid': quora_test['qid']})
submission['prediction'] = preds
submission.to_csv('submission.csv', index=False)