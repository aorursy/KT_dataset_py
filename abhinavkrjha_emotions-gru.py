import os,random,math

import warnings

import numpy as np

import pandas as pd

from nltk.corpus import stopwords

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from keras.preprocessing.text import Tokenizer

from keras import Input,Model

from keras.layers import LSTM,Embedding,Dropout,Activation,Reshape,Dense,GRU

warnings.filterwarnings('ignore')
LOCATION = "../input/emotion-classification"

LOCATION_GLOVE = "../input/glove6b50dtxt/glove.6B.50d.txt"
data = pd.read_csv(os.path.join(LOCATION,'emotion.data'),usecols=['text','emotions']) # as only text and emotions coloumns are useful to us hence use use only them in dataframe

print('The number of rows is',data.shape[0])

print('The number of cols is',data.shape[1])
data.groupby('emotions').size()
data.head()
input_sentences = [text.split(" ") for text in data["text"].values.tolist()]

labels = data["emotions"].values.tolist()



# Initialize word2id and label2id dictionaries that will be used to encode words and labels

word2id = dict()

label2id = dict()



max_words = 0 # maximum number of words in a sentence



# Construction of word2id dict

for sentence in input_sentences:

    for word in sentence:

        # Add words to word2id dict if not exist

        if word not in word2id:

            word2id[word] = len(word2id)

    # If length of the sentence is greater than max_words, update max_words

    if len(sentence) > max_words:

        max_words = len(sentence)

len_vocab = len(word2id)+1

# Construction of label2id and id2label dicts

label2id = {l: i for i, l in enumerate(set(labels))}

id2label = {v: k for k, v in label2id.items()}

id2label
X = data.loc[:,'text']
max_fatures = 30000

max_nb_words = 76000

EMBEDDING_DIM = 50

MAX_SEQUENCE_LENGTH = 32



tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data.loc[:,'text'].values)

encoded_docs= tokenizer.texts_to_sequences(data.loc[:,'text'])



padded_docs_X = pad_sequences(encoded_docs, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

word_index = tokenizer.word_index

num_words = min(max_nb_words,len(word_index))

print('Number of words',num_words)


def setup_embedding_index():

    embedding_index=dict()

    f = open(LOCATION_GLOVE,encoding='utf-8')

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.array(values[1:],dtype='float32')

        embedding_index[word] = coefs

    return embedding_index

embedding_index = setup_embedding_index()
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))

words = (list(word_index.keys()))[:max_nb_words]



for word,i in word_index.items():

    if i>=max_nb_words:

        continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

dims = len(embedding_matrix[0])



embedding_layer = Embedding(num_words,dims,weights = [embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable = False)
# Encode input words and labels

# X = [[word2id[word] for word in sentence] for sentence in input_sentences]

Y = [label2id[label] for label in labels]



# X = pad_sequences(X, max_words)

X = padded_docs_X

# Convert Y to numpy array

Y = to_categorical(Y, num_classes=len(label2id), dtype='float32')



# Print shapes

print("Shape of X: {}".format(X.shape))

print("Shape of Y: {}".format(Y.shape))
embedding_dim = 50 # The dimension of word embeddings



# Define input tensor

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_inputs = embedding_layer(sequence_input)

gru = GRU(64, activation='relu', return_sequences=False)(embedded_inputs)

drop = Dropout(0.2)(gru)

preds = Dense(6, activation='softmax')(drop)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])

model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.fit(X, Y, epochs=5, batch_size=64, validation_split=0.2, shuffle=True)