import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/glove6b50dtxt"))
import pandas as pd

data= pd.read_csv("../input/emojify/emojify_data.csv",usecols=[0,1],names=['colA', 'colB'])

text=data["colA"].tolist()

emojis=np.array(data["colB"])

print("{} +++> {}".format(text[1],emojis[1]))

maxLen = len(max(text, key=len).split())

maxLen = len(max(text, key=len).split())
import emoji

emoji_dictionary = {"0": "\u2764\uFE0F",

                    "1": ":baseball:",

                    "2": ":smile:",

                    "3": ":disappointed:",

                    "4": ":fork_and_knife:"}

def label_to_emoji(label):

    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)





    

C=len(emoji_dictionary)

np.asarray(emojis)

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')

emojis = emojis.reshape(len(emojis), 1)

Y = onehot_encoder.fit_transform(emojis)

print(Y)

def read_glove_vecs(glove_file):

    with open(glove_file, 'r') as f:

        words = set()

        word_to_vec_map = {}

        

        for line in f:

            line = line.strip().split()

            curr_word = line[0]

            words.add(curr_word)

            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1

        words_to_index = {}

        index_to_words = {}

        for w in sorted(words):

            words_to_index[w] = i

            index_to_words[i] = w

            i = i + 1   

    return words_to_index,index_to_words, word_to_vec_map

words_to_index,index_to_words,word_to_vec_map=read_glove_vecs("../input/glove6b50dtxt/glove.6B.50d.txt")
def sentences_to_indices(X, word_to_index,max_len):

    m = X.shape[0] 

    X_indices = np.zeros((m,max_len))

    for i in range(m):

        sentence_words =[words.lower() for words in X[i].split()]

        

        j=0

        for w in sentence_words:

            X_indices[i, j] = word_to_index[w]

            j += 1

    return X_indices

X1 = np.array(["Hello"])

X=sentences_to_indices(X1, words_to_index,maxLen)

print(X)

        

        

        

    

    

    


# GRADED FUNCTION: pretrained_embedding_layer

from keras.layers.embeddings import Embedding

def pretrained_embedding_layer(word_to_vec_map, words_to_index):

    vocab_len=len(words_to_index)+1

    emb_dim = word_to_vec_map["cucumber"].shape[0] 

    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, index in words_to_index.items():

        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)

    embedding_layer.build((None,))

    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

    
from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation

def Emojify_V2(input_shape, word_to_vec_map, words_to_index):

    sentence_indices = Input(input_shape, dtype='int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)

    X = Dropout(0.5)(X)

    X = LSTM(128, return_sequences=False)(X)

    X = Dropout(0.5)(X)

    X = Dense(5)(X)

    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model

model = Emojify_V2((maxLen,), word_to_vec_map, words_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs = 50, batch_size = 32, shuffle=True)
inp=input("Enter the text: ")

inputs=np.array([inp])

X=sentences_to_indices(inputs, words_to_index,maxLen)

print("{}::{}".format(inp,label_to_emoji(np.argmax(model.predict(X)))))