# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
comments_data = pd.read_csv(r'../input/airline-sentiment/Tweets.csv')
comments_data.head(n=10)
import re
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
# extract sentiment(positive/negative/neutral) into labels
labels = comments_data["airline_sentiment"].map({'neutral':0,'negative':1,'positive':2})
labels = np.asarray(labels, dtype=int)

# extract text to be analysed into comments
comments = comments_data["text"]
#remove words which are starts with @ symbols
comments = comments.map(lambda x: " ".join(str(x).split()))
comments = comments.map(lambda x:re.sub('@\w*','',str(x)))
#remove special characters except [a-zA-Z]
comments = comments.map(lambda x:re.sub('[^a-zA-Z]',' ',str(x)))
#remove link starts with https
comments = comments.map(lambda x:re.sub('http.*','',str(x)))
#convert to ndarray
comments = np.asarray(comments)
# Change index value to see comments and corresponding sentiment type
index = 0
print(comments[index], labels[index])
# Find number of unique categories
num_classes = comments_data["airline_sentiment"].unique().shape[0]
# Convert to one-hot vector
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y
Y_one_hot = convert_to_one_hot(labels, C = num_classes)
# Split into training and test data
X_train,X_test,y_train,y_test = train_test_split(comments,Y_one_hot,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
maxLen = 40
#Load Glove Vectors
GLOVE_DIR='../input/glove-global-vectors-for-word-representation/'

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
words = set()
word_to_vec_map = {}
for line in f:
    values = line.strip().split()
    word = values[0]
    words.add(word)
    word_to_vec_map[word] = np.array(values[1:], dtype=np.float64)
    
i = 1
words_to_index = {}
index_to_words = {}
for w in sorted(words):
    words_to_index[w] = i
    index_to_words[i] = w
    i = i + 1
f.close()
#look at a word embedding
word_to_vec_map["hello"].shape
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["hello"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim,trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
def sentences_to_indices(X, word_to_index, max_len):
    
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):
        
        sentence_words = X[i].strip().lower().split(" ")
        j = 0
        
        for w in sentence_words:
            if w and w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j = j + 1
    
    return X_indices
input_shape = (maxLen,)
sentence_indices = Input(shape=input_shape,dtype='int32')
    
# Create the embedding layer pretrained with GloVe Vectors
embedding_layer = pretrained_embedding_layer(word_to_vec_map, words_to_index)
    
# Embedding layer, you get back the embeddings
embeddings = embedding_layer(sentence_indices)   
    
# LSTM layer with 128-dimensional hidden state
X = LSTM(128,return_sequences=True)(embeddings)
X = Dropout(rate=0.5)(X)
# LSTM layer with 128-dimensional hidden state
X = LSTM(128)(X)
X = Dropout(rate=0.5)(X)
# Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
X = Dense(num_classes)(X)
# Add a softmax activation
X = Activation("softmax")(X)
    
# Create Model instance which converts sentence_indices into X.
model = Model(inputs=sentence_indices, outputs=X)    
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
X_train_indices = sentences_to_indices(X_train, words_to_index, maxLen)
model.fit(X_train_indices, y_train, epochs = 20, batch_size = 32, shuffle=True)
X_test_indices = sentences_to_indices(X_test, words_to_index, maxLen)
loss, acc = model.evaluate(X_test_indices, y_test)
print()
print("Test accuracy = ", acc)
#Show mis-classified comments
test_data = np.asarray(["worst experience ever","it was a wonderful experience","it was just fine", "hello how are you doing", "not a very good airline"])
test_data_y = np.asarray([1,2,2,0,1])
test_indices = sentences_to_indices(test_data, words_to_index, maxLen)
pred = model.predict(test_indices)
for i in range(len(test_data)):
    x = test_indices
    num = np.argmax(pred[i])
    act = test_data_y[i]
    print(test_data[i] + '::::Expected='+ str(act) + ', Prediction=' + str(num))


