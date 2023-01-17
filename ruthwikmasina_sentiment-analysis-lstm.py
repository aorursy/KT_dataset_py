import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras.preprocessing.text import Tokenizer
np.random.seed(1)
import matplotlib.pyplot as plt

%matplotlib inline

# Load Data
df = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv', delimiter='\t')
test = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv', delimiter='\t')
pd.set_option('display.max_colwidth', -1)
df.head()
seed = 101 
np.random.seed(seed)

X = df['Phrase']
temp = test['Phrase']
y = to_categorical(df['Sentiment'])
num_classes = df['Sentiment'].nunique()
maxLen = 50
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
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
    return words_to_index, index_to_words, word_to_vec_map
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../input/nlpword2vecembeddingspretrained/glove.6B.100d.txt')
def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`
    """
    
    m = X.shape[0]                                   # number of training examples

    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        

        sentence_words = text_to_word_sequence(X[i],lower=True)
        
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            try:
                word_index =  word_to_index[w]
                if word_index is not None:
                    X_indices[i, j] = word_index
                    j = j+1
            except Exception:            # pass any exception occured
                pass

    
    return X_indices
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 100-dimensional vectors.
    
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["happy"].shape[0]      # define dimensionality of the GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
def get_model(input_shape, word_to_vec_map, word_to_index):
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)     
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
    
    
    return model
model = get_model((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Spilt Train Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y,random_state=seed)
# print(X_train.shape,y_train.shape)
X_train = np.asarray(X_train,dtype=str)
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
X_test = np.asarray(X_test,dtype=str)
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
model.fit(X_train_indices, y_train,validation_data=(X_test_indices, y_test), epochs = 7, batch_size = 128, verbose=2)
temp = np.asarray(temp,dtype=str)
temp_indices = sentences_to_indices(temp, word_to_index, maxLen)
predict_classes = model.predict(temp_indices)
classes_list = []

for x_test_predict in predict_classes:
    classes_list.append(np.argmax(x_test_predict))
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
sub['Sentiment'] = pd.Series(classes_list)
sub.to_csv("lstm_glove.csv", index=False)
