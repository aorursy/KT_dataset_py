!ls ../input
from sklearn.datasets.base import get_data_home, _pkl_filepath
import os
CACHE_NAME = "20news-bydate.pkz"
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"

data_home = get_data_home()
print(data_home)
cache_path = _pkl_filepath(data_home, CACHE_NAME)
print(cache_path)
twenty_home = os.path.join(data_home, "20news_home")
print(twenty_home)

if not os.path.exists(data_home):
    os.makedirs(data_home)
    
if not os.path.exists(twenty_home):
    os.makedirs(twenty_home)
os.path.exists(data_home)
!ls /tmp
!cp ../input/20-newsgroup-sklearn/20news-bydate_py3* /tmp/scikit_learn_data
os.path.exists(cache_path)
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, download_if_missing=False)
twenty_train.data[1]
twenty_train.target_names[twenty_train.target[1]]
texts = twenty_train.data # Extract text
target = twenty_train.target # Extract target
# Load tools we need for preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
vocab_size = 20000
tokenizer = Tokenizer(num_words=vocab_size) # Setup tokenizer
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Generate sequences
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# Create inverse index mapping numbers to words
inv_index = {v: k for k, v in tokenizer.word_index.items()}
# Print out text again
for w in sequences[1]:
    x = inv_index.get(w)
    print(x,end = ' ')
import numpy as np
# Get the average length of a text
avg = sum( map(len, sequences) ) / len(sequences)

# Get the standard deviation of the sequence length
std = np.sqrt(sum( map(lambda x: (len(x) - avg)**2, sequences)) / len(sequences))

avg,std
max_length = 100
data = pad_sequences(sequences, maxlen=max_length)
import numpy as np
from keras.utils import to_categorical
labels = to_categorical(np.asarray(target))
print('Shape of data:', data.shape)
print('Shape of labels:', labels.shape)
import os
glove_dir = '../input/glove-global-vectors-for-word-representation' # This is the folder with the dataset

embeddings_index = {} # We create a dictionary of word -> embedding
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt')) # Open file

# In the dataset, each line represents a new word embedding
# The line starts with the word and the embedding values follow
for line in f:
    values = line.split()
    word = values[0] # The first value is the word, the rest are the values of the embedding
    embedding = np.asarray(values[1:], dtype='float32') # Load embedding
    embeddings_index[word] = embedding # Add embedding to our embedding dictionary
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# Create a matrix of all embeddings
all_embs = np.stack(embeddings_index.values())
emb_mean = all_embs.mean() # Calculate mean
emb_std = all_embs.std() # Calculate standard deviation
emb_mean,emb_std
embedding_dim = 100 # We use 100 dimensional glove vectors
word_index = tokenizer.word_index
nb_words = min(vocab_size, len(word_index)) # How many words are there actually

# Create a random matrix with the same mean and std as the embeddings
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))

# The vectors need to be in the same position as their index. 
# Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

# Loop over all words in the word index
for word, i in word_index.items():
    # If we are above the amount of words we want to use we do nothing
    if i >= vocab_size: 
        continue
    # Get the embedding vector for the word
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding
model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=max_length, 
                    weights = [embedding_matrix], 
                    trainable = False))
model.add(LSTM(128))
model.add(Dense(20))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit(data,labels,validation_split=0.2,epochs=2)
example = data[10] # get the tokens
# Print tokens as text
for w in example:
    x = inv_index.get(w)
    print(x,end = ' ')
# Get prediction
pred = model.predict(example.reshape(1,100))
# Output predicted category
twenty_train.target_names[np.argmax(pred)]
model = Sequential()
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    input_length=max_length, 
                    weights = [embedding_matrix], 
                    trainable = False))

# Now with recurrent dropout with a 10% chance of removing any element
model.add(LSTM(128, recurrent_dropout=0.1)) 
model.add(Dense(20))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

model.fit(data,labels,validation_split=0.2,epochs=2)