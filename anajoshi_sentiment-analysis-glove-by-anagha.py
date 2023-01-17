#dir_path = os.path.dirname(os.path.realpath("cv000_29416.txt"))#

#dir_path
# 1.0 Call libraries

%reset -f

import os

import numpy as np



# 1.1 Library to tokenize text (convert to integer-sequences)

from keras.preprocessing.text import Tokenizer

# 1.2 Make all sentence-tokens of equal length

from keras.preprocessing.sequence import pad_sequences

# 1.3 Modeling layers

from keras.models import Sequential

from keras.layers import Dense, Flatten, Embedding
# 2.0 Constants and files processing

# This dataset contains just 1000 comments each per neg/pos folder

#imdb_dir = '/home/ashok/Documents/10.nlp_workshop/imdb/'

train_dir =os.path.join("/kaggle/input/glove-anagha/", 'train')

#train_dir=os.chdir('/kaggle/input/glove-anagha/train')

#train_dir
# 2.1 Look into folders

os.listdir(train_dir)                # ['pos', 'neg']

os.listdir(train_dir + '/pos')[:5]   # List of files

os.listdir(train_dir+'/neg')[:5]     # List of files
# 2.3

maxlen_comment = 100          # If comment exceeds this, it will be truncated

training_samples = 400        # Use just 400 comments as training data

validation_samples = 500      # Use 500 samples for validation

max_words = 10000             # Select top 10000 words
# List of sentiment labels and comments

#     Start with none

labels = []

texts = []
# How many files are there?

fname = train_dir+'/neg'

len(os.listdir(fname))               # 1000



fname = train_dir+'/pos'

len(os.listdir(fname))  # 1000
# 2.4 Read files from each folder, one-by-one.

#     As we do so, we  read complete

#     file-text as one string element & append it

#     in the list,'texts': "this \n is \t black dog"

#     We also append its sentiment as indicated

#     by its folder name (neg: 0, pos:1)

#     in the list,'labels'.



for label_type in ['neg', 'pos']:

    # 2.3.1 Which directory

    dir_name = os.path.join(train_dir, label_type)

    # 2.3.2 For every file in this folder

    for fname in os.listdir(dir_name):

        # 2.3.3 Open the file

        f = open(os.path.join(dir_name, fname))

        # 2.3.4 Append its text to texts[]

        texts.append(f.read())

        f.close()

        # 2.3.5 And if the directory was 'neg'

        if label_type == 'neg':

            # 2.3.6 All comments are negative

            labels.append(0)

        else:

            labels.append(1)

type(texts)      # list

type(texts[5])   # 5th element is a str. It is a list of strings

texts[5]         # Read this review

labels[5]        # Its label is negative

len(texts)       # 2000 comments

labels[:5]       # look at labels

# 3. Start processing



# 3.1 Create object to tokenize comments. Pick up

#     top 'max_words' tokens (by frequency of occurrence)

#     Its full syntax is:

#   Tokenizer(num_words=None,     # How many top-freq words to take

#             filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~#',  # Filter out these

#             lower=True,         # Convert to lower

#             split=' ',          # Split words here to create tokens

#             char_level=False,   # Word-level and not characater level parsing

#             oov_token=None,     # What token to assign when Out-of-vocab word is seen

#             document_count=0    # Cannot be changed

#             )



tokenizer = Tokenizer(num_words = max_words)

# 3.2 Train the object on full list

tokenizer.fit_on_texts(texts)

# 3.3 Tokenize texts. Actual transformation occurs here

sequences = tokenizer.texts_to_sequences(texts)

# 3.2 Train the object on full list

tokenizer.fit_on_texts(texts)

# 3.3 Tokenize texts. Actual transformation occurs here

sequences = tokenizer.texts_to_sequences(texts)
len(sequences) 

#sequences[0] 

#texts[0] 
# 3.5 Which one of the comments have less than 100 words

for i in np.arange(len(sequences)):

    l = len(sequences[i])

    if (l < 100):

        print(i)     # 45 (only 1 comment)



len(sequences[45])   # 16
# Get word-to-integer dictionary?

word_index = tokenizer.word_index

#word_index
# 3.7 Which are the top few most frequent words

for i in word_index:

    if word_index[i] < 10:

        print(i)



# 3.8

len(word_index )
# 3.9 Make all sequences of equal length



data = pad_sequences(sequences,               # A list of lists

                    maxlen = maxlen_comment,  # MAx length of a sequence

                    padding = "pre",          # pad before

                    truncating = "pre"        # Remove values larger than sequence length

                    )      # Returns numpy array with shape (len(sequences), maxlen)

data[45]      

len(data[45])    # 100

type(data)       # numpy.ndarray

data.shape       # (2000 X 100)
# 3.91 And what about labels?

type(labels)      # list

labels = np.asarray(labels)   # Transform list to array

labels
# 4.0 Shuffle comments randomly

# 4.0.1 First generate a simple sequence

indices = np.arange(data.shape[0])

#4.0.2 Shuffle this sequence

np.random.shuffle(indices)

indices
# 4.1 Extract data and corresponding labels

data = data[indices, ]

labels = labels[indices]

# 4.2 Prepare train and validation data

X_train = data[:training_samples, ]

y_train = labels[:training_samples]



x_val = data[training_samples:training_samples+validation_samples,]

y_val = labels[training_samples:training_samples+validation_samples]
# 5.1 Put all glove vectors in a dictionary

embeddings_index = {}
#dir_path = os.path.dirname(os.path.realpath("glove.6B.50D.txt"))#

#dir_path

#os.listdir('/kaggle/input/glove-6b50d-anagha')  


# 5.2 Start reading the file line by line

#f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'), 'r')

f=open('/kaggle/input/glove-6b50d-anagha/glove.6B.50d.txt','r')

for line in f:

    # 5.2.1 Split each line on ' '

    values = line.split()

    # 5.2. The first token is the word

    word = values[0]

    # 5.2.3 Rest all numbers in the line are vectors for this word

    vectors = np.asarray(values[1:], dtype = 'float32')

    # 5.2.4 Update embeddings dictionary

    embeddings_index[word] = vectors



f.close()
# 5.3 Have a look at few vectors

embeddings_index['at']

len(embeddings_index['at'])            # 50

len(embeddings_index)                   # 400000
# 5.3 Have a look at few vectors

embeddings_index['the']

len(embeddings_index['the'])            # 50

len(embeddings_index)   
# 6. We need to transform this dictionary into

#    a matrix of shape max_words X vector-size

#    OR, as: 10000 X 50 matrix. In this matrix

#    1st row is a vector for 1st word, IInd row

#    for IInd word and so on. The sequence of

#    words is as in word_index:



vector_size = 50

# 6.1 Get an all zero matrix of equal dimension

embedding_matrix = np.zeros(shape = (max_words, vector_size))

embedding_matrix.shape
# 7. Now fill embedding matrix with vectors

# word_index.items() is a tuple of form ('the', )



type(word_index)                # Dictionary

word_index.items()

type(word_index.items())        # dict_items: One iterable collection object

list(word_index.items())[0:5]   # key-value tuple
for word, i in word_index.items():

    # 8.1 If token value is less than 10000

    #     Token coding is as per frequency. Higher

    #     frequency means higher rank. Rank of

    #     1 is highest. Word_index of ('the', 5)

    #     means, 'the' is ranked 5th.

    if i < max_words:

        # 8.2 For the particluar key (ie 'word') get value-vector

        embedding_vector = embeddings_index.get(word)

        # 8.3 Store the vector in the matrix

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector
# 9. Have a look

embedding_matrix

embedding_matrix.shape    # 10000 X 50
# 10. Finally develop the model.

#     Keep it simplest possible ie no lstm or RNN layer

#     after Embedding layer as in file:





model = Sequential()

#  10.1  Adding EMbedding Layer                 10000        50                       100

model.add(Embedding(max_words,vector_size, input_length = maxlen_comment))

model.summary()

# 10.2 Seed embedding layers with glove weights (10000 X 50)

model.layers[0].name

model.layers[0].set_weights([embedding_matrix])

# 10.3 And let weghts in this layer not change with back-propagation

model.layers[0].trainable = False
# 11 Adding some layers by self

# 11.1 Adding LSTMlayer to model with number of cells equal to 'embedding_vector_length'

#     And number of neurons in hidden layer equal to no_of_neurons_in_hidden_state



no_of_neurons_in_hidden_state = 50

# 11.1.1

from keras.layers import LSTM

# Adding LSTM Layer

model.add(LSTM(

               units = no_of_neurons_in_hidden_state,

               return_sequences = False

              )

          )

model.summary()
#11.2 Add Drop Out Layer

from keras.layers import Dropout

# Adding Dropout Layer to increase 

model.add(Dropout(0.2))

model.summary()

# 10.2

model.add(Dense(32, activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))

model.summary()
# 13. compile the model

model.compile(optimizer = 'rmsprop',

              loss = 'binary_crossentropy',

              metrics = ['acc'])
# 14. Train the model for 30 epochs

#     With just 400 training samples we get 59% validation accuracy

history = model.fit(X_train,y_train,

                    epochs = 100,

                    batch_size = 32,

                    validation_data = (x_val,y_val)

                    )
# 15 Draw the learning curve

import matplotlib.pyplot as plt

def plot_learning_curve():

    val_acc = history.history['val_acc']

    tr_acc=history.history['acc']

    epochs = range(1, len(val_acc) +1)

    plt.plot(epochs,val_acc, 'b', label = "Validation accu")

    plt.plot(epochs, tr_acc, 'r', label = "Training accu")

    plt.title("Training and validation accuracy")

    plt.legend()

    plt.show()

# 15.1

plot_learning_curve()