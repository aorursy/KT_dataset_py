# import necessary libraries

import warnings

warnings.filterwarnings("ignore")



import numpy as np



from matplotlib import pyplot as plt



from nltk.corpus import brown

from nltk.corpus import treebank

from nltk.corpus import conll2000



import seaborn as sns



from gensim.models import KeyedVectors



from keras.preprocessing.sequence import pad_sequences

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Embedding

from keras.layers import Dense, Input

from keras.layers import TimeDistributed

from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN

from keras.models import Model

from keras.preprocessing.text import Tokenizer



from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
# load POS tagged corpora from NLTK

treebank_corpus = treebank.tagged_sents(tagset='universal')

brown_corpus = brown.tagged_sents(tagset='universal')

conll_corpus = conll2000.tagged_sents(tagset='universal')

tagged_sentences = treebank_corpus + brown_corpus + conll_corpus
# let's look at the data

tagged_sentences[7]
X = [] # store input sequence

Y = [] # store output sequence



for sentence in tagged_sentences:

    X_sentence = []

    Y_sentence = []

    for entity in sentence:         

        X_sentence.append(entity[0])  # entity[0] contains the word

        Y_sentence.append(entity[1])  # entity[1] contains corresponding tag

        

    X.append(X_sentence)

    Y.append(Y_sentence)
num_words = len(set([word.lower() for sentence in X for word in sentence]))

num_tags   = len(set([word.lower() for sentence in Y for word in sentence]))
print("Total number of tagged sentences: {}".format(len(X)))

print("Vocabulary size: {}".format(num_words))

print("Total number of tags: {}".format(num_tags))
# let's look at first data point

# this is one data point that will be fed to the RNN

print('sample X: ', X[0], '\n')

print('sample Y: ', Y[0], '\n')
# In this many-to-many problem, the length of each input and output sequence must be the same.

# Since each word is tagged, it's important to make sure that the length of input sequence equals the output sequence

print("Length of first input sequence  : {}".format(len(X[0])))

print("Length of first output sequence : {}".format(len(Y[0])))
# encode X



word_tokenizer = Tokenizer()                      # instantiate tokeniser

word_tokenizer.fit_on_texts(X)                    # fit tokeniser on data

X_encoded = word_tokenizer.texts_to_sequences(X)  # use the tokeniser to encode input sequence
# encode Y



tag_tokenizer = Tokenizer()

tag_tokenizer.fit_on_texts(Y)

Y_encoded = tag_tokenizer.texts_to_sequences(Y)
# look at first encoded data point



print("** Raw data point **", "\n", "-"*100, "\n")

print('X: ', X[0], '\n')

print('Y: ', Y[0], '\n')

print()

print("** Encoded data point **", "\n", "-"*100, "\n")

print('X: ', X_encoded[0], '\n')

print('Y: ', Y_encoded[0], '\n')
# make sure that each sequence of input and output is same length



different_length = [1 if len(input) != len(output) else 0 for input, output in zip(X_encoded, Y_encoded)]

print("{} sentences have disparate input-output lengths.".format(sum(different_length)))
# check length of longest sentence

lengths = [len(seq) for seq in X_encoded]

print("Length of longest sentence: {}".format(max(lengths)))
sns.boxplot(lengths)

plt.show()
# Pad each sequence to MAX_SEQ_LENGTH using KERAS' pad_sequences() function. 

# Sentences longer than MAX_SEQ_LENGTH are truncated.

# Sentences shorter than MAX_SEQ_LENGTH are padded with zeroes.



# Truncation and padding can either be 'pre' or 'post'. 

# For padding we are using 'pre' padding type, that is, add zeroes on the left side.

# For truncation, we are using 'post', that is, truncate a sentence from right side.



MAX_SEQ_LENGTH = 100  # sequences greater than 100 in length will be truncated



X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
# print the first sequence

print(X_padded[0], "\n"*3)

print(Y_padded[0])
# assign padded sequences to X and Y

X, Y = X_padded, Y_padded
# word2vec



path = '../input/wordembeddings/GoogleNews-vectors-negative300.bin'





# load word2vec using the following function present in the gensim library

word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
# word2vec effectiveness

word2vec.most_similar(positive = ["King", "Woman"], negative = ["Man"])
# assign word vectors from word2vec model



EMBEDDING_SIZE  = 300  # each word in word2vec model is represented using a 300 dimensional vector

VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1



# create an empty embedding matix

embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))



# create a word to index dictionary mapping

word2id = word_tokenizer.word_index



# copy vectors from word2vec model to the words present in corpus

for word, index in word2id.items():

    try:

        embedding_weights[index, :] = word2vec[word]

    except KeyError:

        pass
# check embedding dimension

print("Embeddings shape: {}".format(embedding_weights.shape))
# let's look at an embedding of a word

embedding_weights[word_tokenizer.word_index['joy']]
# use Keras' to_categorical function to one-hot encode Y

Y = to_categorical(Y)
# print Y of the first output sequqnce

print(Y.shape)
# split entire data into training and testing sets

TEST_SIZE = 0.15

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=4)
# split training data into training and validation sets

VALID_SIZE = 0.15

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=VALID_SIZE, random_state=4)
# print number of samples in each set

print("TRAINING DATA")

print('Shape of input sequences: {}'.format(X_train.shape))

print('Shape of output sequences: {}'.format(Y_train.shape))

print("-"*50)

print("VALIDATION DATA")

print('Shape of input sequences: {}'.format(X_validation.shape))

print('Shape of output sequences: {}'.format(Y_validation.shape))

print("-"*50)

print("TESTING DATA")

print('Shape of input sequences: {}'.format(X_test.shape))

print('Shape of output sequences: {}'.format(Y_test.shape))
# total number of tags

NUM_CLASSES = Y.shape[2]
# create architecture



rnn_model = Sequential()



# create embedding layer - usually the first layer in text problems

rnn_model.add(Embedding(input_dim     =  VOCABULARY_SIZE,         # vocabulary size - number of unique words in data

                        output_dim    =  EMBEDDING_SIZE,          # length of vector with which each word is represented

                        input_length  =  MAX_SEQ_LENGTH,          # length of input sequence

                        trainable     =  False                    # False - don't update the embeddings

))



# add an RNN layer which contains 64 RNN cells

rnn_model.add(SimpleRNN(64, 

              return_sequences=True  # True - return whole sequence; False - return single output of the end of the sequence

))



# add time distributed (output at each sequence) layer

rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
rnn_model.compile(loss      =  'categorical_crossentropy',

                  optimizer =  'adam',

                  metrics   =  ['acc'])
# check summary of the model

rnn_model.summary()
rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history

plt.plot(rnn_training.history['acc'])

plt.plot(rnn_training.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc="lower right")

plt.show()
# create architecture



rnn_model = Sequential()



# create embedding layer - usually the first layer in text problems

rnn_model.add(Embedding(input_dim     =  VOCABULARY_SIZE,         # vocabulary size - number of unique words in data

                        output_dim    =  EMBEDDING_SIZE,          # length of vector with which each word is represented

                        input_length  =  MAX_SEQ_LENGTH,          # length of input sequence

                        trainable     =  True                     # True - update the embeddings while training

))



# add an RNN layer which contains 64 RNN cells

rnn_model.add(SimpleRNN(64, 

              return_sequences=True  # True - return whole sequence; False - return single output of the end of the sequence

))



# add time distributed (output at each sequence) layer

rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
rnn_model.compile(loss      =  'categorical_crossentropy',

                  optimizer =  'adam',

                  metrics   =  ['acc'])
# check summary of the model

rnn_model.summary()
rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history

plt.plot(rnn_training.history['acc'])

plt.plot(rnn_training.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc="lower right")

plt.show()
# create architecture



rnn_model = Sequential()



# create embedding layer - usually the first layer in text problems

rnn_model.add(Embedding(input_dim     =  VOCABULARY_SIZE,         # vocabulary size - number of unique words in data

                        output_dim    =  EMBEDDING_SIZE,          # length of vector with which each word is represented

                        input_length  =  MAX_SEQ_LENGTH,          # length of input sequence

                        weights       = [embedding_weights],      # word embedding matrix

                        trainable     =  True                     # True - update the embeddings while training

))



# add an RNN layer which contains 64 RNN cells

rnn_model.add(SimpleRNN(64, 

              return_sequences=True  # True - return whole sequence; False - return single output of the end of the sequence

))



# add time distributed (output at each sequence) layer

rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
rnn_model.compile(loss      =  'categorical_crossentropy',

                  optimizer =  'adam',

                  metrics   =  ['acc'])
# check summary of the model

rnn_model.summary()
rnn_training = rnn_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history

plt.plot(rnn_training.history['acc'])

plt.plot(rnn_training.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc="lower right")

plt.show()
# create architecture



lstm_model = Sequential()

lstm_model.add(Embedding(input_dim     = VOCABULARY_SIZE,         # vocabulary size - number of unique words in data

                         output_dim    = EMBEDDING_SIZE,          # length of vector with which each word is represented

                         input_length  = MAX_SEQ_LENGTH,          # length of input sequence

                         weights       = [embedding_weights],     # word embedding matrix

                         trainable     = True                     # True - update embeddings_weight matrix

))

lstm_model.add(LSTM(64, return_sequences=True))

lstm_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
lstm_model.compile(loss      =  'categorical_crossentropy',

                   optimizer =  'adam',

                   metrics   =  ['acc'])
# check summary of the model

lstm_model.summary()
lstm_training = lstm_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history

plt.plot(lstm_training.history['acc'])

plt.plot(lstm_training.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc="lower right")

plt.show()
# create architecture



gru_model = Sequential()

gru_model.add(Embedding(input_dim     = VOCABULARY_SIZE,

                        output_dim    = EMBEDDING_SIZE,

                        input_length  = MAX_SEQ_LENGTH,

                        weights       = [embedding_weights],

                        trainable     = True

))

gru_model.add(GRU(64, return_sequences=True))

gru_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
gru_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
# check summary of model

gru_model.summary()
gru_training = gru_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history

plt.plot(gru_training.history['acc'])

plt.plot(gru_training.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc="lower right")

plt.show()
# create architecture



bidirect_model = Sequential()

bidirect_model.add(Embedding(input_dim     = VOCABULARY_SIZE,

                             output_dim    = EMBEDDING_SIZE,

                             input_length  = MAX_SEQ_LENGTH,

                             weights       = [embedding_weights],

                             trainable     = True

))

bidirect_model.add(Bidirectional(LSTM(64, return_sequences=True)))

bidirect_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
bidirect_model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
# check summary of model

bidirect_model.summary()
bidirect_training = bidirect_model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_validation, Y_validation))
# visualise training history

plt.plot(bidirect_training.history['acc'])

plt.plot(bidirect_training.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc="lower right")

plt.show()
loss, accuracy = rnn_model.evaluate(X_test, Y_test, verbose = 1)

print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))
loss, accuracy = lstm_model.evaluate(X_test, Y_test, verbose = 1)

print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))
loss, accuracy = gru_model.evaluate(X_test, Y_test, verbose = 1)

print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))
loss, accuracy = bidirect_model.evaluate(X_test, Y_test, verbose = 1)

print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))