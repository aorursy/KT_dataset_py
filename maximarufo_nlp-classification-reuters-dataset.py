import keras
from keras.datasets import reuters
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Embedding, Dropout, Activation, LSTM, Bidirectional
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import collections
# First we get the whole dataset

# save np.load
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

(x_tot, y_tot), (_, _) = reuters.load_data(
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=2)

# Labels in dataset (y_tot) are indexes. From https://github.com/keras-team/keras/issues/12072, we have read the actual strings
classidx =  {'copper': 6, 'livestock': 28, 'gold': 25, 'money-fx': 19, 'ipi': 30, 'trade': 11, 'cocoa': 0, 'iron-steel': 31, 'reserves': 12, 'tin': 26, 'zinc': 37, 'jobs': 34, 'ship': 13, 'cotton': 14, 'alum': 23, 'strategic-metal': 27, 'lead': 45, 'housing': 7, 'meal-feed': 22, 'gnp': 21, 'sugar': 10, 'rubber': 32, 'dlr': 40, 'veg-oil': 2, 'interest': 20, 'crude': 16, 'coffee': 9, 'wheat': 5, 'carcass': 15, 'lei': 35, 'gas': 41, 'nat-gas': 17, 'oilseed': 24, 'orange': 38, 'heat': 33, 'wpi': 43, 'silver': 42, 'cpi': 18, 'earn': 3, 'bop': 36, 'money-supply': 8, 'hog': 44, 'acq': 4, 'pet-chem': 39, 'grain': 1, 'retail': 29}
# Number of samples and classes
num_classes = max(y_tot) - min(y_tot) + 1
print('# of Samples: {}'.format(len(x_tot)))
print('# of Classes: {}'.format(num_classes))

# Reverse dictionary to see words instead of integers
# Note that the indices are offset by 3 because 0, 1, and 2 are reserved indices for “padding,” “start of sequence,” and “unknown.”
word_to_wordidx = reuters.get_word_index(path="reuters_word_index.json")
word_to_wordidx = {k:(v+2) for k,v in word_to_wordidx.items()}
word_to_wordidx["<PAD>"] = 0
word_to_wordidx["<START>"] = 1
word_to_wordidx["<UNK>"] = 2
wordidx_to_word = {value:key for key,value in word_to_wordidx.items()}
classidx_to_class = {value:key for key,value in classidx.items()}

# Number of words
print('# of Words (including PAD, START and UNK): {}'.format(len(word_to_wordidx)))
# Now we decode the newswires, using the wordidx_to_word dictionary
def decode_newswire (sample):
    """
    Decodes a Newswire
    
    Arguments:
    sample -- one of the samples in the reuters dataset

    Returns:
    decode_newswire -- a string representing the newswires
    """
    return ' '.join([wordidx_to_word[wordidx] for wordidx in sample])
decoded_newswires = [decode_newswire(sample) for sample in x_tot]
# We print some examples to check if everything is ok
example_num = 1234
print ("ENCODED: ", x_tot[example_num])
print("\nNEWSWIRE: ", decoded_newswires[example_num])
print ("\nCLASS: ", classidx_to_class[y_tot[example_num]])
# We do some statistcs of the length of the documents

documents_word_lenght = [len (sample) for sample in x_tot]
documents_ch_lenght = [len (sample) for sample in decoded_newswires]

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.suptitle('Document Length Distribution')
f.set_size_inches((20, 5))
ax1.hist(documents_word_lenght, bins=100)
ax1.set_title('In words')
ax2.hist(documents_ch_lenght, bins=100)
ax2.set_title('In characters')

plt.show()

print('Mean Lenght (in words): {}'.format(np.mean(documents_word_lenght)))
print('Mean Lenght (in characters): {}'.format(np.mean(documents_ch_lenght)))
print('Max Lenght (in words): {}'.format(np.max(documents_word_lenght)))
print('Max Lenght (in characters): {}'.format(np.max(documents_ch_lenght)))
print('Min Lenght (in words): {}'.format(np.min(documents_word_lenght)))
print('Min Lenght (in characters): {}'.format(np.min(documents_ch_lenght)))
# Class Distribution

y_hist, y_bin_edges =  np.histogram(y_tot, bins=num_classes)

sorted_num_of_ocurrences = np.sort(y_hist)
sorted_classes = [classidx_to_class[key] for key in np.argsort(y_hist)]

plt.figure(num=None, figsize=(10, 10), dpi=80)
plt.barh(sorted_classes, sorted_num_of_ocurrences, align='center')
plt.yticks(np.arange(num_classes), sorted_classes)
plt.xlabel('Number of Ocurrences')
plt.title('Class Distribution')

ax = plt.gca()
for i, v in enumerate(sorted_num_of_ocurrences):
    ax.text(v + 3, i-0.25, str(v), color='blue')

plt.show()
# Words Distribution

top = 50

words_with_repetition = []
for x in x_tot: 
    words_with_repetition.extend(x[1:])

x_hist, x_bin_edges =  np.histogram(words_with_repetition, bins=len(word_to_wordidx), range=(0,len(word_to_wordidx)-1))

sorted_num_of_ocurrences = np.sort(x_hist)[-top:]
sorted_words = [wordidx_to_word[key] for key in np.argsort(x_hist)[-top:]]

plt.figure(num=None, figsize=(10, 10), dpi=80)
plt.barh(sorted_words, sorted_num_of_ocurrences, align='center')
plt.yticks(np.arange(top), sorted_words)
plt.xlabel('Number of Ocurrences')
plt.title('Top {} Words Distribution'.format(top))

ax = plt.gca()
for i, v in enumerate(sorted_num_of_ocurrences):
    ax.text(v + 3, i-0.25, str(v), color='blue')

plt.show()
num_sentences_with_label_in_it = 0

def count_num_sentences_with_label_in_it(x_tot):
    """
    Generator for counting number of sentences which contain the class word within the text
    
    Arguments:
    sample -- total dataset

    Returns:
    decode_newswire -- generator with True or False
    """
    for x in x_tot:
        words = [wordidx_to_word[wordidx] for wordidx in x]
        for label in classidx:
            yield (label in words)
                
num_sentences_with_label_in_it = np.sum(count_num_sentences_with_label_in_it(x_tot))

print('{:.2f}% of the examples have the label within the text.'.format(100*num_sentences_with_label_in_it/len(x_tot)))
top_classes = sorted_classes[-10:]
print("Top 10 most frequent classes: ", top_classes)
print("Dataset will now be filtered by those classes")
top_indexes = [classidx[label] for label in top_classes]
to_keep = [i for i,x in enumerate(y_tot) if x in top_indexes]
y_tot_filtered = y_tot[to_keep]
x_tot_filtered = x_tot[to_keep]
print('# of Samples kept: {}'.format(len(x_tot_filtered)))
print("We will only keep newswires with max 1000 words")
documents_word_lenght = [len (sample) for sample in x_tot_filtered]
to_keep = [idx for idx, value in enumerate(documents_word_lenght) if value <= 1000]
y_tot_filtered = y_tot_filtered[to_keep]
x_tot_filtered = x_tot_filtered[to_keep]
print('# of Samples kept: {}'.format(len(x_tot_filtered)))
# Tokenizing
num_words_to_tokenize = len(word_to_wordidx)
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=num_words_to_tokenize)
x_tot_matrix = tokenizer.sequences_to_matrix(x_tot_filtered, mode='binary')
y_tot_categorical = keras.utils.to_categorical(y_tot_filtered, num_classes)
print ("When tokenizing the examples, we are just marking which words appear in the sentence. Therefore, we lose the information about the sequence itself, that is, the information which is implied in the order of the words.")
# Test Split
test_split = 0.1
test_num = round(len(x_tot_filtered)*test_split)
x_test_matrix = x_tot_matrix[:test_num]
x_test_seq = x_tot_filtered[:test_num]
x_train_matrix = x_tot_matrix[test_num:]
x_train_seq = x_tot_filtered[test_num:]
y_test_cat = y_tot_categorical[:test_num]
y_train_cat = y_tot_categorical[test_num:]
print ("We keep aside some examples for testing.")
# Padding
documents_word_lenght = [len (sample) for sample in x_tot_filtered]
maxlen = np.max(documents_word_lenght)
x_train_pad = pad_sequences(x_train_seq, maxlen=maxlen)
x_test_pad =  pad_sequences(x_test_seq, maxlen=maxlen)
print ("Padding is for making all sequences of the same length, so we can then use them as inputs of neural networks.")
model = Sequential()
model.add(Dense(512, input_shape=(num_words_to_tokenize,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 5
validation_split = 0.1

start = time.clock()
history = model.fit(x_train_matrix, y_train_cat, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=validation_split)
end = time.clock()
print('Time spent:', end-start)
score = model.evaluate(x_test_matrix, y_test_cat, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Validation Curves')
f.set_size_inches((20, 5))

# VALIDATION LOSS curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

## VALIDATION ACCURACY curves
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()
embedded_dim = 32
conv_filters = 32
conv_kernel = 3

model = Sequential()
model.add(Embedding(input_dim=len(word_to_wordidx), output_dim=embedded_dim, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

batch_size = 32
epochs = 5
validation_split = 0.1

start = time.clock()
history = model.fit(x_train_pad, y_train_cat, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_split=validation_split)
end = time.clock()
print('Time spent:', end-start)
score = model.evaluate(x_test_pad, y_test_cat, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Validation Curves')
f.set_size_inches((20, 5))

# VALIDATION LOSS curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

## VALIDATION ACCURACY curves
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()
embedded_dim = 32
conv_filters = 32
conv_kernel = 3
pool_size=2

model = Sequential()
model.add(Embedding(input_dim=len(word_to_wordidx), output_dim=embedded_dim, input_length=maxlen))
model.add(Convolution1D(filters=conv_filters, kernel_size=conv_kernel, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

batch_size = 32
epochs = 5
validation_split = 0.1

start = time.clock()
history = model.fit(x_train_pad, y_train_cat, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_split=validation_split)
end = time.clock()
print('Time spent:', end-start)
score = model.evaluate(x_test_pad, y_test_cat, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Validation Curves')
f.set_size_inches((20, 5))

# VALIDATION LOSS curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

## VALIDATION ACCURACY curves
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()
# We define two functions that will use later

def loadGloveModel(gloveFile):
    """
    Loads GloVe Model
    
    Arguments:
    gloveFile -- path to the glove file

    Returns:
    model -- a word_to_vec_map, where keys are words, and values are vectors (represented by arrays)
    """
    
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def pretrained_embedding_layer(word_to_vec_map, word_to_wordidx):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_wordidx -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_wordidx) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros(((vocab_len, emb_dim)))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_wordidx.items():
        if word in word_to_vec_map:
            emb_matrix[index, :] = word_to_vec_map[word]
        else:
            emb_matrix[index, :] = word_to_vec_map["random"]  #just to set something when work is not in word_to_vec_map

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
word_to_vec_map = loadGloveModel("../input/glove6b50dtxt/glove.6B.50d.txt")

sentence_indices = Input(shape=(maxlen,), dtype='int32')
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_wordidx)
embeddings = embedding_layer(sentence_indices)
X = LSTM(32,return_sequences=True)(embeddings)
X = Dropout(0.5)(X)
X = LSTM(32,return_sequences=False)(X)
X = Dropout(0.5)(X)
X = Dense(num_classes, activation="softmax")(X)
model = Model(inputs = sentence_indices, outputs = X)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 5
validation_split = 0.1

start = time.clock()
history = model.fit(x_train_pad, y_train_cat, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=validation_split)
end = time.clock()
print('Time spent:', end-start)
score = model.evaluate(x_test_pad, y_test_cat, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Validation Curves')
f.set_size_inches((20, 5))

# VALIDATION LOSS curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

## VALIDATION ACCURACY curves
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()

sentence_indices = Input(shape=(maxlen,), dtype='int32')
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_wordidx)
embeddings = embedding_layer(sentence_indices)
X = Bidirectional(LSTM(32,return_sequences=True))(embeddings)
X = Dropout(0.5)(X)
X = Bidirectional(LSTM(32,return_sequences=False))(X)
X = Dropout(0.5)(X)
X = Dense(num_classes, activation="softmax")(X)
model = Model(inputs = sentence_indices, outputs = X)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 32
epochs = 5
validation_split = 0.1

start = time.clock()
history = model.fit(x_train_pad, y_train_cat, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1, validation_split=validation_split)
end = time.clock()
print('Time spent:', end-start)
score = model.evaluate(x_test_pad, y_test_cat, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Validation Curves')
f.set_size_inches((20, 5))

# VALIDATION LOSS curves
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

## VALIDATION ACCURACY curves
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()
# We define two functions that will use later

def seq_to_vec(seq, word_to_vec_map):
    """
    Sequence to vector
    
    Arguments:
    seq -- a sequence sample of the train set
    word_to_vec_map -- output of loadGloveModel

    Returns:
    mean_vec_for_seq -- The mean of all vectors corresponding to each word in the sequence
    """    
    
    vec = (word_to_vec_map[wordidx_to_word[wordidx]]
                 if wordidx_to_word[wordidx] in word_to_vec_map
                 else
                 np.zeros((len(list(word_to_vec_map.values())[0])))
                 for wordidx in seq )
    
    mean_vec_for_seq = np.sum(vec,0) / len(seq)
    
    return mean_vec_for_seq

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    dot = np.dot(u,v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cosine_similarity = dot/(norm_u*norm_v)    
    return cosine_similarity

def check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold):
    if (( distance > mean_cosine_similarity+std_threshold*std_cosine_similarity )
        or ( distance < mean_cosine_similarity-std_threshold*std_cosine_similarity )):
        print("Rejected")
    else:
        print("Accepted")
    return
gen_acum_vec = (seq_to_vec(seq, word_to_vec_map) for seq in x_train_seq)
sentences_vec_acum = np.sum(gen_acum_vec,0)
sentences_vec_mean = sentences_vec_acum / len(x_train_seq)
cosine_similarity_list = [ cosine_similarity(seq_to_vec(seq, word_to_vec_map), sentences_vec_mean)
              for seq in x_train_seq ]
mean_cosine_similarity = np.mean(cosine_similarity_list)
std_cosine_similarity = np.std(cosine_similarity_list)
print('Mean of cosine similarity {}'.format(mean_cosine_similarity))
print('Std of cosine similarity {}'.format(std_cosine_similarity))
# Devaiation threshold to use
std_threshold = 1.0
# Check if it rejects or accepts an actual sentence of the training
example = 123
distance = cosine_similarity(seq_to_vec(x_train_seq[example], word_to_vec_map), sentences_vec_mean)
check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold)
# Check if it rejects or accepts random sentences, but with words of the training set
random_seq = [123, 1546, 3587, 2671, 189, 36, 875, 124]
distance = cosine_similarity(seq_to_vec(random_seq, word_to_vec_map), sentences_vec_mean)
check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold)

random_seq = [1546, 123, 124, 875, 2671, 36, 3587, 189, 7574, 45, 77, 2457]
distance = cosine_similarity(seq_to_vec(random_seq, word_to_vec_map), sentences_vec_mean)
check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold)

random_seq = [123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123]
distance = cosine_similarity(seq_to_vec(random_seq, word_to_vec_map), sentences_vec_mean)
check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold)
# Check if it rejects or accepts random sentences, with words from an open set
random_sentence = "this is a random sentence to test the algorithm"
random_sentence_splitted = random_sentence.replace(',',' ').split()
vec = [word_to_vec_map[word] for word in random_sentence_splitted]
mean_vec_for_seq = np.sum(vec,0) / len(random_sentence_splitted)
distance = cosine_similarity(mean_vec_for_seq, sentences_vec_mean)
check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold)

random_sentence = "if we create a document by putting random words together, this document must be rejected"
random_sentence_splitted = random_sentence.replace(',',' ').split()
vec = [word_to_vec_map[word] for word in random_sentence_splitted]
mean_vec_for_seq = np.sum(vec,0) / len(random_sentence_splitted)
distance = cosine_similarity(mean_vec_for_seq, sentences_vec_mean)
check_rejection(distance, mean_cosine_similarity, std_cosine_similarity, std_threshold)
def compute_pmi(word_1,word_2, x_tot, word_to_wordidx):
    
    if ((word_1 in word_to_wordidx) and (word_2 in word_to_wordidx)):
        words_with_repetition = []
        for x in x_tot: 
            words_with_repetition.extend(x[1:])
        x_hist, x_bin_edges =  np.histogram(words_with_repetition, bins=len(word_to_wordidx), range=(0,len(word_to_wordidx)-1))

        count_word_1 = x_hist[word_to_wordidx[word_1]]
        count_word_2 = x_hist[word_to_wordidx[word_2]]
        co_occurrences = (True for x in x_tot for i in range(len(x)-1) if (word_to_wordidx[word_1], word_to_wordidx[word_2])==(x[i],x[i+1]))
        count_co_occurrences = np.sum(co_occurrences)
        
        if (count_co_occurrences):
            pmi = np.log(len(words_with_repetition) * count_co_occurrences / (count_word_1 * count_word_2))
            return pmi
        else:
            print ("No co-ocurrences found")
        
    elif word_1 not in word_to_wordidx:
        print ("{} is not in the dictionary".format(word_1))
    elif word_2 not in word_to_wordidx:
        print ("{} is not in the dictionary".format(word_2))  
    
compute_pmi("buenos","aires", x_tot, word_to_wordidx)
compute_pmi("in","at", x_tot, word_to_wordidx)
compute_pmi("buenos","at", x_tot, word_to_wordidx)
top_bigrams_num = 30
bigrams = [(x[i],x[i+1]) for x in x_tot for i in range(1,len(x)-1)] # I skip <START> character
counter = collections.Counter(bigrams)
most_commons = [most_common for most_common in counter.most_common(top_bigrams_num)]
encoded_bigrams = [most_common[0] for most_common in most_commons]
decoded_bigrams = [decode_newswire(most_common[0]) for most_common in most_commons]
bigrams_counts = [most_common[1] for most_common in most_commons]
bigrams_pmi = [compute_pmi(wordidx_to_word[encoded_bigram[0]],wordidx_to_word[encoded_bigram[1]], x_tot, word_to_wordidx) for encoded_bigram in encoded_bigrams]

plt.figure(num=None, figsize=(20, 10), dpi=80)
plt.barh(decoded_bigrams, bigrams_counts, align='center')
plt.yticks(np.arange(top_bigrams_num), decoded_bigrams)
plt.xlabel('Number of Ocurrences')
plt.title('Top {} Bigrams'.format(top_bigrams_num))

plt.gca().invert_yaxis()
ax = plt.gca()
for i, v in enumerate(bigrams_counts):
    ax.text(v , i, "Counts: " + str(v) + "/ PMI: " + "{:2.4f}".format(bigrams_pmi[i]), color='blue')

plt.show()