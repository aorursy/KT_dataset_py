import os

import sys

import numpy as np

import keras

import shap

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D

from keras.layers import MaxPool1D

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam

import warnings

warnings.filterwarnings('ignore')
# check dataset is added properly 

!ls '../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'
TEXT_DATA_DIR = r'../input/20-newsgroup-original/20_newsgroup/20_newsgroup/'

GLOVE_DIR = r'../input/glove6b/'

# make the max word length to be constant

MAX_WORDS = 10000

MAX_SEQUENCE_LENGTH = 100

VALIDATION_SPLIT = 0.20

# the dimension of vectors to be used

EMBEDDING_DIM = 100

# filter sizes of the different conv layers 

filter_sizes = [3,4,5]

num_filters = 512

embedding_dim = 100

drop = 0.5

batch_size = 30

epochs = 2
texts = []  # list of text samples

labels_index = {}  # dictionary mapping label name to numeric id

labels = []  # list of label ids



for name in sorted(os.listdir(TEXT_DATA_DIR)):

    path = os.path.join(TEXT_DATA_DIR, name)

    if os.path.isdir(path):

        label_id = len(labels_index)

        labels_index[name] = label_id

        for fname in sorted(os.listdir(path)):

            if fname.isdigit():

                fpath = os.path.join(path, fname)

                if sys.version_info < (3,):

                    f = open(fpath)

                else:

                    f = open(fpath, encoding='latin-1')

                t = f.read()

                i = t.find('\n\n')  # skip header

                if 0 < i:

                    t = t[i:]

                texts.append(t)

                f.close()

                labels.append(label_id)



print(labels_index)

print('Found %s texts.' % len(texts))
tokenizer  = Tokenizer(num_words = MAX_WORDS)

tokenizer.fit_on_texts(texts)

sequences =  tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

print("Unique words : {}".format(len(word_index)))



data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)



# keep original clf value

labels_clf = labels

# transform label to one hot encoding

labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)

print('Shape of label tensor:', labels.shape)

print(labels)
# split the data into a training set and a validation set

indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])



x_train = data[:-nb_validation_samples]

y_train = labels[:-nb_validation_samples]

x_test = data[-nb_validation_samples:]

y_test = labels[-nb_validation_samples:]
embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))

for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector
from keras.layers import Embedding



embedding_layer = Embedding(len(word_index) + 1,

                            EMBEDDING_DIM,

                            weights = [embedding_matrix],

                            input_length = MAX_SEQUENCE_LENGTH,

                            trainable = False)
inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding = embedding_layer(inputs)



print(embedding.shape)

reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedding)

print(reshape.shape)



conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)



maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)

maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)

maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)



concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

flatten = Flatten()(concatenated_tensor)

dropout = Dropout(drop)(flatten)

output = Dense(units=20, activation='softmax')(dropout)



# this creates a model that includes

model = Model(inputs=inputs, outputs=output)



checkpoint = ModelCheckpoint('weights_cnn_sentece.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)



model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print("Traning Model...")

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(x_test, y_test))
# select a set of samples to take an expectation over

distrib_samples = x_train[:100]

session = keras.backend.tensorflow_backend.get_session()

# session had to be manually specified

# otherwise looked for Keras.._SESSION ct. which doesn't exist!

explainer = shap.DeepExplainer(model, distrib_samples, session)

num_explanations = 10
shap_values = explainer.shap_values(x_test[:num_explanations])
num2word = {}

for w in word_index.keys():

    num2word[word_index[w]] = w

x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i]))) for i in range(10)])

shap.summary_plot(shap_values, feature_names = list(num2word.values()), class_names = list(labels_index.keys()),)
# init the JS visualization code

shap.initjs()

# create dict to invert word_idx k,v order

num2word = {}

for w in word_index.keys():

    num2word[word_index[w]] = w

x_test_words = np.stack([np.array(list(map(lambda x: num2word.get(x, "NONE"), x_test[i]))) for i in range(10)])



# plot the explanation of a given prediction

class_num = 9

input_num = 5

shap.force_plot(explainer.expected_value[class_num], shap_values[class_num][input_num], x_test_words[input_num])
# reverse idx for labels

num2label = {}

for w in labels_index.keys():

    num2label[labels_index[w]] = w

x_test_labels = np.stack([np.array(list(map(lambda x: num2label.get(x, "NONE"), x_test[i]))) for i in range(10)])
# generate 10 predictions

y_pred = model.predict(x_test[:10])

sample = 8

true_class = list(y_test[sample]).index(1)

pred_class = list(y_pred[sample]).index(max(y_pred[sample]))

# one hot encoded result

print(f'Predicted vector is {y_pred[sample]} = Class {pred_class} = {num2label[pred_class]}')

# filter padding words

print(f'Input features/words:')

print(x_test_words[sample][np.where(x_test_words[sample] != 'NONE')])

print(f'True class is {true_class} = {num2label[true_class]}')

max_expected = list(explainer.expected_value).index(max(explainer.expected_value))

print(f'Explainer expected value is {explainer.expected_value}, i.e. class {max_expected} is the most common.')
kernel_explainer = shap.KernelExplainer(model.predict, distrib_samples)

kernel_shap_values = kernel_explainer.shap_values(x_test[:num_explanations])
# plot the explanation of a given prediction

class_num = 13

input_num = 8

shap.force_plot(kernel_explainer.expected_value[class_num], kernel_shap_values[class_num][input_num], x_test_words[input_num])
# explanations of the output for the given class 

# y center value is base rate for the given background data

shap.force_plot(kernel_explainer.expected_value[class_num], kernel_shap_values[class_num], x_test_words[:10])