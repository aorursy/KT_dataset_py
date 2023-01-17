# Load packages

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline



from tensorflow.keras import optimizers

from tensorflow.keras.layers import (Conv1D, Dense, Embedding, Flatten,

                                     GlobalAveragePooling1D, GRU, Input,

                                     LSTM, MaxPooling1D)

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical
df = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv')



text = df['text']

target = np.array(df['target'], dtype=np.int32)



text_test = df_test['text']
target_train, target_val, text_train, text_val = train_test_split(

    target, text, test_size=0.1, random_state=42)
print(f'Number of elements in the train set: {len(target_train)}.')
print(f'Number of elements in the validation set: {len(target_val)}.')
print(f'Number of elements in the test set: {len(text_test)}.')
# Convert train/validation text set to list.

text_train = [text for text in text_train]

text_val = [text for text in text_val]

text_test = [text for text in text_test]
# Define text classifier

text_classifier = make_pipeline(

    TfidfVectorizer(min_df=3, max_df=0.8, ngram_range=(1, 2)),

    LogisticRegression())
%%time

_ = text_classifier.fit(text_train, target_train)
print(f'Percentage of good classified on the train set: {np.round(text_classifier.score(text_train, target_train), 3)}%.')
print(f'Percentage of good classified on the validation set: {np.round(text_classifier.score(text_val, target_val), 3)}%.')
# Prediction the test set

subm = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

subm['target'] = text_classifier.predict(text_test)

subm.to_csv('model_logistic.csv', index=False)
MAX_NB_WORDS = 20000



# Vectorize the text samples into a 2D integer tensor

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, char_level=False)

tokenizer.fit_on_texts(text_train)



sequences = tokenizer.texts_to_sequences(text_train)

sequences_val = tokenizer.texts_to_sequences(text_val)

sequences_test = tokenizer.texts_to_sequences(text_test)



word_index = tokenizer.word_index

print(f'There are {len(word_index)} unique tokens.')
index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())
# Rebuild one tweet

" ".join([index_to_word[i] for i in sequences[0]])
# Length of sequences

seq_len = [len(s) for s in sequences]

print(f'Average length of the sequences: {np.mean(seq_len)}.')

print(f'Max length of the sequences: {np.max(seq_len)}.')
plt.hist(seq_len, bins=30)

plt.show()
MAX_SEQUENCE_LENGTH = np.max(seq_len)



# Pad sequences with 0s

x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)



print(f'Shape of train data tensor: {x_train.shape}')

print(f'Shape of validation data tensor: {x_val.shape}')

print(f'Shape of test data tensor: {x_test.shape}')
y_train = to_categorical(target_train)

y_val = to_categorical(target_val)



print(f'Shape of train label tensor: {y_train.shape}')

print(f'Shape of validation label tensor: {y_val.shape}')
EMBEDDING_DIM = 20

N_CLASSES = 2



# Input a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

embedded_sequences = embedding_layer(sequence_input)



average_layer = GlobalAveragePooling1D()

average_sequences = average_layer(embedded_sequences)



dense_layer = Dense(N_CLASSES, activation='softmax')

sequence_output = dense_layer(average_sequences)



model = Model(sequence_input, sequence_output)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=0.001),

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

                    validation_data=(x_val, y_val),

                    epochs=20, batch_size=128)
fig = plt.figure(figsize=(10, 5))

    

ax1 = fig.add_subplot(121)

ax1.plot(history.history['loss'], label='Loss')

ax1.plot(history.history['val_loss'], label='Validation loss')

ax1.set(title='Model loss', xlabel='Epochs', ylabel='Loss')

ax1.legend()



ax2 = fig.add_subplot(122)

ax2.plot(history.history['accuracy'], label='Accuracy')

ax2.plot(history.history['val_accuracy'], label='Validation accuracy')

ax2.set(title='Model accuracy', xlabel='Epochs', ylabel='Accuracy')

ax2.legend()



plt.show()
# Prediction the test set

subm = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

subm['target'] = np.argmax(model.predict(x_test), axis=1)

subm.to_csv('model_cbow.csv', index=False)
# 1D convolution

EMBEDDING_DIM = 20

N_CLASSES = 2



# Input a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

embedded_sequences = embedding_layer(sequence_input)



# A 1D convolution with 64 output channels

x = Conv1D(64, 4, activation='relu')(embedded_sequences)

# MaxPool divides the length of the sequence by 5

x = MaxPooling1D(5)(x)

# A 1D convolution with 32 output channels

x = Conv1D(32, 2, activation='relu')(x)

# MaxPool divides the length of the sequence by 5

x = MaxPooling1D(5)(x)

x = Flatten()(x)



dense_layer = Dense(N_CLASSES, activation='softmax')

sequence_output = dense_layer(x)



model = Model(sequence_input, sequence_output)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=0.0001),

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

                    validation_data=(x_val, y_val),

                    epochs=20, batch_size=128)
fig = plt.figure(figsize=(10, 5))

    

ax1 = fig.add_subplot(121)

ax1.plot(history.history['loss'], label='Loss')

ax1.plot(history.history['val_loss'], label='Validation loss')

ax1.set(title='Model loss', xlabel='Epochs', ylabel='Loss')

ax1.legend()



ax2 = fig.add_subplot(122)

ax2.plot(history.history['accuracy'], label='Accuracy')

ax2.plot(history.history['val_accuracy'], label='Validation accuracy')

ax2.set(title='Model accuracy', xlabel='Epochs', ylabel='Accuracy')

ax2.legend()



plt.show()
# Prediction the test set

subm = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

subm['target'] = np.argmax(model.predict(x_test), axis=1)

subm.to_csv('model_conv1d.csv', index=False)
# LSTM

EMBEDDING_DIM = 20

N_CLASSES = 2



# Input a sequence of MAX_SEQUENCE_LENGTH integers

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=True)

embedded_sequences = embedding_layer(sequence_input)



# A 1D convolution with 64 output channels

x = Conv1D(64, 4, activation='relu')(embedded_sequences)

# MaxPool divides the length of the sequence by 5

x = MaxPooling1D(5)(x)

# A 1D convolution with 32 output channels

x = Conv1D(32, 2, activation='relu')(x)

# MaxPool divides the length of the sequence by 5

x = MaxPooling1D(5)(x)

x = LSTM(32)(x)



dense_layer = Dense(N_CLASSES, activation='softmax')

sequence_output = dense_layer(x)



model = Model(sequence_input, sequence_output)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=0.0001),

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

                    validation_data=(x_val, y_val),

                    epochs=20, batch_size=128)
fig = plt.figure(figsize=(10, 5))

    

ax1 = fig.add_subplot(121)

ax1.plot(history.history['loss'], label='Loss')

ax1.plot(history.history['val_loss'], label='Validation loss')

ax1.set(title='Model loss', xlabel='Epochs', ylabel='Loss')

ax1.legend()



ax2 = fig.add_subplot(122)

ax2.plot(history.history['accuracy'], label='Accuracy')

ax2.plot(history.history['val_accuracy'], label='Validation accuracy')

ax2.set(title='Model accuracy', xlabel='Epochs', ylabel='Accuracy')

ax2.legend()



plt.show()
# Prediction the test set

subm = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

subm['target'] = np.argmax(model.predict(x_test), axis=1)

subm.to_csv('model_lstm.csv', index=False)
%%time

embeddings_index = {}

embeddings_vectors = []

with open('../input/glove-twitter/glove.twitter.27B.25d.txt', 'rb') as file:

    word_idx = 0

    for line in file:

        values = line.decode('utf-8').split()

        word = values[0]

        vector = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = word_idx

        embeddings_vectors.append(vector)

        word_idx = word_idx + 1
inv_idx = {v: k for k, v in embeddings_index.items()}

print(f'Found {word_idx} different words in the file.')
# Check that embedding elements have a common size

vec_list = [idx for (idx, vec) in enumerate(embeddings_vectors) if len(vec) != 25]

for idx in vec_list:

    print(idx)

    embeddings_vectors[idx] = np.append(embeddings_vectors[idx], 0)
#Stack all embeddings in a large numpy array

glove_embeddings = np.vstack(embeddings_vectors)

glove_norms = np.linalg.norm(glove_embeddings, axis=-1, keepdims=True)

glove_embeddings_normed = glove_embeddings / glove_norms

print(f'The shape of the Glove embeddings is: {glove_embeddings.shape}.')
def get_embedding(word):

    idx = embeddings_index.get(word)

    if idx is None:

        return None

    else:

        return glove_embeddings[idx]

    

def get_normed_embedding(word):

    idx = embeddings_index.get(word)

    if idx is None:

        return None

    else:

        return glove_embeddings_normed[idx]
print(f'The embeddings for the word `computer` is {get_embedding("computer")}.')
def get_most_similar(words, top_n=5):

    query_emb = 0

    # If words is a list

    if type(words) == list:

        for word in words:

            query_emb += get_embedding(word)

        query_emb = query_emb / np.linalg.norm(query_emb)

    else:

        query_emb = get_normed_embedding(words)

        

    # Computation of cosine similarities

    cosines_sim = np.dot(glove_embeddings_normed, query_emb)

    

    # Top n most similar indexes corresponding to cosines

    idxs = np.argsort(cosines_sim)[::-1][:top_n]

    

    return [(inv_idx[idx], cosines_sim[idx]) for idx in idxs]
# Get similar words

get_most_similar(['usa', 'france'], top_n=10)
# Compute t-SNE representation

word_emb_tsne = TSNE(perplexity=30).fit_transform(glove_embeddings_normed[:1000])
# Plot t-SNE representation

plt.figure(figsize=(40, 40))

axis = plt.gca()

np.set_printoptions(suppress=True)

plt.scatter(word_emb_tsne[:, 0], word_emb_tsne[:, 1], marker='.', s=1)



for idx in range(1000):

    plt.annotate(inv_idx[idx], xy=(word_emb_tsne[idx, 0], word_emb_tsne[idx, 1]),

                 xytext=(0, 0), textcoords='offset points')

    

plt.show()
EMBEDDING_DIM
EMBEDDING_DIM = 25



# Prepare embedding matrix

nb_words_in_matrix = 0

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for word, i in word_index.items():

    if i >= MAX_NB_WORDS:

        continue

    embedding_vector = get_embedding(word)

    if embedding_vector is not None:

        # Words not found in embedding index will be all-zeros

        embedding_matrix[i] = embedding_vector

        nb_words_in_matrix = nb_words_in_matrix + 1
print(f'There are {nb_words_in_matrix} non-null words in the embedding matrix.')
# Build a layer with pre-trained embeddings

pretrained_embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,

                                       weights=[embedding_matrix],

                                       input_length=MAX_SEQUENCE_LENGTH)
# Define model

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedded_sequences = pretrained_embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)

predictions = Dense(N_CLASSES, activation='softmax')(average)



model = Model(sequence_input, predictions)



# We do not want to fine-tune embeddings

model.layers[1].trainable = False



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=0.001),

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

                    validation_data=(x_val, y_val),

                    epochs=100, batch_size=64)
fig = plt.figure(figsize=(10, 5))

    

ax1 = fig.add_subplot(121)

ax1.plot(history.history['loss'], label='Loss')

ax1.plot(history.history['val_loss'], label='Validation loss')

ax1.set(title='Model loss', xlabel='Epochs', ylabel='Loss')

ax1.legend()



ax2 = fig.add_subplot(122)

ax2.plot(history.history['accuracy'], label='Accuracy')

ax2.plot(history.history['val_accuracy'], label='Validation accuracy')

ax2.set(title='Model accuracy', xlabel='Epochs', ylabel='Accuracy')

ax2.legend()



plt.show()
# Prediction the test set

subm = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

subm['target'] = np.argmax(model.predict(x_test), axis=1)

subm.to_csv('model_glove.csv', index=False)