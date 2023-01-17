# data libs

import pandas as pd

import numpy as np

import seaborn as sns



from matplotlib import pyplot as plt

from matplotlib.gridspec import GridSpec



sns.set_palette('Pastel1')

sns.set_style('whitegrid')



# fix the seed to make this notebook reproducible

np.random.seed = 69
data = pd.read_csv('../input/spam.csv', encoding='latin1')

data.head()
# drop unnamed columns

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# rename columns

data.columns = ['label', 'text']

data.head()
def get_lengths_of_texts(data):

    texts = data['text'].values

    return [len(text) for text in texts]



lengths = get_lengths_of_texts(data)



mean_length = np.mean(lengths)

std_length = np.std(lengths)



spam_lengths = get_lengths_of_texts(data[data['label'] == 'spam'])



mean_spam_length = np.mean(spam_lengths)

std_spam_length = np.std(spam_lengths)



normal_lengths = get_lengths_of_texts(data[data['label'] == 'ham'])



mean_normal_length = np.mean(normal_lengths)

std_normal_length = np.std(normal_lengths)
def annotate_values(ax, values):

    for react, val in zip(ax.patches, values):

        h = react.get_height()

        ax.text(react.get_x() + react.get_width() / 2, h - (h * 0.1),

            "{0:.2f}".format(val), ha='center', style='italic', fontsize=13)



grid = GridSpec(1, 2)

fig = plt.figure(figsize=(15,5))

fig.suptitle('Stats from the length of text')



ax = plt.subplot(grid[0, 0])

values = [mean_spam_length, std_spam_length]

sns.barplot(x=['Mean', 'Std'], y=values, ax=ax)

ax.set_title('Mean & Std for Spam messages')

annotate_values(ax, values)



ax = plt.subplot(grid[0, 1])

values = [mean_normal_length, std_normal_length]

sns.barplot(x=['Mean', 'Std'], y=values, ax=ax)

ax.set_title('Mean & Std for Normal messages')

annotate_values(ax, values)

plt.show()
def isspam(message):

    return len(message) >= 138



print("What am I?")

vld = data[data['label'] == 'spam'].sample(1)

print('>', 'Spam' if isspam(vld['text']) else 'Not Spam')

print("Actually I'm a", *vld['label'].values)
data.head()
# a simple generator function to compute the label

def encode_labels(labels):

    return [1 if 'spam' in label else 0 for label in labels]
from keras.preprocessing.text import Tokenizer





class TextEncoder:

    def __init__(self):

        self.tokenizer = Tokenizer()

    

    def fit(self, texts):

        self.tokenizer.fit_on_texts(texts)

    

    def transform(self, texts):

        return self.tokenizer.texts_to_matrix(texts)

    

    def fit_transform(self, texts):

        self.fit(texts)

        return self.transform(texts)

    

    @property

    def dim(self):

        return len(self.tokenizer.word_index) + 1

    
texts = data['text'].values

encoder = TextEncoder()

X = encoder.fit_transform(texts)

y = encode_labels(data['label'].values)
from keras.models import Sequential

from keras.layers import Dense, Dropout



model = Sequential()

# dimensions calulated by our tokenizer

dims = encoder.dim

# add input layer

model.add(Dense(2, input_dim=dims))

# add output lauer

model.add(Dense(1, activation='sigmoid'))

# compile the model

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# fit it!

history = model.fit(X, y, shuffle=True, validation_split=.2, epochs=100, batch_size=50)
grid = GridSpec(2, 2)



fig = plt.figure(figsize=(13, 10))

fig.suptitle('Model Metrics')



# first plot



ax = plt.subplot(grid[0, :])

ax.set_title('Model Loss')

ax.plot(history.history['loss'], color='c', lw=2)

ax.plot(history.history['val_loss'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Loss')

# annotate

ax.annotate('Start Overfitting', xy=(8, 0.06),

            xytext=(10, 0.1),

            arrowprops=dict(arrowstyle='->'))



ax.annotate('Look at this gape', xy=(85, 0.06),

            xytext=(85, 0.1))



ax.legend(['Train Loss', 'Val Loss'], loc='best')



# Second plot



ax = plt.subplot(grid[1, :])

ax.set_title('Model Accuracy')

ax.plot(history.history['acc'], color='c', lw=2)

ax.plot(history.history['val_acc'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Accuracy')

# annotate

ax.annotate('Start Overfitting', xy=(8, 0.985),

            xytext=(10, 0.97),

            arrowprops=dict(arrowstyle='->'))



ax.annotate('Look at this gape', xy=(85, 0.06),

            xytext=(85, 0.1))



ax.legend(['Train Accuracy', 'Val Accuracy'], loc='best')



plt.show()
model0 = Sequential()

# dimensions calulated by our tokenizer

dims = encoder.dim

# add input layer

model0.add(Dense(2, input_dim=dims))

# add output lauer

model0.add(Dense(1, activation='sigmoid'))

# compile the model

model0.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# fit it!

history = model0.fit(X, y, shuffle=True, validation_split=.2, epochs=10, batch_size=32)
grid = GridSpec(2, 2)



fig = plt.figure(figsize=(13, 10))

fig.suptitle('Model Metrics')



# first plot



ax = plt.subplot(grid[0, :])

ax.set_title('Model Loss')

ax.plot(history.history['loss'], color='c', lw=2)

ax.plot(history.history['val_loss'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Loss')

# annotate

ax.annotate('Start Overfitting', xy=(8, 0.06),

            xytext=(10, 0.1),

            arrowprops=dict(arrowstyle='->'))



ax.annotate('Look at this gape', xy=(85, 0.06),

            xytext=(85, 0.1))



ax.legend(['Train Loss', 'Val Loss'], loc='best')



# Second plot



ax = plt.subplot(grid[1, :])

ax.set_title('Model Accuracy')

ax.plot(history.history['acc'], color='c', lw=2)

ax.plot(history.history['val_acc'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Accuracy')

ax.legend(['Train Accuracy', 'Val Accuracy'], loc='best')



plt.show()
model1 = Sequential()

# dimensions calulated by our tokenizer

dims = encoder.dim

# add input layer

model1.add(Dense(2, input_dim=dims))

# add output lauer

model1.add(Dense(1, activation='sigmoid'))

# compile the model

model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# fit it!

history = model1.fit(X, y, shuffle=True, validation_split=.2, epochs=5, batch_size=32)
grid = GridSpec(2, 2)



fig = plt.figure(figsize=(13, 10))

fig.suptitle('Model Metrics')



# first plot



ax = plt.subplot(grid[0, :])

ax.set_title('Model Loss')

ax.plot(history.history['loss'], color='c', lw=2)

ax.plot(history.history['val_loss'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Loss')

ax.legend(['Train Loss', 'Val Loss'], loc='best')



# Second plot



ax = plt.subplot(grid[1, :])

ax.set_title('Model Accuracy')

ax.plot(history.history['acc'], color='c', lw=2)

ax.plot(history.history['val_acc'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Accuracy')

ax.legend(['Train Accuracy', 'Val Accuracy'], loc='best')



plt.show()
model2 = Sequential()

# dimensions calulated by our tokenizer

dims = encoder.dim

# add input layer

model2.add(Dense(2, input_dim=dims))

model2.add(Dense(2, activation='relu'))

model2.add(Dropout(0.5))

model2.add(Dense(2, activation='relu'))

# add output lauer

model2.add(Dense(1, activation='sigmoid'))

# compile the model

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit it!

history = model2.fit(X, y, shuffle=True, validation_split=.2, epochs=50, batch_size=50)
grid = GridSpec(2, 2)



fig = plt.figure(figsize=(13, 10))

fig.suptitle('Model Metrics')



# first plot



ax = plt.subplot(grid[0, :])

ax.set_title('Model Loss')

ax.plot(history.history['loss'], color='c', lw=2)

ax.plot(history.history['val_loss'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Loss')

ax.legend(['Train Loss', 'Val Loss'], loc='best')



# Second plot



ax = plt.subplot(grid[1, :])

ax.set_title('Model Accuracy')

ax.plot(history.history['acc'], color='c', lw=2)

ax.plot(history.history['val_acc'], color='darkorange', lw=2)

ax.set_xlabel('Epochs')

ax.set_ylabel('Accuracy')

ax.legend(['Train Accuracy', 'Val Accuracy'], loc='best')



plt.show()