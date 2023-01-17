import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import re
from collections import Counter
from tqdm import tqdm
from time import time

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, utils, applications, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
stopwords_eng = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split as tts
def tvt_split(X, y, split_sizes=[8,1,1], stratify=True):
    split_sizes = np.array(split_sizes)
    if stratify:
        train_X, test_X, train_y, test_y = tts(X, y, test_size=split_sizes[2]/split_sizes.sum(), stratify=y)
        train_X, val_X, train_y, val_y = tts(train_X, train_y, test_size=split_sizes[1]/(split_sizes[0]+split_sizes[1]), stratify=train_y)
    else:
        train_X, test_X, train_y, test_y = tts(X, y, test_size=split_sizes[2]/split_sizes.sum())
        train_X, val_X, train_y, val_y = tts(train_X, train_y, test_size=split_sizes[1]/(split_sizes[0]+split_sizes[1]))
    return train_X, val_X, test_X, train_y, val_y, test_y


input_dir = '../input/enron-email-dataset'

data = pd.read_csv(input_dir+'/emails.csv')
data.info()
print(data.file[0])
print(data.message[0])
metadata_pattern = re.compile(r'^.+(?=:)', re.M) # this way we will get greedy behavior
metadata_names = metadata_pattern.findall(data.message[0])
metadata_names
metadata_pattern = re.compile(r'^.+?(?=:)', re.M) # we place ? after the quantifier (+) to prevent greedy behavior
metadata_names = metadata_pattern.findall(data.message[0])
metadata_names
for metadatum_name in metadata_names:
    pattern = re.compile(r'(?<=%s:\s).+$'%metadatum_name, re.M)
    print(metadatum_name, pattern.search(data.message[0]))
    
for metadatum_name in tqdm(metadata_names):
    pattern = re.compile(r'(?<=%s:\s).+$'%metadatum_name, re.M)
    data[metadatum_name] = data.message.apply(lambda x: pattern.search(x).group() if pattern.search(x) != None else None)
weekday_pattern = re.compile(r'\A\w+(?=,?)', re.M) # from the beginning of the input to the first comma, composed of one or more alphanumeric characters
monthday_pattern = re.compile(r'(?<=,\s)\d+\b', re.M) # from the first comma+whitespace, composed of one or more digits, ending with a word boundary
month_pattern = re.compile(r'(?<=\d\s)\w+(?=\s\d)', re.M)
year_pattern = re.compile(r'(?<=[A-Z][a-z]{2}\s)\d+(?=\s\d\d:)', re.M)
hour_pattern = re.compile(r'(?<=\d{4}\s)\d\d(?=:\d\d)', re.M)
minute_pattern = re.compile(r'(?<=\d\d:)\d\d(?=:\d\d)', re.M)
second_pattern = re.compile(r'(?<=\d\d:\d\d:)\d\d(?=\s)', re.M)
data['Weekday'] = data['Date'].apply(lambda x: weekday_pattern.search(x).group())
data['Monthday'] = data['Date'].apply(lambda x: monthday_pattern.search(x).group())
data['Month'] = data['Date'].apply(lambda x: month_pattern.search(x).group())
data['Year'] = data['Date'].apply(lambda x: year_pattern.search(x).group())
data['Hour'] = data['Date'].apply(lambda x: hour_pattern.search(x).group())
data['Minute'] = data['Date'].apply(lambda x: minute_pattern.search(x).group())
data['Second'] = data['Date'].apply(lambda x: second_pattern.search(x).group())
print(set(data['Weekday']))
print(set(data['Monthday']))
print(set(data['Month']))
print(set(data['Year']))
print(set(data['Hour']))
print(set(data['Minute']))
print(set(data['Second']))
data['Date'][0]
time_multi_pattern = re.compile(
    pattern=r'(?P<Weekday>\A[A-Z][a-z]{2})\W+?(?P<Monthday>\d+)\W+?(?P<Month>[A-Z][a-z]{2})\W+?(?P<Year>\d{4})\W+?(?P<Hour>\d\d)\W+?(?P<Minute>\d\d)\W+?(?P<Second>\d\d)',
    flags=re.M
)

time_multi_pattern.groupindex
for new_col_name in time_multi_pattern.groupindex:
    data[new_col_name] = data['Date'].apply(lambda x: time_multi_pattern.search(x).groupdict()[new_col_name])
print(set(data['Weekday']))
print(set(data['Monthday']))
print(set(data['Month']))
print(set(data['Year']))
print(set(data['Hour']))
print(set(data['Minute']))
print(set(data['Second']))
# Define the patterns:
xfn_pattern = re.compile(r'%s.*$'%metadata_names[-1], re.M) # X-FileName: is the last one in the metadata_names list (thus [-1] index) 
content_pattern = re.compile('[^\n].*', flags=re.S) #anything that's not a newline (caret negates \n) followed by any (including none) number of any characters

# Test the patterns:
xfn_end = xfn_pattern.search(data.message[10]).end() # Find the X-FileName information and take the index of its last character 
match = content_pattern.search(data.message[10], pos=xfn_end) # Start searching from this index
print(match.group())
data['Message Content'] = data.message.apply(
    lambda x: content_pattern.search(
        x,
        pos = xfn_pattern.search(x).end()
    ).group()
)
data['Message Length'] = data['Message Content'].apply(lambda x: len(x))

sns.distplot(data['Message Length'])
for order_of_magnitude in reversed(range(2,6)):
    max_ = 10**order_of_magnitude
    print("Messages not longer than %i characters:"%max_)
    plt.hist(data.query('`Message Length`<@max_')['Message Length'], bins=100)
    #histplot(data.query('`Message Length`<@max_'), x='Message Length')
    plt.show()
# Extract the labels of the two most productive e-mail senders

label_0, label_1 = list(data['From'].value_counts().index)[:2]
label_0, label_1
# A function to remove stopwords (not very informative words, like "the", "a", "and", and so on)
def remove_stopwords(txt):
    for stopword in stopwords_eng:
        while stopword in txt:
            txt = txt.replace(stopword,'')
    return txt

# Number of messages per one class - I chose 1280, because after splitting it 8:1:1 into train:val:test sets it gives us 1024:128:128, which are all powers of 2
n_per_class = 1280

# Take that many messages sent by this person after shuffling them (.sample() method)
messages_0 = data.query('From==@label_0')['Message Content'].sample(frac=1)[:n_per_class].values
messages_1 = data.query('From==@label_1')['Message Content'].sample(frac=1)[:n_per_class].values

# Remove stopwords by applying the function defined above
messages_0 = [remove_stopwords(s) for s in messages_0]
messages_1 = [remove_stopwords(s) for s in messages_1]

# Maximum number of words, we are going to embed
max_features = 2048

# Number of embedding dimensions in the word embedding space constructed by the embedding layer
embed_dim = 256

# The length of the sequence (message) - The longer messages will be cropped to 256, while the shorter ones will be padded with zeros to be exactly 256 tokens long
maxlen = 256

# Take messages from both classes and shuffle them before feeding to tokenizer
messages_all = messages_0+messages_1
np.random.shuffle(messages_all)

# The tokenizer ascribes a number to each token (word) in the sequence
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(messages_all)
word_index = tokenizer.word_index # This dictionary translates each word to its index (corresponding number)

# Transform messages into sequences of numbers corresponding to its particular words
seqs_0 = tokenizer.texts_to_sequences(messages_0)
seqs_1 = tokenizer.texts_to_sequences(messages_1)

# Pad sequences, i.e. make them exactly 256 tokens long (as described above)
seqs_0 = pad_sequences(seqs_0, maxlen=maxlen)
seqs_1 = pad_sequences(seqs_1, maxlen=maxlen)

# Concatenate the sequences
seqs_all = np.concatenate([seqs_0, seqs_1], axis=0)

# Create and concatenate the labels
labels_0 = np.zeros(shape=(seqs_0.shape[0]))
labels_1 = np.ones(shape=(seqs_1.shape[0]))
labels_all = np.concatenate([labels_0, labels_1], axis=0)

# Split the dataset into training, validaiton and test sets in default proportions 8:1:1
train_X, val_X, test_X, train_y, val_y, test_y = tvt_split(seqs_all, labels_all, stratify=True)
for X, y in [train_X, train_y], [val_X, val_y], [test_X, test_y]:
    print(X.shape, y.shape)
    print(np.bincount(y.astype(np.int32)))
model = models.Sequential(layers=[
    layers.Embedding(input_dim=max_features, output_dim=embed_dim, input_length=maxlen),
    layers.Bidirectional(layers.GRU(32, activation='relu', return_sequences=True, dropout=.1, recurrent_dropout=.1)),
    layers.Bidirectional(layers.GRU(32, activation='relu', return_sequences=False, dropout=.1, recurrent_dropout=.1)),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(.2),
    layers.Dense(32, activation='relu', kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(.1),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
callbacks_list = [
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, save_freq='epoch'), # Save the best model, i.e. the one with the lowest validation loss (measured after each epoch)
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=5), # Reduce the learning rate by a factor of 5 if validation loss has not been decreasing for 5 epochs
    callbacks.EarlyStopping(patience=10) # Stop training after 10 epochs of no validation loss reduction
]

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

EPOCHS = 24

history = model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = EPOCHS, batch_size=64,
    shuffle = True,
    verbose = 1,
    callbacks = callbacks_list
)

model.save('last_model.h5') # Save the final model, so that we can compare it with the best model (the one with the lowest validation loss)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_acc = history.history['acc']
val_acc = history.history['val_acc']

lr = history.history['lr']


plt.plot(np.arange(1,EPOCHS+1), train_loss, 'r--', label='Training loss')
plt.plot(np.arange(1,EPOCHS+1), val_loss, 'g-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.plot(np.arange(1,EPOCHS+1), train_acc, 'r--', label='Training accuracy')
plt.plot(np.arange(1,EPOCHS+1), val_acc, 'g-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(np.arange(1,EPOCHS+1), lr, 'b-', label='Learning Rate')
plt.yscale('log')
plt.title('Learning rate')
plt.legend()
plt.show()
model = models.load_model('last_model.h5')
print("\t\tLAST MODEL:")
print('Training set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(train_X, train_y, verbose=0)))
print('Validation set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(val_X, val_y, verbose=0)))
print('Testing set evaluation:\t\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(test_X, test_y, verbose=0)))

model = models.load_model('best_model.h5')
print("\n\t\tBEST MODEL:")
print('Training set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(train_X, train_y, verbose=0)))
print('Validation set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(val_X, val_y, verbose=0)))
print('Testing set evaluation:\t\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(test_X, test_y, verbose=0)))

