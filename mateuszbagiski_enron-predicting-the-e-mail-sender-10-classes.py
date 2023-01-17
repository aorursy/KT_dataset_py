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


# For removing stopwords
from nltk.corpus import stopwords
stopwords_eng = set(stopwords.words('english'))
def remove_stopwords(txt):
    for stopword in stopwords_eng:
        while stopword in txt:
            txt = txt.replace(stopword,'')
    return txt

# For convenient train:val:test splitting
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
data.head()
print(data.message[0])
sender_pattern = re.compile('(?<=From:\s).*') # Because we do not pass the re.S flag, the dot does not match the newline char (\n) and the match is cut at the end of the line
receiver_pattern = re.compile('(?<=To:\s).*')

print(sender_pattern.search(data.message[0]).group())
print(sender_pattern.search(data.message[123123]).group())
print(sender_pattern.search(data.message[9999]).group())
xfn_pattern = re.compile('X-FileName:.*')
content_pattern = re.compile('[^\n].*\Z', re.S)

print(data.message[0])
xfn_end = xfn_pattern.search(data.message[0]).end()
match = content_pattern.search(data.message[0], pos=xfn_end)

print("Message:")
print(match.group())
data['Sender'] = data.message.apply(lambda x: sender_pattern.search(x).group())
data['Receiver'] = data.message.apply(lambda x: receiver_pattern.search(x).group())
data['Content'] = data.message.apply(
    lambda x: content_pattern.search(
        x,
        pos = xfn_pattern.search(x).end()
    ).group())
data['Length'] = data.Content.apply(lambda x: len(x))

print(data.isnull().any())
senders_list = data.Sender.value_counts().index[:10].tolist() # 1)
n_per_class = 1280

texts = [] # 2)
labels = [] # 3)
for i, sender in tqdm(enumerate(senders_list)):
    sender_texts = data.query(' `Sender` == @sender ').sample(frac=1)['Content'][:n_per_class].apply(lambda x: remove_stopwords(x)).values.tolist() # 2)
    texts += sender_texts # 2)
    labels += (np.ones(shape=(n_per_class,))*i).tolist() # 3)
    

max_features = 512  # Maximum number of words, we are going to embed
embed_dim = 128      # Number of embedding dimensions in the word embedding space constructed by the embedding layer
maxlen = 200         # The length of the sequence (message) - The longer messages will be cropped to 256, while the shorter ones will be padded with zeros to be exactly 256 tokens long


tokenizer = Tokenizer(num_words=max_features) # 4)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

seqs = tokenizer.texts_to_sequences(texts) # 5)
seqs = pad_sequences(seqs, maxlen=maxlen) # 6)

labels = np.array(labels)

train_X, val_X, test_X, train_y, val_y, test_y = tvt_split(seqs, labels, stratify=True) # 7)
# Sanity check for proper shapes and the equal distribution of classes
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
    layers.Dense(10, activation='softmax')
])
model.summary()
def lr_scheduler(epoch, lr):
    if epoch==8 or epoch==12:
        return lr/8
    else:
        return lr
        

callbacks_list = [
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, save_freq='epoch'),
    callbacks.LearningRateScheduler(lr_scheduler)
]

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

EPOCHS = 32

history = model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = EPOCHS, batch_size=64,
    shuffle = True,
    verbose = 1,
    callbacks = callbacks_list
)

pd.DataFrame(history.history).to_csv('history.csv')
model.save('last_model.h5')
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_acc = history.history['acc']
val_acc = history.history['val_acc']

lr = history.history['lr']

x = np.arange(1, EPOCHS+1)

plt.plot(x, train_loss, 'r--', label='Training loss')
plt.plot(x, val_loss, 'g-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.plot(x, train_acc, 'r--', label='Training accuracy')
plt.plot(x, val_acc, 'g-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(x, lr, 'b-', label='Learning Rate')
plt.yscale('log')
plt.title('Learning rate')
plt.legend()
plt.show()
model.load_weights('last_model.h5')
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
print("\t\tLAST MODEL:")
print('Training set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(train_X, train_y, verbose=0)))
print('Validation set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(val_X, val_y, verbose=0)))
print('Testing set evaluation:\t\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(test_X, test_y, verbose=0)))

best_model = models.load_model('best_model.h5')
best_model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
print("\n\t\tBEST MODEL:")
print('Training set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(best_model.evaluate(train_X, train_y, verbose=0)))
print('Validation set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(best_model.evaluate(val_X, val_y, verbose=0)))
print('Testing set evaluation:\t\tLoss: %.4f\tAccuracy: %.4f' % tuple(best_model.evaluate(test_X, test_y, verbose=0)))
cm = np.zeros(shape=(10,10))

for X, y in tqdm(zip(test_X, test_y)):
    pred = model.predict(X.reshape(1,-1)).argmax().astype(np.int32)
    y = int(y)
    cm[pred,y] += 1
cm = pd.DataFrame(cm, columns=senders_list, index=senders_list)
cm
cl = []
for row in cm.index:
    for col in cm.columns:
        cl.append([cm.loc[col, row], row, col])
        
cl = pd.DataFrame(cl, columns=['Confusions', 'Predicted', 'Actual']).sort_values(by='Confusions', ascending=False)
cl = cl.query('`Predicted`!=`Actual`').reset_index(drop=True)
cl.head()
cl['Reverse'] = cl.apply(lambda x: cl.query('`Predicted`==@x.Actual & `Actual`==@x.Predicted').values[0][0] , axis=1)
cl.iloc[:10]
# Sanity check
cl.query('Confusions==2')
receivers_list = data.Receiver.value_counts().index[:10].tolist() # 1)
n_per_class = 1280

texts = [] # 2)
labels = [] # 3)
for i, receiver in tqdm(enumerate(receivers_list)):
    receiver_texts = data.query(' `Receiver` == @receiver ').sample(frac=1)['Content'][:n_per_class].apply(lambda x: remove_stopwords(x)).values.tolist() # 2)
    texts += receiver_texts # 2)
    labels += (np.ones(shape=(n_per_class,))*i).tolist() # 3)
    
max_features = 512  # Maximum number of words, we are going to embed
embed_dim = 128      # Number of embedding dimensions in the word embedding space constructed by the embedding layer
maxlen = 200         # The length of the sequence (message) - The longer messages will be cropped to 256, while the shorter ones will be padded with zeros to be exactly 256 tokens long


tokenizer = Tokenizer(num_words=max_features) # 4)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

seqs = tokenizer.texts_to_sequences(texts) # 5)
seqs = pad_sequences(seqs, maxlen=maxlen) # 6)

labels = np.array(labels)

train_X, val_X, test_X, train_y, val_y, test_y = tvt_split(seqs, labels, stratify=True) # 7)
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
    layers.Dense(10, activation='softmax')
])

def lr_scheduler(epoch, lr):
    if epoch==8 or epoch==12:
        return lr/8
    else:
        return lr
        

callbacks_list = [
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, save_freq='epoch'),
    callbacks.LearningRateScheduler(lr_scheduler)
]

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

EPOCHS = 32

history = model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = EPOCHS, batch_size=64,
    shuffle = True,
    verbose = 1,
    callbacks = callbacks_list
)

pd.DataFrame(history.history).to_csv('history.csv')
model.save('last_model.h5')
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_acc = history.history['acc']
val_acc = history.history['val_acc']

lr = history.history['lr']

x = np.arange(1, EPOCHS+1)

plt.plot(x, train_loss, 'r--', label='Training loss')
plt.plot(x, val_loss, 'g-', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

plt.plot(x, train_acc, 'r--', label='Training accuracy')
plt.plot(x, val_acc, 'g-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(x, lr, 'b-', label='Learning Rate')
plt.yscale('log')
plt.title('Learning rate')
plt.legend()
plt.show()
model.load_weights('last_model.h5')
model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
print("\t\tLAST MODEL:")
print('Training set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(train_X, train_y, verbose=0)))
print('Validation set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(val_X, val_y, verbose=0)))
print('Testing set evaluation:\t\tLoss: %.4f\tAccuracy: %.4f' % tuple(model.evaluate(test_X, test_y, verbose=0)))

best_model = models.load_model('best_model.h5')
best_model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
print("\n\t\tBEST MODEL:")
print('Training set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(best_model.evaluate(train_X, train_y, verbose=0)))
print('Validation set evaluation:\tLoss: %.4f\tAccuracy: %.4f' % tuple(best_model.evaluate(val_X, val_y, verbose=0)))
print('Testing set evaluation:\t\tLoss: %.4f\tAccuracy: %.4f' % tuple(best_model.evaluate(test_X, test_y, verbose=0)))
