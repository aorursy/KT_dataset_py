import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from tqdm import tqdm

from collections import Counter

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, utils, callbacks

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
def dec2binseq(x, L=0, as_string=False):
    binx = bin(x)[2:]
    if len(binx)<L:
        binx = (L-len(binx))*"0"+binx
    if not as_string:
        binx = [int(c) for c in binx]
    return binx
    
def binseq2dec(x):
    dec = 0
    L = len(x)
    for i, d in enumerate(x):
        dec += int(d) * 2**(L-i-1)
    return dec

def generate_equations(n_equations=1024, num_min=0, num_max=64):
    X = []
    y = []
    for i in range(n_equations):
        n1, n2 = np.random.randint(num_min, num_max, (2))
        n3 = n1+n2
        n1b, n2b, n3b = dec2binseq(n1, 9), dec2binseq(n2, 9), dec2binseq(n3, 9)
        op = i%2 # 0 for +, 1 for -
        new_X = []
        new_y = []
        if op==0:
            #new_X += [int(c) for c in n1b]
            #new_X.append(op)
            #new_X += [int(c) for c in n2b]
            #new_y += [int(c) for c in n3b]
            new_X += [*n1b, op, *n2b]
            new_y += n3b
        elif op==1:
            #new_X += [int(c) for c in n3b]
            #new_X.append(op)
            #new_X += [int(c) for c in n2b]
            #new_y += [int(c) for c in n1b]
            new_X += [*n3b, op, *n2b]
            new_y += n1b
        X.append(new_X)
        y.append(new_y)
    return np.array(X), np.array(y)
            
#X, y = generate_equations(4)
x = 66
print(x)
xb = dec2binseq(x, 20, True)
print(xb)
xd = binseq2dec(xb)
print(xd)
model = models.Sequential(layers=[
    layers.Dense(64, activation='relu', kernel_regularizer='l2', input_shape=(19,)),
    layers.BatchNormalization(),
    layers.Dropout(.2),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.BatchNormalization(),
    layers.Dropout(.2),
    layers.Dense(9, activation='sigmoid')
])
model.summary()
train_X, train_y = generate_equations(4096)
val_X, val_y = generate_equations(512)
test_X, test_y = generate_equations(512)
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['mae']
)

EPOCHS=128
history = model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 128,
    callbacks=[callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=6)],
    verbose = 0
)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_mae = history.history['mae']
val_mae = history.history['val_mae']

lr = history.history['lr']


plt.plot(np.arange(1,EPOCHS+1), train_loss, 'r--', label='Training loss')
plt.plot(np.arange(1,EPOCHS+1), val_loss, 'g-', label='Validation loss')
plt.title('Training and validation MAE')
plt.legend()
plt.show()

plt.plot(np.arange(1,EPOCHS+1), train_mae, 'r--', label='Training MAE')
plt.plot(np.arange(1,EPOCHS+1), val_mae, 'g-', label='Validation MAE')
plt.title('Training and validation MAE')
plt.legend()
plt.show()

plt.plot(np.arange(1,EPOCHS+1), lr, 'b-', label='Learning Rate')
plt.yscale('log')
plt.title('Learning rate')
plt.legend()
plt.show()

lr[-1]
model.compile(
    optimizer=optimizers.RMSprop(lr=lr[-1]/10),
    loss='binary_crossentropy',
    metrics=['mae']
)

history = model.fit(
    train_X, train_y,
    validation_data = (val_X, val_y),
    epochs = 128,
    callbacks=[callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=6)],
    verbose = 0
)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_mae = history.history['mae']
val_mae = history.history['val_mae']

lr = history.history['lr']

sns.set_style('darkgrid')

plt.plot(np.arange(1,EPOCHS+1), train_loss, 'r--', label='Training loss')
plt.plot(np.arange(1,EPOCHS+1), val_loss, 'g-', label='Validation loss')
plt.title('Training and validation MAE')
plt.legend()
plt.show()

plt.plot(np.arange(1,EPOCHS+1), train_mae, 'r--', label='Training MAE')
plt.plot(np.arange(1,EPOCHS+1), val_mae, 'g-', label='Validation MAE')
plt.title('Training and validation MAE')
plt.legend()
plt.show()

plt.plot(np.arange(1,EPOCHS+1), lr, 'b-', label='Learning Rate')
plt.yscale('log')
plt.title('Learning rate')
plt.legend()
plt.show()

lr[-1]
for test_idx in np.random.randint(0, test_X.shape[0], (6)):
    pred = model.predict(test_X[test_idx].reshape(1,-1)).round().astype(np.int32).squeeze()
    target = test_y[test_idx]
    print("Pred:\t", pred)
    print("Target:\t", target, "\tMatch: ", np.all(pred==target))
    print()
train_score = Counter()
val_score = Counter()
test_score = Counter()

for X, y in tqdm(zip(train_X, train_y)):
    if np.all(model.predict(X.reshape(1,-1)).round().astype(np.int32).squeeze() == y):
        train_score['Correct'] += 1
    else:
        train_score['Incorrect'] += 1
        
for X, y in tqdm(zip(val_X, val_y)):
    if np.all(model.predict(X.reshape(1,-1)).round().astype(np.int32).squeeze() == y):
        val_score['Correct'] += 1
    else:
        val_score['Incorrect'] += 1

for X, y in tqdm(zip(test_X, test_y)):
    if np.all(model.predict(X.reshape(1,-1)).round().astype(np.int32).squeeze() == y):
        test_score['Correct'] += 1
    else:
        test_score['Incorrect'] += 1

        
print("Training:\n\tCorrect: %i\tIncorrect: %i\t\t~=%.2f%%" % (train_score['Correct'], train_score['Incorrect'], 100*train_score['Correct']/train_X.shape[0]) )
print("Training:\n\tCorrect: %i\tIncorrect: %i\t\t~=%.2f%%" % (val_score['Correct'], val_score['Incorrect'], 100*val_score['Correct']/(val_X.shape[0])) )
print("Training:\n\tCorrect: %i\tIncorrect: %i\t\t~=%.2f%%" % (test_score['Correct'], test_score['Incorrect'], 100*test_score['Correct']/(test_X.shape[0])) )
