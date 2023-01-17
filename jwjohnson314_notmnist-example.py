import numpy as np
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPool2D, Dropout, Dense, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

% matplotlib inline
im_root = '../input/notMNIST_large/notMNIST_large'
dirs = os.listdir(im_root)
fig, ax = plt.subplots(2, 5, figsize=(16, 6))

for i in range(2):
    for j in range(5):
        dr = random.choice(dirs)
        im = random.choice(os.listdir(os.path.join(im_root, dr)))
        ax[i, j].imshow(plt.imread(os.path.join(im_root, dr, im)), cmap='gray')
        ax[i, j].set_title(dr)
        ax[i, j].axis('off')
xtr = np.zeros((60000, 28, 28))
ytr = np.zeros((60000))

xte = np.zeros((10000, 28, 28))
yte = np.zeros((10000))

im_root = '../input/notMNIST_large/notMNIST_large'
dirs = os.listdir(im_root)

label_dict = defaultdict()
for idx, dr in enumerate(dirs):
    label_dict[idx] = dr
    ims = os.listdir(os.path.join(im_root, dr))
    random.shuffle(ims)
    for i in range(6000):
        im = None
        while im is None:
            try:
                im = plt.imread(os.path.join(im_root, dr, ims[i]))
            except:
                del ims[i]
                continue
        xtr[idx * 6000 + i, :, :] = im  # plt.imread(os.path.join(im_root, dr, ims[i]))
        ytr[idx * 6000 + i] = idx
    for i in range(1000):
        im = None
        while im is None:
            try:
                im = plt.imread(os.path.join(im_root, dr, ims[6000 + i]))
            except:
                del ims[i]
                continue
        xte[idx * 1000 + i, :, :] = im  # plt.imread(os.path.join(im_root, dr, ims[12000+i]))
        yte[idx * 1000 + i] = idx
label_dict.items()
xtr = np.expand_dims(xtr, -1).astype('float32') / 255
xte = np.expand_dims(xte, -1).astype('float32') / 255

ytr = np.array([[1 if ytr[i] == j else 0 for j in range(10)] for i in range(len(ytr))])
yte = np.array([[1 if yte[i] == j else 0 for j in range(10)] for i in range(len(yte))])
inputs = Input(shape=(28, 28, 1))
x = Conv2D(16, 5, activation='relu', padding='same')(inputs)
x = MaxPool2D()(x)
x = SeparableConv2D(32, 5, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dropout(0.25)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.25)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

early_stop = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['acc'])
val_idx = np.random.choice(list(range(len(xtr))), size=int(0.2 * len(xtr)), replace=False)
train_idx = [i for i in range(len(xtr)) if i not in val_idx]

x_train = xtr[train_idx, :, :, :]
y_train = ytr[train_idx, :]

x_val = xtr[val_idx, :, :, :]
y_val = ytr[val_idx, :]
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                    batch_size=256, epochs=40, 
                    callbacks=[early_stop, reduce_lr])
plt.style.use('bmh')
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
ax[0].plot(history.history['loss'], label='train loss')
ax[0].plot(history.history['val_loss'], label='val loss')
ax[0].legend()
ax[1].plot(history.history['acc'], label='train acc')
ax[1].plot(history.history['val_acc'], label='val acc')
ax[1].legend()
preds = model.predict(xte)
sum(yte.argmax(axis=1) == preds.argmax(axis=1)) / len(xte)
print(classification_report(yte.argmax(axis=1), preds.argmax(axis=1)))
print(confusion_matrix(yte.argmax(axis=1), preds.argmax(axis=1)))
