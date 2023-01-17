import pandas as pd 

import numpy as np 



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping



import matplotlib.pyplot as plt

%matplotlib notebook
X_train = pd.read_csv("../input/train.csv")

y_train = X_train.label

X_train = X_train.drop("label", axis=1)
X_train
X_train /= X_train.max().max()

X_train
X_train.shape
28*28
X_train.values.reshape(-1,28,28,1).shape
X_train = X_train.values.reshape(-1,28,28,1)
fig, ax = plt.subplots(8, 8, figsize=(6, 6))

for i, axi in enumerate(ax.flat):

    axi.imshow(X_train[i, :, :, 0], cmap='binary')

    axi.set(xticks=[], yticks=[])
test_idx = np.random.choice(range(X_train.shape[0]), int(X_train.shape[0] * 0.1), replace=False)

train_idx = [i for i in range(X_train.shape[0]) if i not in test_idx]

X_test = X_train[test_idx]

y_test = pd.get_dummies(y_train[test_idx])

X_train = X_train[train_idx]

y_train = pd.get_dummies(y_train[train_idx])

y_train
model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (4, 4), activation='relu'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, batch_size=320, epochs=1000, verbose=1,

          validation_split=0.1,

          callbacks=[EarlyStopping(monitor='val_acc', patience=3,

                                   verbose=1, mode='auto', restore_best_weights=True)])
model.predict(X_test)
preds = model.predict(X_test).argmax(axis=1)

pd.crosstab(y_test.idxmax(axis=1), preds)
(y_test.idxmax(axis=1) == preds).sum() / len(y_test)
from keras.callbacks import EarlyStopping

test_probs = []

for i in range(10):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(128, (4, 4), activation='relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=320, epochs=1000, verbose=1,

              validation_split=0.1,

              callbacks=[EarlyStopping(monitor='val_acc', patience=3,

                                       verbose=1, mode='auto', restore_best_weights=True)])

    

    test_probs.append(model.predict(X_test))
preds = np.sum(test_probs, axis=0).argmax(axis=1)

pd.crosstab(y_test.idxmax(axis=1), preds)
(y_test.idxmax(axis=1) == preds).sum() / len(y_test)