import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from keras import layers

from keras import models

from keras.utils import to_categorical



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
PATH = "/kaggle/input/digit-recognizer/"

train_df = pd.read_csv(PATH + 'train.csv')

train_df.shape
X = train_df.iloc[:, 1:].values

y = train_df.iloc[:, 0].values
# view an example image from the dataset

plt.imshow(X[1600].reshape((28, 28)))
# split training data into training and validation splits

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12) 



# form a basic random forest (default hyper-parameters)

rf_model = RandomForestClassifier()



# fit model with training data

rf_model.fit(X_train, y_train)



# predict output labels

preds = rf_model.predict(X_val)



# view accuracy

accuracy_score(preds, y_val)
PATH = "/kaggle/input/digit-recognizer/"

test_df = pd.read_csv(PATH + 'test.csv')

test_df.shape
rf_model = RandomForestClassifier()

rf_model.fit(X, y)
PATH = "/kaggle/input/digit-recognizer/"

test_df = pd.read_csv(PATH + 'test.csv')

test_df.shape
X_test = test_df.values

test_preds = rf_model.predict(X_test)



test_preds
image_id = np.arange(1, len(test_preds) + 1)
image_id.shape
test_preds.shape
submission = pd.DataFrame({'Imageid' : image_id, 'Label' : test_preds})

submission
submission.to_csv('random_forrest_submission.csv', index=False)
X.shape
# split data into training / validation splits

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=12) 



# standardise / normalise each of our pixels

X_train = X_train.astype('float32') / 255.0

X_val = X_val.astype('float32') / 255.0



# one-hot encode our labels

y_train_one_hot = to_categorical(y_train)

y_val_one_hot = to_categorical(y_val)
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))



model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_one_hot, epochs=40, 

                    batch_size=512, validation_data=(X_val, y_val_one_hot))
hist_dict = history.history



trg_loss = hist_dict['loss']

val_loss = hist_dict['val_loss']



trg_acc = hist_dict['accuracy']

val_acc = hist_dict['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

plt.ylabel("Loss")

plt.xlabel("Epochs")

plt.legend(loc='best')

plt.show()
plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Loss")

plt.ylabel("Accuracy")

plt.xlabel("Epochs")

plt.legend(loc='best')

plt.show()
X_full = X.astype('float32') / 255.0

y_one_hot = to_categorical(y)
model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))



model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(X_full, y_one_hot, epochs=40, batch_size=512)
hist_dict = history.history



trg_loss = history.history['loss']

trg_acc = history.history['accuracy']



epochs = range(1, len(trg_acc) + 1)



fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(epochs, trg_loss, label='Training Loss')

ax[1].plot(epochs, trg_acc, label='Training Accuracy')

ax[0].set_ylabel('Loss')

ax[1].set_ylabel('Accuracy')



plt.show()
PATH = "/kaggle/input/digit-recognizer/"

test_df = pd.read_csv(PATH + 'test.csv')

test_df.shape
# make our predictions and convert back into sparse from one-hot encoded

X_test = test_df.values

test_preds = model.predict(X_test)

test_preds = np.argmax(test_preds, axis=1)



test_preds.shape
image_id = np.arange(1, len(test_preds) + 1)
submission = pd.DataFrame({'Imageid' : image_id, 'Label' : test_preds})

submission
submission.to_csv('deep_neural_network.csv', index=False)
# reshape and standardise our data

X = X.reshape((-1, 28, 28, 1))

X_std = X.astype('float32') / 255.0



# obtain test set for predictions later

X_test = test_df.values

X_test = X_test.reshape((-1, 28, 28, 1))

X_test_std = X_test.astype('float32') / 255.0



# one-hot encode our labels

y_one_hot = to_categorical(y)



print(f"Training data: {X.shape}, Training Labels: {y_one_hot.shape} \n"

      f"Test shape: {X_test.shape}")
# split data into training / validation splits

X_train, X_val, y_train, y_val = train_test_split(X_std, y_one_hot, test_size=0.2, random_state=12) 



print(f"X_train: {X_train.shape}, y_train: {y_train.shape} \n"

      f"X_val: {X_val.shape}, y_val: {y_val.shape} \n")
cnn_model = models.Sequential()

cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', 

                            input_shape=(X.shape[1], X.shape[2], X.shape[3])))

cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(layers.Flatten())

cnn_model.add(layers.Dense(64, activation='relu'))

cnn_model.add(layers.Dense(10, activation='softmax'))



cnn_model.summary()
cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(X_train, y_train, epochs=20, 

                        batch_size=128, validation_data=(X_val, y_val))
hist_dict = history.history



trg_loss = hist_dict['loss']

val_loss = hist_dict['val_loss']



trg_acc = hist_dict['accuracy']

val_acc = hist_dict['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



# plot losses and accuracies for training and validation 

fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(1, 2, 1)

plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

ax.set_ylabel("Loss")

ax.set_xlabel("Epochs")

plt.legend(loc='best')



ax = fig.add_subplot(1, 2, 2)

plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Accuracy")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.show()
cnn_model = models.Sequential()

cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', 

                            input_shape=(X.shape[1], X.shape[2], X.shape[3])))

cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))

cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))

cnn_model.add(layers.Flatten())

cnn_model.add(layers.Dense(128, activation='relu'))

cnn_model.add(layers.Dense(10, activation='softmax'))



cnn_model.summary()
cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



history = cnn_model.fit(X_train, y_train, epochs=20, 

                        batch_size=128, validation_data=(X_val, y_val))
hist_dict = history.history



trg_loss = hist_dict['loss']

val_loss = hist_dict['val_loss']



trg_acc = hist_dict['accuracy']

val_acc = hist_dict['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



# plot losses and accuracies for training and validation 

fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(1, 2, 1)

plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

ax.set_ylabel("Loss")

ax.set_xlabel("Epochs")

plt.legend(loc='best')



ax = fig.add_subplot(1, 2, 2)

plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Accuracy")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.show()
print(f"Final validation accuracy: {val_acc[-1]}")
def create_cnn_model():

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', 

                            input_shape=(X.shape[1], X.shape[2], X.shape[3])))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', 

                  loss='categorical_crossentropy', 

                  metrics=['accuracy'])

    return model
X_full = X.astype('float32') / 255.0

y_full = to_categorical(y)



cnn_model = create_cnn_model()

cnn_model.summary()
history = cnn_model.fit(X_full, y_full, epochs=20, batch_size=64)
hist_dict = history.history



trg_loss = history.history['loss']

trg_acc = history.history['accuracy']



epochs = range(1, len(trg_acc) + 1)



fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(epochs, trg_loss, label='Training Loss')

ax[1].plot(epochs, trg_acc, label='Training Accuracy')

ax[0].set_ylabel('Loss')

ax[1].set_ylabel('Accuracy')



plt.show()
PATH = "/kaggle/input/digit-recognizer/"

test_df = pd.read_csv(PATH + 'test.csv')

test_df.shape



# format our test data appropriately for our CNN model

X_test = test_df.values

X_test = X_test.reshape((-1, 28, 28, 1))

X_test_std = X_test.astype('float32') / 255.0





test_preds = cnn_model.predict(X_test_std)

test_preds = np.argmax(test_preds, axis=1)



test_preds.shape
submission = pd.DataFrame({'Imageid' : np.arange(1, len(test_preds) + 1), 

                           'Label' : test_preds})

submission.head()
submission.to_csv('convolutional_neural_network.csv', index=False)