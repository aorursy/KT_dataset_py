# for numerical analysis

import numpy as np 

# to store and process in a dataframe

import pandas as pd 



# for ploting graphs

import matplotlib.pyplot as plt

# advancec ploting

import seaborn as sns



# image processing

import matplotlib.image as mpimg



# train test split

from sklearn.model_selection import train_test_split

# model performance metrics

from sklearn.metrics import confusion_matrix, classification_report



# utility functions

from tensorflow.keras.utils import to_categorical

# sequential model

from tensorflow.keras.models import Sequential

# layers

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout





# from keras.optimizers import RMSprop

# from keras.preprocessing.image import ImageDataGenerator

# from keras.callbacks import ReduceLROnPlateau
# list of files

! ls ../input/digit-recognizer
# import train and test dataset

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
# training dataset

train.head()
# test dataset

test.head()
# looking for missing values

print(train.isna().sum().sum())

print(test.isna().sum().sum())
plt.figure(figsize=(8, 5))

sns.countplot(train['label'], palette='Dark2')

plt.title('Train labels count')

plt.show()
train['label'].value_counts().sort_index()
# first few train images with labels

fig, ax = plt.subplots(figsize=(18, 8))

for ind, row in train.iloc[:8, :].iterrows():

    plt.subplot(2, 4, ind+1)

    plt.title(row[0])

    img = row.to_numpy()[1:].reshape(28, 28)

    fig.suptitle('Train images', fontsize=24)

    plt.axis('off')

    plt.imshow(img, cmap='magma')
# first few test images

fig, ax = plt.subplots(figsize=(18, 8))

for ind, row in test.iloc[:8, :].iterrows():

    plt.subplot(2, 4, ind+1)

    img = row.to_numpy()[:].reshape(28, 28)

    fig.suptitle('Test images', fontsize=24)

    plt.axis('off')

    plt.imshow(img, cmap='magma')
# split into image and labels and convert to numpy array

X = train.iloc[:, 1:].to_numpy()

y = train['label'].to_numpy()



# test dataset

test = test.loc[:, :].to_numpy()



for i in [X, y, test]:

    print(i.shape)
# normalize the data

# ==================



X = X / 255.0

test = test / 255.0
# reshape dataset

# ===============



# shape of training and test dataset

print(X.shape)

print(test.shape)



# reshape the dataframe to 3x3 matrix with 1 channel grey scale values

X = X.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)



# shape of training and test dataset

print(X.shape)

print(test.shape)
# one hot encode target

# =====================



# shape and values of target

print(y.shape)

print(y[0])



# convert Y_train to categorical by one-hot-encoding

y_enc = to_categorical(y, num_classes = 10)



# shape and values of target

print(y_enc.shape)

print(y_enc[0])
# train test split

# ================



# random seed

random_seed = 2



# train validation split

X_train, X_val, y_train_enc, y_val_enc = train_test_split(X, y_enc, test_size=0.3)



# shape

for i in [X_train, y_train_enc, X_val, y_val_enc]:

    print(i.shape)
g = plt.imshow(X_train[0][:,:,0])

print(y_train_enc[0])
g = plt.imshow(X_train[9][:,:,0])

print(y_train_enc[9])
INPUT_SHAPE = (28,28,1)

OUTPUT_SHAPE = 10

BATCH_SIZE = 128

EPOCHS = 10

VERBOSE = 2
model = Sequential()



model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=INPUT_SHAPE))

model.add(MaxPool2D((2,2)))



model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPool2D((2,2)))



model.add(Flatten())



model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train_enc,

                    epochs=EPOCHS,

                    batch_size=BATCH_SIZE,

                    verbose=VERBOSE,

                    validation_split=0.3)
plt.figure(figsize=(14, 5))



plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')



plt.savefig('./foo.png')

plt.show()
# model loss and accuracy on validation set

model.evaluate(X_val, y_val_enc, verbose=False)
# predicted values

y_pred_enc = model.predict(X_val)



# actual

y_act = [np.argmax(i) for i in y_val_enc]



# decoding predicted values

y_pred = [np.argmax(i) for i in y_pred_enc]



print(y_pred_enc[0])

print(y_pred[0])
print(classification_report(y_act, y_pred))
fig, ax = plt.subplots(figsize=(7, 7))

sns.heatmap(confusion_matrix(y_act, y_pred), annot=True, 

            cbar=False, fmt='1d', cmap='Blues', ax=ax)

ax.set_title('Confusion Matrix', loc='left', fontsize=16)

ax.set_xlabel('Predicted')

ax.set_ylabel('Actual')

plt.show()
# predicted values

y_pred_enc = model.predict(test)



# decoding predicted values

y_pred = [np.argmax(i) for i in y_pred_enc]



print(y_pred_enc[0])

print(y_pred[0])
# predicted targets of each images

# (labels above the images are predicted labels)

fig, ax = plt.subplots(figsize=(18, 12))

for ind, row in enumerate(test[:15]):

    plt.subplot(3, 5, ind+1)

    plt.title(y_pred[ind])

    img = row.reshape(28, 28)

    fig.suptitle('Predicted values', fontsize=24)

    plt.axis('off')

    plt.imshow(img, cmap='cividis')
# X_train, X_val, y_train_enc, y_val_enc