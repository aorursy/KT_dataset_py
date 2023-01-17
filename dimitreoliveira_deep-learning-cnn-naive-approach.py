import os
import ast
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

%matplotlib inline
def drawing_to_np(drawing, shape=(28, 28)):
    # evaluates the drawing array
    drawing = eval(drawing)
    fig, ax = plt.subplots()
    for x,y in drawing:
        ax.plot(x, y, marker='.')
        ax.axis('off')        
    fig.canvas.draw()
    # Close figure so it won't get displayed while transforming the set
    plt.close(fig)
    # Convert images to numpy array
    np_drawing = np.array(fig.canvas.renderer._renderer)
    # Take only one channel
    np_drawing =np_drawing[:, :, 1]    
    # Normalize data
    np_drawing = np_drawing / 255.
    return cv2.resize(np_drawing, shape) # Resize array


def plot_metrics_primary(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(20,7))

    ax1.plot(acc, label='Train Accuracy')
    ax1.plot(val_acc, label='Validation accuracy')
    ax1.legend(loc='best')
    ax1.set_title('Accuracy')

    ax2.plot(loss, label='Train loss')
    ax2.plot(val_loss, label='Validation loss')
    ax2.legend(loc='best')
    ax2.set_title('Loss')

    plt.xlabel('Epochs')
    
    
def plot_confusion_matrix(cnf_matrix, labels): 
    cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
    plt.figure(figsize=(20,7))
    sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
    plt.show()
TRAIN_PATH = '../input/train_simplified/'
TEST_PATH = '../input/test_simplified.csv'
SUBMISSION_NAME = 'submission.csv'

train = pd.DataFrame()
for file in os.listdir(TRAIN_PATH)[:5]:
    train = train.append(pd.read_csv(TRAIN_PATH + file, usecols=[1, 5], nrows=2000))
# Shuffle data
train = shuffle(train, random_state=123)
test = pd.read_csv(TEST_PATH, usecols=[0, 2], nrows=100)
# Model parameters
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
N_CLASSES = train['word'].nunique()
HEIGHT = 28
WIDTH = 28
CHANNEL = 1
print('Train set shape: ', train.shape)
print('Train set features: %s' % train.columns.values)
print('Train number of label categories: %s' % N_CLASSES)
train.head()
#Fixing labels.
train['word'] = train['word'].replace(' ', '_', regex=True)
# Get labels and one-hot encode them.
classes_names = train['word'].unique()
labels = pd.get_dummies(train['word']).values
train.drop(['word'], axis=1, inplace=True)
# Transform drawing into numpy arrays
train['drawing_np'] = train['drawing'].apply(drawing_to_np)
# Reshape arrays
train_drawings = np.asarray([x.reshape(HEIGHT, WIDTH, CHANNEL) for x in train['drawing_np'].values])
train.head()
x_train, x_val, y_train, y_val = train_test_split(train_drawings, labels, test_size=0.1, random_state=1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(HEIGHT, WIDTH, CHANNEL)))
model.add(Conv2D(32, kernel_size=(5,5),padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3),padding='Same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3),padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(N_CLASSES, activation = "softmax"))

optimizer = optimizers.adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer , loss="categorical_crossentropy", metrics=["accuracy"])
print('Dataset size: %s' % train.shape[0])
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: (%s, %s, %s)' % (HEIGHT, WIDTH, CHANNEL))

model.summary()
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val))
plot_metrics_primary(history.history['acc'], history.history['val_acc'], history.history['loss'], history.history['val_loss'])
cnf_matrix = confusion_matrix(np.argmax(y_val, axis=1), model.predict_classes(x_val))
plot_confusion_matrix(cnf_matrix, classes_names)
# Transform drawing into numpy arrays.
test['drawing_np'] = test['drawing'].apply(drawing_to_np)
# Reshape arrays.
test_drawings = np.asarray([x.reshape(HEIGHT, WIDTH, CHANNEL) for x in test['drawing_np'].values])
predictions = model.predict(test_drawings)
top_3_predictions = np.asarray([np.argpartition(pred, -3)[-3:] for pred in predictions])
top_3_predictions = ['%s %s %s' % (classes_names[pred[0]], classes_names[pred[1]], classes_names[pred[2]]) for pred in top_3_predictions]
test['word'] = top_3_predictions
submission = test[['key_id', 'word']]
submission.to_csv(SUBMISSION_NAME, index=False)
submission.head(10)