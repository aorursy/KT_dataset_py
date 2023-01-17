%matplotlib inline

import glob
import os
import cv2
import numpy as np

import tqdm

import matplotlib.pyplot as plt
train_files = glob.glob('../input/dice-d4-d6-d8-d10-d12-d20/dice/train/*/*')
len(train_files)
test_files = glob.glob('../input/dice-d4-d6-d8-d10-d12-d20/dice/valid/*/*')
len(test_files)
DIMS = (196, 196, 3)

def read_image(path):
    label = os.path.basename(os.path.dirname(path))
    
    image = cv2.imread(path)
    image = cv2.resize(image, (DIMS[:2]))
    
    return image / 255.0

def randomize_dataset(X, labels):
    # randomize dataset
    ids = list(range(len(X)))
    np.random.shuffle(ids)
    
    for (i, j) in enumerate(ids):
        X[i], X[j] = X[j], X[i]
        labels[i], labels[j] = labels[j], labels[i]
        
    return (X, labels)
        
def read_dataset(train_files):
    dataset = []
    
    X = np.zeros((len(train_files), *DIMS), dtype=np.float32)
        
    for (i, file) in tqdm.tqdm_notebook(enumerate(train_files)):
        image = read_image(file)
        X[i]  = image
    
    labels = [os.path.basename(os.path.dirname(path)) for path in train_files]
    X, labels = randomize_dataset(X, labels)
    
    classes  = set([label for label in labels])
    id2class = dict([(i, c) for (i, c) in enumerate(classes)])
    class2id = dict([(c, i) for (i, c) in enumerate(classes)])
    
    Y = np.zeros((len(train_files), len(classes)), dtype=np.bool)
    
    for (i, clazz) in enumerate(labels):
        Y[i, class2id[clazz]] = 1
        
    return (X, Y, classes, id2class, class2id)
X, Y, classes, id2class, class2id = read_dataset(train_files)
X_test, Y_test, _, _, _ = read_dataset(test_files)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Dropout, Dense
model = Sequential()

model.add(Conv2D(8, (5, 5), activation='relu', padding='same', input_shape=DIMS))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D())

model.add(GlobalAveragePooling2D())

model.add(Dense(len(classes), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X, Y, batch_size=16, epochs=10, verbose=1, validation_split=0.1)
model.evaluate(X_test, Y_test)