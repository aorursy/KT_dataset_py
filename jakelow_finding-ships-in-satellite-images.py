import json
import numpy as np

import keras
with open('../input/shipsnet.json', 'r') as f:
    d = json.load(f)
    data, labels, locations, scene_ids = d['data'], d['labels'], d['locations'], d['scene_ids']
x = np.array(data).astype('uint8').reshape((-1, 3, 80, 80)) # 3 channels, 80x80 pixels
y = np.array(labels).astype('uint8')
# flip each x[i] from [layer][row][col] to [row][col][layer]
x = x.transpose([0,2,3,1])
# normalize pixel values to [0, 1]
x = x / 255
# turn y boolean values into categorical values (i.e. [p_notship, p_ship] vectors)
y = keras.utils.np_utils.to_categorical(y)
# expand training data by rotating and flipping
x_prime = np.concatenate((
    x,
    np.rot90(x, 1, (1,2)),
    np.rot90(x, 2, (1,2)),
    np.rot90(x, 3, (1,2)),
    np.flip(x, 1),
    np.flip(x, 1)
    ))

y_prime = np.concatenate((y, y, y, y, y, y))
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

model = keras.models.Sequential()

model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(80, 80, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (9,9), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (9,9), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history = model.fit(x_prime, y_prime, epochs=12, validation_split=0.2, shuffle=True)
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(15,5))

left = fig.add_subplot(1, 2, 1)
right = fig.add_subplot(1, 2, 2)

left.plot(history.history['acc'])
left.plot(history.history['val_acc'])
left.set_title('model accuracy')
left.set_ylabel('accuracy')
left.set_xlabel('epoch')
left.legend(['train', 'test'], loc='lower right')

right.plot(history.history['loss'])
right.plot(history.history['val_loss'])
right.set_title('model loss')
right.set_ylabel('loss')
right.set_xlabel('epoch')
right.legend(['train', 'test'], loc='upper right')

plt.show()
import PIL

scene = PIL.Image.open('../input/scenes/scenes/sfbay_1.png')
tensor = np.array(scene).astype('uint8') / 255
# currently unused
import math

def euclidian_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def manhattan_distance(a, b):
    return abs((a[0] - b[0]) + (a[1] - b[1]))    
width, height = scene.size
STEP_SIZE = 10

ships = {}

for row in range(0, height, STEP_SIZE):
    for col in range(0, width, STEP_SIZE):
        area = tensor[row:row+80, col:col+80, 0:3]
        
        if area.shape != (80, 80, 3):
            continue
            
        prediction = model.predict(np.array([area]))
        score = prediction[0][1]
        
        if score > 0.5:
            print(f"found ship at [{row},{col}] with score {score}")
            ships[row, col] = score
from matplotlib import pyplot as plt
from matplotlib import patches

fig = plt.figure(figsize=(16,32))
ax = fig.add_subplot(3, 1, 1)

ax.imshow(tensor)

for ship in ships:
    row, col = ship
    ax.add_patch(patches.Rectangle((col, row), 80, 80, edgecolor='r', facecolor='none'))

plt.show()