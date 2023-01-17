import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = keras.models.Sequential()

model.add(Conv2D(filters=20, kernel_size=(5, 5), strides=(3, 3), activation='relu', input_shape=(512, 512, 3)))
model.add(Conv2D(filters=20, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=50, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Conv2D(filters=100, kernel_size=(5, 5), activation='relu'))
model.add(Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=150, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.2))

model.add(Flatten())
model.add(Dense(units=40, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
from keras import optimizers

optim = optimizers.Adam(
    lr=0.001, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=None, 
    decay=1e-6, 
    amsgrad=False
)

model.compile(**{
    'loss': 'binary_crossentropy',
    'optimizer': optim,
    'metrics': [
        'accuracy',
    ],
})
import os
import numpy as np

#PRODUCTION
# This would produce a list of 30703 files in my dataset, which have been extracted to a directory
#image_paths = [n for n in os.listdir('/home/ec2-user/data/') if n.endswith('.npy')]
#KAGGLE
import random
image_paths = [random.choice(('anime', 'notanime')) for _ in range(200)]
#END KAGGLE

def get_data(key):
    #PRODUCTION
    # Each file is an image resized to a (3, 512, 512) float32 array and np.save'd
    # This takes up 6MB/image * 30703 images = 180 GB of disk space
    #return np.load('/home/ec2-user/data/' + key)
    #KAGGLE
    # Instead of loading actual images, we'll just pretend with ones and zeros
    if key == 'anime':
        return np.ones((512, 512, 3), dtype='float32')
    else:
        return np.zeros((512, 512, 3), dtype='float32')
import random

#PRODUCTION
# Since we're only doing a demo, no need for the full larger set
#withhold_test = 5000
#KAGGLE
withhold_test = 50
#ENDKAGGLE

random.shuffle(image_paths)
training_set = image_paths[withhold_test:]
test_set = image_paths[:withhold_test]

def pull_values(dataset):
    while True:
        random.shuffle(dataset)
        for key in dataset:
            yield (get_data(key), key)
            
def get_generator(dataset, batch_size):
    batch_inputs = []
    batch_targets = []
    batch_count = 0
    for data, key in pull_values(dataset):
        batch_inputs.append(data)
        if key[:5] == 'anime':
            batch_targets.append((0, 1))
        else:
            batch_targets.append((1, 0))
        batch_count += 1
        if batch_count == batch_size:
            batch_count = 0
            yield np.asarray(batch_inputs), np.asarray(batch_targets)
            batch_inputs = []
            batch_targets = []

train_generator = get_generator(training_set, 50)
test_generator = get_generator(test_set, 10)
from keras.callbacks import TensorBoard
from time import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit_generator(**{
    'generator': train_generator,
    'steps_per_epoch': len(training_set) // 50,
    
    'validation_data': test_generator,
    'validation_steps': len(test_set) // 10,
    
    'workers': 3,
    'epochs': 3,
    
    'verbose': 1,
    'callbacks': [
        tensorboard,
    ]
})
from matplotlib.pyplot import imshow

img0 = get_data(image_paths[1])

%matplotlib inline
imshow(img0)