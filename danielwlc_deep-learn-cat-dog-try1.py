import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
%matplotlib inline 

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import keras

print("Keras -V: {}".format(keras.__version__))
print("OpenCV -V: {}".format(cv2.__version__))
from keras.optimizers import SGD, RMSprop, Adam

# config
FORCE_CACHE = False #True

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'
CACHE_DIR = '../working/'
CACHE_FILE = 'cache.hdf5'

RANDOM_SEED = 1980
CHUNK_SIZE = 2000
IMG_ROWS = 128
IMG_COLS = 128
IMG_CHANNELS = 3
PIXEL_DEPTH = 255

NB_EPOCH = 50
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = RMSprop() 
N_HIDDEN = 128
NB_CLASSES = 2
VALIDATION_SPLIT=0.2
DROPOUT = 0.3

INPUT_SHAPE = (IMG_CHANNELS, IMG_ROWS, IMG_COLS)
CACHE_FILE = CACHE_FILE[:-5] + str(IMG_ROWS) + '.hdf5'
CACHE = os.path.join(CACHE_DIR, CACHE_FILE)
from keras.utils import np_utils
from itertools import repeat
from sklearn.model_selection import train_test_split

np.random.seed(RANDOM_SEED)  # for reproducibility

# get the data
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]
test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

train_images = train_dogs + train_cats

# prepare target y
labels = list(repeat(1, len(train_dogs))) + list(repeat(0, len(train_cats)))
labels = np_utils.to_categorical(labels, NB_CLASSES)

# training/test Split
X_train, X_test, Y_train, Y_test = train_test_split(train_images, labels,
                                                    test_size=VALIDATION_SPLIT,
                                                    random_state=RANDOM_SEED)
import h5py

def read_images(image_files):
    #cv2.imread return BGR image
    images = [cv2.imread(file, cv2.IMREAD_COLOR) for file in image_files]
    #blobFromImages(swapRB=True) return RGB image
    data = cv2.dnn.blobFromImages(images, size=(IMG_ROWS,IMG_COLS), 
                                  scalefactor=1./PIXEL_DEPTH,
                                  swapRB=True, crop=False)
    
    #assert data is not None
    return data #RGB images

def create_hdf5_cache_file(chunksize, folder = './', file_path = 'data.hdf5'):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    with h5py.File(os.path.join(folder, file_path), "w") as f:
        x_list = [X_train, X_test]
        y_list = [Y_train, Y_test]
        group_list = ['train', 'test']

        for x, y, group in zip(x_list, y_list, group_list):
            print('Start processing subgroup: {}'.format(group))
            grp = f.create_group(group)
            count = len(x)
            dx = grp.create_dataset("dx",(count,)+INPUT_SHAPE,'float32')
            grp["dy"] = y
            for i in range(0, count, chunksize):
                print('Processing x: {} of {}'.format(i, count))
                dx[i:i+chunksize,:,:,:] = read_images(x[i:i+chunksize])
            print('Processed subgroup: {}'.format(group))

    print("Completed writing hdf5 file: {}{}".format(folder,file_path))

if FORCE_CACHE or not os.path.isfile(CACHE):
    create_hdf5_cache_file(CHUNK_SIZE, CACHE_DIR, CACHE_FILE)
else:
    print("Make use of original hdf5 file: {}{}".format(CACHE_DIR, CACHE_FILE))
from keras.utils import HDF5Matrix

class BatchDataGenerator(keras.utils.Sequence):
    def __init__(self, hdf5_file, group, batch_size, start=0, end=None):
        self.file = hdf5_file
        self.group = group
        self.batch_size = batch_size
        self.start = start
        if end is None:
            with h5py.File(hdf5_file, "r") as f:
                self.end = f[group]["dy"].len()
        else:
            self.end = end
            
        self.length = self.end - self.start
        self.start_list = list(range(self.start, self.end, self.batch_size))
        self.end_list = [j+self.batch_size-1 for j in self.start_list]
        self.end_list[-1] = min(self.end_list[-1], self.end)       
    
    def __len__(self):
        return int(np.ceil(self.length / float(self.batch_size)))
    
    def __getitem__(self, idx):
        params = {'start': self.start_list[idx],
                  'end': self.end_list[idx]}
        batch_x = HDF5Matrix(self.file, self.group+'/dx', **params)
        batch_y = HDF5Matrix(self.file, self.group+'/dy', **params)
        return batch_x, batch_y

def generate_full_data(hdf5_file, group, start=0, end=None):
    with h5py.File(hdf5_file, "r") as f:
        if end is None:
            end = f[group]["dy"].len()
        x = f[group]["dx"][start:end]
        y = f[group]["dy"][start:end]
    
    return x, y

def get_train_split_indices(hdf5_file, group, split_ratio):
    with h5py.File(hdf5_file, "r") as f:
        data_size = f[group]["dy"].len()
    
    return 0, int(data_size*(1-split_ratio)), int(data_size*(1-split_ratio))+1, data_size
train_st,train_ed,valid_st,valid_ed = get_train_split_indices(CACHE,'train',VALIDATION_SPLIT)

training_generator = BatchDataGenerator(CACHE,'train',BATCH_SIZE,train_st,train_ed)
validation_generator = BatchDataGenerator(CACHE,'train',BATCH_SIZE,valid_st,valid_ed)

validation_data = generate_full_data(CACHE,'train',valid_st,valid_ed)
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.core import Flatten

# model
K.set_image_dim_ordering("th")
#define the convnet 
class myNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        model.add(Conv2D(20, kernel_size=5, padding="same", 
                         input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(50, kernel_size=5, padding="same", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(classes, activation='sigmoid'))
        return model

# initialize the optimizer and model
model = myNet.build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)

model.summary()
from keras.callbacks import TensorBoard, EarlyStopping

model.compile(loss='binary_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

esCallBack = EarlyStopping(monitor='val_loss', patience=20,  
                           min_delta = 0, verbose = 0, mode = 'min')
history = model.fit_generator(training_generator,
                              epochs=NB_EPOCH,
                              validation_data=validation_data,
                              verbose=VERBOSE,
                              workers=4,
                              callbacks=[esCallBack])
X_test_data, Y_test_data = generate_full_data(CACHE,'test')

score = model.evaluate(X_test_data, Y_test_data, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()