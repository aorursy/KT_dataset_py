# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_files       
from keras.utils import np_utils

from glob import glob

import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint  

import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings

import ast
import math

# Have you installed your own package in Kernels yet? 
# If you need to, you can use the "Settings" bar on the right to install `simplification`
from simplification.cutil import simplify_coords

import matplotlib.pyplot as plt
import matplotlib.style as style
import glob
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
# define function to load train, test, and validation datasets
#def load_dataset(path):
#    data = load_files(path)
#    dog_files = np.array(data['filenames'])
#    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
#    return dog_files, dog_targets

file_list = glob.glob('../input/train_simplified' + '/*.csv')

train_files = []
train_targets = []
for x in file_list[:10]:
    train_data = pd.read_csv(x)
    train_data["word"] = train_data["word"].replace(' ', '_', regex=True)
    train_files.append(train_data["drawing"])
    train_targets.append(train_data["word"])

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer

y_train = np.asarray(train_targets)
X_train = np.asarray(train_files)

print(train_data["drawing"][0].shape)
print(X_train.shape)
print(y_train.shape)
#one_hot = MultiLabelBinarizer()

# One-hot encode data
#one_hot.fit_transform(y_train)

#one_hot.transform(y_train)


# integer encoding
#label_encoder = LabelEncoder()
#integer_encoded = label_encoder.fit_transform(y_train)
#print(integer_encoded)

# binary encoding
#onehot_encoder = OneHotEncoder(sparse=False)
#integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

#encoded = to_categorical(integer_encoded, num_classes=10)
#print(y_train)


class_targets.head
#print(type(y_train))
#file_list = glob.glob('../input/train_simplified' + '/*.csv')
#dfs=np.array([pd.read_csv(fp).values for fp in file_list])

#data = pd.read_csv('../input/train_simplified/roller coaster.csv',
#                   index_col='key_id',
#nrows=100)#
#train_data = []
#i=0
#for i in file_list:
#    train_data = [pd.read_csv(i).values for fp in file_list]
    
#train_data = pd.read_csv(file_list[0])
    
    #classes[i] = file_list[i]
    #classes[i] = classes[i].replace('../input/train_simplified/', '', regex=True)
#print(train_data.head())
#print(file_list[0])
#print(train_data["drawing"][0])
#print(file_list)
#data.head()
data['word'] = data['word'].replace(' ', '_', regex=True)
data.head()
test_raw = pd.read_csv('../input/test_raw.csv', index_col='key_id')
first_ten_ids = test_raw.iloc[:10].index
raw_images = [ast.literal_eval(lst) for lst in test_raw.loc[first_ten_ids, 'drawing'].values]
def resample(x, y, spacing=1.0):
    output = []
    n = len(x)
    px = x[0]
    py = y[0]
    cumlen = 0
    pcumlen = 0
    offset = 0
    for i in range(1, n):
        cx = x[i]
        cy = y[i]
        dx = cx - px
        dy = cy - py
        curlen = math.sqrt(dx*dx + dy*dy)
        cumlen += curlen
        while offset < cumlen:
            t = (offset - pcumlen) / curlen
            invt = 1 - t
            tx = px * invt + cx * t
            ty = py * invt + cy * t
            output.append((tx, ty))
            offset += spacing
        pcumlen = cumlen
        px = cx
        py = cy
    output.append((x[-1], y[-1]))
    return output
  
def normalize_resample_simplify(strokes, epsilon=1.0, resample_spacing=1.0):
    if len(strokes) == 0:
        raise ValueError('empty image')

    # find min and max
    amin = None
    amax = None
    for x, y, _ in strokes:
        cur_min = [np.min(x), np.min(y)]
        cur_max = [np.max(x), np.max(y)]
        amin = cur_min if amin is None else np.min([amin, cur_min], axis=0)
        amax = cur_max if amax is None else np.max([amax, cur_max], axis=0)

    # drop any drawings that are linear along one axis
    arange = np.array(amax) - np.array(amin)
    if np.min(arange) == 0:
        raise ValueError('bad range of values')

    arange = np.max(arange)
    output = []
    for x, y, _ in strokes:
        xy = np.array([x, y], dtype=float).T
        xy -= amin
        xy *= 255.
        xy /= arange
        resampled = resample(xy[:, 0], xy[:, 1], resample_spacing)
        simplified = simplify_coords(resampled, epsilon)
        xy = np.around(simplified).astype(np.uint8)
        output.append(xy.T.tolist())

    return output
simplified_drawings = []
for drawing in raw_images:
    simplified_drawing = normalize_resample_simplify(drawing)
    simplified_drawings.append(simplified_drawing)
for index, raw_drawing in enumerate(raw_images, 0):
    
    plt.figure(figsize=(6,3))
    
    for x,y,t in raw_drawing:
        plt.subplot(1,2,1)
        plt.plot(x, y, marker='.')
        plt.axis('off')

    plt.gca().invert_yaxis()
    plt.axis('equal')

    for x,y in simplified_drawings[index]:
        plt.subplot(1,2,2)
        plt.plot(x, y, marker='.')
        plt.axis('off')

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()  

# define function to load train, test, and validation datasets
#def load_dataset(path):
#    data = load_files(path)
#    dog_files = np.array(data['filenames'])
#    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
#    return dog_files, dog_targets

train_data = pd.read_csv('../input/train_simplified')

# load train, test, and validation datasets
train_files = data["drawing"]
train_targets = data["word"]

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(filters =16,kernel_size = 2, padding = 'same',activation = 'relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters =32,kernel_size = 2, padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters =64,kernel_size = 2, padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters =16,kernel_size = 2, padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters =16,kernel_size = 2, padding = 'same',activation = 'relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())


model.add(Dense(133, activation='softmax'))

### TODO: Define your architecture.

model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 50

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
### TODO: Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load('/data/bottleneck_features/DogInceptionV3Data.npz')
train_inception = bottleneck_features['train']
valid_inception = bottleneck_features['valid']
test_inception = bottleneck_features['test']

print(test_inception.shape)
### TODO: Define your architecture.
inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_inception.shape[1:]))
inception_model.add(Dense(133, activation='softmax'))

inception_model.summary()
### TODO: Compile the model.
inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
### TODO: Train the model.
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

inception_model.fit(train_inception, train_targets, 
          validation_data=(valid_inception, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
### TODO: Load the model weights with the best validation loss.
inception_model.load_weights('saved_models/weights.best.InceptionV3.hdf5')
### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
inception_predictions = [np.argmax(inception_model.predict(np.expand_dims(feature, axis=0))) for feature in test_inception]

# report test accuracy
test_accuracy = 100*np.sum(np.array(inception_predictions)==np.argmax(test_targets, axis=1))/len(inception_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
submission = pd.read_csv('../input/sample_submission.csv', index_col='key_id')
# Don't forget, your multi-word labels need underscores instead of spaces!
my_favorite_words = ['donut', 'roller_coaster', 'smiley_face']  
submission['word'] = " ".join(my_favorite_words)
submission.to_csv('my_favorite_words.csv')
submission.head()
