# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
samples  = pd.read_csv('../input/sample_submission.csv')
samples.head()

CLASS = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11
}
INV_CLASS = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}

import os
import cv2
train_paths = {'fpath':[],
               'imgname':[],
                'label':[] }
test_paths = {'fpath':[],
              'imgname':[]  }
for root,dirs,files in os.walk('../input'):
    for f in files:
        fpath = os.path.join(root,f)
        if 'train' in root:
            imgname = fpath.split('/')[-1]
            label = fpath.split('/')[-2]
            train_paths['imgname'].append(imgname)
            train_paths['label'].append(label)
            train_paths['fpath'].append(fpath)
        elif 'test' in root:
            imgname = fpath.split('/')[-1]
            test_paths['imgname'].append(imgname)
            test_paths['fpath'].append(fpath)         
print(train_paths['imgname'][1] , train_paths['fpath'][1] , train_paths['label'][1])
from sklearn.utils import shuffle
#a = np.array([1,2,3,4])
#b = np.array([1,2,3,4])
#a , b = shuffle(a,b,random_state = 123)
#print(a) Output : [4 1 2 3]
#print(b) Output : [4 1 2 3]
train_paths['fpath'],train_paths['imgname'],train_paths['label'] = shuffle(train_paths['fpath'],train_paths['imgname'],train_paths['label'],random_state = 123)
#print(train_paths['label'][1] , train_paths['label'][2],train_paths['label'][3])
#Output : Common Chickweed Charlock Loose Silky-bent
from matplotlib.pyplot import imshow
img_test = cv2.imread(train_paths['fpath'][1])
imshow(img_test)
print(train_paths['label'][1])
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_paths['label'])
encoded_Y = encoder.transform(train_paths['label'])
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y[1])
print(encoded_Y[1])
print(np.argmax(dummy_y[1]))
org_y = encoder.inverse_transform(encoded_Y)
print(org_y[1])
from skimage.transform import resize
def batch_generator(paths):
    counter = 25
    image_inp = []
    output_label = []
    while counter<=len(paths['fpath']):
        for i in range(counter-25 , counter):
            image = cv2.imread(paths['fpath'][i])
            image = resize(image,(200,200,3))
            output = dummy_y[i]
            image_inp.append(image)
            output_label.append(output)
        counter +=25
        yield(np.asarray(image_inp), np.asarray(output_label))
        
       # counter +=25
       # image_inp = []
       # output_label = []
#def array_gen(inp_array):
 #   for i in range(len(inp_array)):
  #      yield(inp_array[i])
#tatti = [1,2,3,4,5,6]
#chut = array_gen(tatti)
#print(next(chut))
#print(next(chut))
#print(next(chut))
#print(next(chut))
first_batch = batch_generator(train_paths)
images , output_labels= next(first_batch)
print(output_labels)
print(images.shape)
imshow(images[1])

from keras.utils import plot_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Maximum
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
model = Sequential()
model.add(Conv2D(120,(3,3),strides=(1,1) , padding = 'valid' , data_format='channels_last' , input_shape = (200,200,3)))
model.add(MaxPooling2D(pool_size=(2,2) ,padding='valid' , strides=None ))
model.add(BatchNormalization())
model.add(Conv2D(120 , (3,3) , padding = 'same'))
model.add(MaxPooling2D(pool_size = (3,3) , padding = 'valid' , strides = None))
model.add(Flatten())
model.add(Dense(50 , activation='relu'))
model.add(Dropout(.25))
model.add(Dense(12 , activation = 'softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd , loss = 'categorical_crossentropy' , metrics = ['acc'])
batches = batch_generator(train_paths)
model.fit_generator(batches,steps_per_epoch=190 , epochs = 4 , verbose=1 , use_multiprocessing=False, initial_epoch=0)