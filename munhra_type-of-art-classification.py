# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import sys
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from sklearn.model_selection import train_test_split


np.random.seed(42)

# Art categories
art_categories = ['sculpture', 'iconography', 'engraving', 'drawings', 'painting'] # mapped as 0, 1 ,2 ,3 and 4
crop_width_size = 50
crop_height_size = 50
def check_dataset_balance():
    for category in art_categories:
        print(category)
        file_list = os.listdir('../input/cropped-50-dataset/50_cropped_dataset_updated/50_cropped_dataset_updated/training_set/{}'.format(category))
        file_list_2 = os.listdir('../input/cropped-50-dataset-2/50_cropped_dataset_updated 2/50_cropped_dataset_updated 2/training_set/{}'.format(category))
        print('{} -> '.format(len(file_list_2)+len(file_list))+category)
check_dataset_balance()
batch_size = 128
num_classes = 5
epochs = 10
img_rows, img_cols = 50, 50
#Machine Learng Preparation for Art
def load_data():
    feature_vector = []
    target = []
    category_counter = 0
    
    for category in art_categories:

        file_list = os.listdir('../input/cropped-50-dataset/50_cropped_dataset_updated/50_cropped_dataset_updated/training_set/{}'.format(category))
        file_list_2 = os.listdir('../input/cropped-50-dataset-2/50_cropped_dataset_updated 2/50_cropped_dataset_updated 2/training_set/{}'.format(category))
        
        
        for i in tqdm(range(len(file_list)), file=sys.stdout):
            file_name = '../input/cropped-50-dataset/50_cropped_dataset_updated/50_cropped_dataset_updated/training_set/'+category+'/'+file_list[i]
            #print(file_list[i])
            raw_img = cv2.imread(file_name)
            if not raw_img is None:
                feature_vector.append(raw_img)
                target.append(category_counter)
        
        for i in tqdm(range(len(file_list_2)), file=sys.stdout):
            file_name = '../input/cropped-50-dataset-2/50_cropped_dataset_updated 2/50_cropped_dataset_updated 2/training_set/'+category+'/'+file_list_2[i]
            raw_img = cv2.imread(file_name)
            if not raw_img is None:
                feature_vector.append(raw_img)
                target.append(category_counter)

        print('Category {} mapped as {}'.format(category, category_counter))
        category_counter = category_counter + 1
        
    
    return feature_vector,target
#load_data()
def split_train_test_data():
    feature_vector, target = load_data()
    print(len(target))
    x_train, x_test, y_train, y_test = train_test_split(feature_vector, target, test_size=0.15)
    
    #print(len(x_train[0]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    print(y_train[0].shape)
                      
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    #input_shape = (img_rows, img_cols, 3)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #print(y_test)
    
    return x_train, y_train, x_test, y_test
#split_train_test_data()
x_train, y_train, x_test, y_test = split_train_test_data()
input_shape = (img_rows, img_cols, 3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.40)) #0.4 removed as it is bad of tflite # 0.25
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25)) #0.25 removed as it is bad of tflite # 0.5
model.add(Dense(num_classes, activation='softmax')) #sigmoid

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=5)