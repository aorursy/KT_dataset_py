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
from os import listdir
listdir('../input/train-jpg')[:10]
len(listdir('../input/train-jpg'))
test_path = '../input/test-jpg-v2/'
listdir(test_path)[:10]
len(listdir('../input/test-jpg-v2'))
import pandas as pd
labels_df = pd.read_csv('../input/train_v2.csv')
labels_df.head()
# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_list[:10]
labels_set = set(labels_list)
len(labels_list)
import numpy as np
len(np.unique(np.array(labels_list)))
len(labels_set)
labels_set
for c_label in labels_set:
    labels_df[c_label] = labels_df['tags'].map(lambda finding: 1.0 if c_label in finding else 0)
labels_df.head()
labels_set = list(labels_set)
labels_df[labels_set].values
len(labels_df) == len(labels_df[labels_set]) 
labels_df.apply(lambda x: [x[labels_set].values],1)[:10]
labels_df['labels_vec'] = labels_df.apply(lambda x: [x[labels_set].values],1).map(lambda x: x[0])
labels_df['labels_vec'].head()
labels_df.head()
train_data = pd.DataFrame()
train_data['image_path'] = listdir('../input/train-jpg')
train_data['image_name'] = train_data['image_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
train_data.head()
train_data = pd.merge(train_data,labels_df)
train_data.head()
train_data['path'] = '../input/train-jpg/' + train_data['image_path']
train_data.head()
from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (128, 128)
core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
path_col = 'path'
base_dir = os.path.dirname(train_data[path_col].values[0])
base_dir
def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(train_data, 
                                   test_size = 0.25, 
                                   random_state = 2018)
print('train', train_df.shape[0], 'validation', valid_df.shape[0])
train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'labels_vec', 
                            target_size = IMG_SIZE,
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'labels_vec', 
                            target_size = IMG_SIZE,
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'labels_vec', 
                            target_size = IMG_SIZE,
                            batch_size = 1024)) # one big batch
t_x, t_y = next(train_gen)
t_x.shape[1:]
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
%matplotlib inline
num_classes = len(labels_set)
pretrained_model_1 = VGG16(include_top=False, input_shape=t_x.shape[1:])
base_model = pretrained_model_1 # Topless
optimizer1 = keras.optimizers.Adam()
# Add top layer
x = base_model.output
x = Conv2D(100, kernel_size = (3,3), padding = 'valid')(x)
x = Flatten()(x)
x = Dropout(0.75)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# Train top layer
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer1, 
              metrics=['accuracy'])
model.summary()
model.fit_generator(train_gen,steps_per_epoch=100,validation_data = (test_X, test_Y), 
                                  epochs = 10)
import glob
from glob import glob
test_image_paths = glob(test_path +'*.jpg', recursive=True)
test_image_paths[:10]
X_test = pd.DataFrame()
X_test['path'] = test_image_paths
submission = pd.read_csv('../input/sample_submission_v2.csv')
submission.head()
X_test['image_name'] = X_test['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
X_test.head()
test_gen = flow_from_dataframe(core_idg, X_test, 
                             path_col = 'path',
                            y_col = 'image_name', 
                            target_size = IMG_SIZE,
                            batch_size = 256) # we can use much larger batches for evaluation
pred_Y =  model.predict_generator(test_gen,verbose = 1)
y_map ={i:l for i,l in enumerate(labels_set)} 
y_map
train_data.head()
thresholds = [0.2] * len(labels_set)
thresholds
i = 0
for prediction in pred_Y:
    i = i+1
    if i == 10:
        break;
    print(prediction)
type(prediction)
prediction
for i, value in enumerate(prediction):
    print(i, value)
len(prediction)
len(thresholds)
len(y_map)
predictions_label = []
for prediction in pred_Y:
    labels = [y_map[i] for i,value in enumerate(prediction) if value > thresholds[i]]
    predictions_label.append(labels)
predictions_label[:10]
submission = pd.DataFrame()
submission['image_name'] = X_test['image_name']
tags_list = [None] * len(predictions_label)
for i, tags in enumerate(predictions_label):
    tags_list[i] = ' '.join(map(str, tags))
submission['tags'] = tags_list
submission.head()
submission.to_csv('predictions.csv',index = False)