# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import glob
from glob import glob

# Any results you write to the current directory are saved as output.
train_path = '../input/train'
path_name = train_path + '/**/*.jpg'
train_image_paths = glob(path_name, recursive=True)
train_image_paths[:10]
train_categories = list(map(os.path.basename,train_image_paths))
train_categories[:3]
labels =[]
for category in train_categories:
    labels.append(category[:3])
labels[:10]
len(labels)
len(train_image_paths)
num_classes = len(np.unique(labels))
num_classes
#Encode labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
loadedLabels = np.asarray(labels)
encoder.fit(loadedLabels)
encoded_loadedLabels = encoder.transform(loadedLabels)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
labels_Hot = to_categorical(encoded_loadedLabels, num_classes = num_classes)
labels_Hot[:3]
labels[:3]
df= pd.DataFrame()
df
df['path']=train_image_paths
df['path'].head()
df['labels'] = list(labels_Hot)
df.head()
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
train_df, valid_df = train_test_split(df, 
                                   test_size = 0.25, 
                                   random_state = 2018)
len(train_df)
len(valid_df)
train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'labels', 
                            target_size = IMG_SIZE,
                            batch_size = 32)

valid_gen = flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'labels', 
                            target_size = IMG_SIZE,
                            batch_size = 256) # we can use much larger batches for evaluation
# used a fixed dataset for evaluating the algorithm
test_X, test_Y = next(flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'labels', 
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
test_image_paths = glob('../input/test/*.jpg', recursive=True)
test_image_paths[0:3]
X_test = pd.DataFrame()
X_test['path'] = test_image_paths
X_test['labels'] = X_test['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
X_test[0:3]
test_gen = flow_from_dataframe(core_idg, X_test, 
                             path_col = 'path',
                            y_col = 'labels', 
                            target_size = IMG_SIZE,
                            batch_size = 256) # we can use much larger batches for evaluation
len(X_test)
pred_Y =  model.predict_generator(test_gen,verbose = 1)
len(test_image_paths)
len(pred_Y)
predictions = pred_Y[:,1]
predictions[0:3]
len(predictions)
submission=pd.DataFrame()
submission["id"]=X_test["labels"]
submission["label"] = predictions
submission.to_csv("predictions.csv",index=False)
