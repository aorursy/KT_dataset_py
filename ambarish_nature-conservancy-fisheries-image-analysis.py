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
train_path ='../input/train/'
listdir(train_path)[:10]
len(listdir(train_path))
test_path = '../input/test_stg1/'
listdir(test_path)[:10]
len(listdir(test_path))
submission = pd.read_csv('../input/sample_submission_stg1.csv')
submission.head()
from tqdm import tqdm
def loadBatchImages(path):
    catList = listdir(path)
    loadedImages = []
    loadedLabels = []
    for cat in catList:
        if not cat.startswith('.'):
            deepPath = path+cat+"/"
            imageList = listdir(deepPath)
            for images in tqdm(imageList):
                img = deepPath + images
                loadedLabels.append(cat)
                loadedImages.append(img)
            
    return loadedImages, loadedLabels
loadedImages, loadedLabels = loadBatchImages(train_path)
num_classes = len(np.unique(loadedLabels))
num_classes
#Encode labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
loadedLabels = np.asarray(loadedLabels)
encoder.fit(loadedLabels)
encoded_loadedLabels = encoder.transform(loadedLabels)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
labels_Hot = to_categorical(encoded_loadedLabels, num_classes = num_classes)
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
df= pd.DataFrame()
df['path']=loadedImages
df['labels'] = list(labels_Hot)
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
print('train', train_df.shape[0], 'validation', valid_df.shape[0])
train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'labels', 
                            target_size = IMG_SIZE,
                            batch_size = 128)

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
img_dim = t_x.shape[1:]
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score
num_classes = len(labels_Hot[0])
num_classes
input_tensor = Input(shape=img_dim)
base_model = VGG16(include_top=False,input_shape=img_dim)
    
bn = BatchNormalization()(input_tensor)
x = base_model(bn)
x = Flatten()(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(input_tensor, output)
model.summary()
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
from keras.optimizers import Adam
history = History()
callbacks = [history, 
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights.best.hdf5', verbose=1, save_best_only=True, 
                             save_weights_only=True, mode='auto')]
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])
batch_size = 128
steps_per_epoch = len(train_df)/batch_size
steps_per_epoch
model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,validation_data = (test_X, test_Y), 
                                  epochs = 25,callbacks = callbacks)
model.load_weights('weights.best.hdf5')
import glob
from glob import glob
test_image_paths = glob(test_path +'*.jpg', recursive=True)
test_image_paths[:10]
X_test = pd.DataFrame()
X_test['path'] = test_image_paths
X_test['image'] = X_test['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
X_test.head()
test_gen = flow_from_dataframe(core_idg, X_test, 
                             path_col = 'path',
                            y_col = 'image', 
                            target_size = IMG_SIZE,
                            batch_size = 256) # we can use much larger batches for evaluation
pred_Y =  model.predict_generator(test_gen,verbose = 1)
len(pred_Y)
pred_Y[0]
submission = pd.DataFrame()
X_test['image'].head()
submission['image'] = X_test['image']
unique_labels = np.unique(loadedLabels)
unique_labels
encoder.fit(unique_labels)
encoder.transform(unique_labels)
for i,label in enumerate(list(unique_labels)):
    print(i,label)
for i,label in enumerate(list(unique_labels)):
    submission[label] = pred_Y[:,i]
submission.head()
submission.to_csv('predictions.csv',index = False)