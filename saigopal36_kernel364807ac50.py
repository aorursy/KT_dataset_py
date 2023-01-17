# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_files       

from keras.utils import np_utils

import numpy as np

from glob import glob

from sklearn.model_selection import train_test_split



# define function to load datasets

def load_dataset(path):

    data = load_files(path)

    files = np.array(data['filenames'])

    targets = np_utils.to_categorical(np.array(data['target']), 10)

    return files, targets



# load train, test, and validation datasets

train_files, train_targets = load_dataset('/kaggle/input/distracteddriverdetection/MDDS/train')

#valid_files, valid_targets = load_dataset('dogImages/valid')

test_files,test_targets= load_dataset('/kaggle/input/distracteddriverdetection/MDDS/test')





# load list of names

names = [item[17:19] for item in sorted(glob("/kaggle/input/distracteddriverdetection/MDDS/train/*/"))]



# break training set into training and validation sets

#(train_files, valid_files) = train_files[:18000], train_files[18000:]

#(train_targets, valid_targets) = train_targets[:18000], train_targets[18000:]

train_files, valid_files, train_targets, valid_targets = train_test_split(train_files, train_targets, test_size=0.2, random_state=42)



# print statistics about the dataset



print('There are %s total images.\n' % len(np.hstack([train_files, valid_files, test_files])))

print('There are %d training images.' % len(train_files))

print('There are %d total training categories.' % len(names))

print('There are %d validation images.' % len(valid_files))

print('There are %d test images.'% len(test_files))
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
from PIL import ImageFile                            

ImageFile.LOAD_TRUNCATED_IMAGES = True                 



# pre-process the data for Keras

train_tensors = paths_to_tensor(train_files).astype('float32')/255 - 0.5

valid_tensors = paths_to_tensor(valid_files).astype('float32')/255 - 0.5

test_tensors = paths_to_tensor(test_files).astype('float32')/255 - 0.5
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

from keras.applications.vgg16 import VGG16

model = VGG16(include_top=False)

model.summary()
bottleneck_features_train_VGG16 = np.asarray([model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in train_tensors],dtype=np.float32)

#bottleneck_features_train_VGG16 = model.predict_generator(generator, nb_train_samples // batch_size)

np.save(open('/kaggle/working/bottleneck_features_train_VGG16.npy', 'wb'),bottleneck_features_train_VGG16)





bottleneck_features_valid_VGG16 = np.asarray([model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in valid_tensors],dtype=np.float32)

#bottleneck_features_train_VGG16 = model.predict_generator(generator, nb_train_samples // batch_size)

np.save(open('/kaggle/working/bottleneck_features_valid_VGG16.npy', 'wb'),bottleneck_features_valid_VGG16)





bottleneck_features_test_VGG16 = np.asarray([model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in test_tensors],dtype=np.float32)

np.save(open('/kaggle/working/bottleneck_features_test_VGG16.npy', 'wb'),bottleneck_features_test_VGG16)
bottleneck_features = np.load('/kaggle/input/dogtrainedmodels/bottleneck_features/bottleneck_features/DogVGG16Data.npz')

train_VGG16 = bottleneck_features['train']

valid_VGG16 = bottleneck_features['valid']

test_VGG16 = bottleneck_features['test']
print(bottleneck_features_train_VGG16.shape)

print(bottleneck_features_valid_VGG16.shape)

print(bottleneck_features_test_VGG16.shape)
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dropout, Flatten, Dense

from keras.models import Sequential

VGG16_model = Sequential()

VGG16_model.add(GlobalAveragePooling2D(input_shape=bottleneck_features_train_VGG16.shape[1:]))

VGG16_model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))



VGG16_model.summary()
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='/kaggle/working/weights.best.VGG16.hdf5_transfer_learning', 

                               verbose=1, save_best_only=True)



VGG16_model.fit(bottleneck_features_train_VGG16, train_targets, 

          validation_data=(bottleneck_features_valid_VGG16, valid_targets),

          epochs=400, batch_size=16, callbacks=[checkpointer], verbose=1)
VGG16_model.load_weights('/kaggle/working/weights.best.VGG16.hdf5_transfer_learning')
VGG16_predictions = [VGG16_model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in bottleneck_features_test_VGG16]
test_files_final = [item_test[15:] for item_test in test_files]

VGG16_subm = np.column_stack((np.asarray(test_files_final), np.asarray(VGG16_predictions,dtype=np.float32)))
print(VGG16_subm[1:3])


from keras.applications.vgg16 import VGG16

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Dropout, Flatten, Dense



from keras.applications.vgg16 import VGG16

base_model = VGG16(include_top=False)

base_model.summary()
VGG16_top_model = Sequential()

VGG16_top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))

VGG16_top_model.add(Dense(10, activation='softmax', kernel_initializer='glorot_normal'))





VGG16_top_model.load_weights('/kaggle/working/weights.best.VGG16.hdf5_transfer_learning')



model = Model(input= base_model.input, output= VGG16_top_model(base_model.output))



VGG16_top_model.summary()
for layer in model.layers[:15]:

    layer.trainable = False



# compile the model with a SGD/momentum optimizer

# and a very slow learning rate.

model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint



checkpointer = ModelCheckpoint(filepath='/kaggle/working/weights.best.VGG16.hdf5_fine_tuning', 

                               verbose=1, save_best_only=True)



model.fit(train_tensors, train_targets, 

          validation_data=(valid_tensors, valid_targets),

          epochs=10, batch_size=16, callbacks=[checkpointer], verbose=1)
model.load_weights('/kaggle/working/weights.best.VGG16.hdf5_fine_tuning')
VGG16_predictions_fine_tuned = [model.predict(np.expand_dims(tensor, axis=0))[0] for tensor in test_tensors]
VGG16_subm_fine_tuned = np.column_stack((np.asarray(test_files_final), np.asarray(VGG16_predictions_fine_tuned,dtype=np.float32)))
print(VGG16_subm_fine_tuned[1:3])