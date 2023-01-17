# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
from keras.preprocessing import image as Kimage
from keras.utils import np_utils
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint  
from keras import optimizers

from tqdm import tqdm
import pydicom
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
model_type = 'FC1024'
dropout = 0.2
optimizer_type = 'Adam'
learning_rate = 1e-4
Augmentation_Indicator = False
epochs = 20
batch_size = 8
transfer_learning = True
random_state = 1607
# Load train labels
input_data = pd.read_csv("../input/stage_1_detailed_class_info.csv")
input_data['img_path'] = '../input/stage_1_train_images/' + input_data['patientId'] + '.dcm'

# Convert class into categorical variable
input_data['class'] = pd.Categorical(input_data['class'])
input_data['target'] = input_data['class'].cat.codes

# Start with around 1500 - 2000 images 
# This step is not needed when it is going to be trained with full resources 
remove, input_data  = train_test_split(input_data, 
                                test_size=0.01, 
                                random_state=random_state,
                                stratify=input_data['class'])

print('Total images taken: {}'.format(input_data.shape[0]))
input_data.head()
# Split train and test images
train, test = train_test_split(input_data, 
                                test_size=0.20, 
                                random_state=random_state,
                                stratify=input_data['class'])

# Split train and validation images
train, valid = train_test_split(train, 
                                test_size=0.20, 
                                random_state=random_state,
                                stratify=train['class'])

print('Total train images taken: {}'.format(train.shape[0]))
print('Total validation images taken: {}'.format(valid.shape[0]))
print('Total test images taken: {}'.format(test.shape[0]))

# Show the class balance in train images
fig = plt.figure(figsize=(4,4), dpi=100)
ax = plt.subplot(111)
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
train['class'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', 
                                   startangle=90, fontsize=10, colors = colors)

ax = plt.subplot(111)
train['class'].value_counts().plot(kind='bar', ax=ax, color=colors)
"""fig, axs = plt.subplots(1,2, dpi=120, figsize=(4,4))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
train['class'].value_counts().plot(kind='pie', ax=axs[0], autopct='%1.1f%%', 
                                   startangle=90, fontsize=10, colors = colors)
train['class'].value_counts().plot(kind='bar', ax=axs[1], color=colors)
#plt.legend(loc="right", fontsize=10)
#fig.subplots_adjust(wspace=2)"""
for i, row in enumerate(train.head().values):
    image_name = row[0]
    pneumonia_class = row[1]
    image_path = row[2]
    ds = pydicom.dcmread(image_path)
    plt.figure()
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.title(pneumonia_class)
    #print(pneumonia_class)
"""# Print sample images
im_per_row = 4
length = train.head().values.shape[0]
#fig, ax = plt.subplots(2, im_per_row, figsize=(6,6), dpi=100)
fig = plt.figure()
j = 1
for i, row in enumerate(train.head().values):
    image = row[0]
    pneumonia_class = row[1]
    ds = pydicom.dcmread('../input/stage_1_train_images/'+image+'.dcm')
    fig.add_subplot(j,4,i+1, sharex=True)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.title(pneumonia_class)
    if i % length == 0:
        j+=1
    #print(pneumonia_class)"""
def load_dicom_image(img_path):
    img_arr = pydicom.read_file(img_path).pixel_array
    img_arr = img_arr/img_arr.max()
    slice_value = (255*img_arr).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(slice_value)
    Kimage.pil_image = img
    return Kimage.pil_image

# Convert 3D tensors to 4D tensors where each 4D tensor is a different image
def path_to_tensor(img_path):
    # Read the dcm image using pydicom
    img = load_dicom_image(img_path)
    # convert PIL.Image.Image type to 3D tensor
    x = Kimage.img_to_array(img)
    # Since it is a grayscale image convert into three channels
    x = np.squeeze(np.repeat(x[:, :, np.newaxis], 3, axis=2), axis=3)
    # convert 3D tensor to 4D tensor with shape and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [preprocess_input(path_to_tensor(img_path)) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# Load all the tensors and re-scale the data
train_tensors = paths_to_tensor(train['img_path'])
valid_tensors = paths_to_tensor(valid['img_path'])
test_tensors = paths_to_tensor(test['img_path'])

# Load all the targets
train_targets = np_utils.to_categorical(np.array(train['target']), 3)
valid_targets = np_utils.to_categorical(np.array(valid['target']), 3)
test_targets = np_utils.to_categorical(np.array(test['target']), 3)
test_tensors.shape
if transfer_learning:
    # Load Xception model from keras
    base_model = Xception(input_shape=(1024, 1024, 3), weights='imagenet', include_top=False)
    base_model.trainable = False
    input_shape=base_model.get_output_shape_at(0)[1:]
    learning_name = 'XceptionTransferLearning'
else:
    input_shape=(1024, 1024, 3)
    learning_name = 'OwnCNN'
if model_type == 'FC1024':
    # Build the final layer of the model
    model = Sequential()

    model.add(Conv2D(filters=1024, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters=512, kernel_size=2, padding='same', activation='tanh'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(3, activation='softmax'))

    model.summary()
elif model_type == 'FC16':
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(dropout))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='tanh'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(Dropout(dropout))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(3, activation='softmax'))

    model.summary()
else:
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=input_shape))
    model.add(Dense(3, activation='softmax'))

    model.summary()
if transfer_learning:# Combine pre-trained model and customized final layers
    final_model = Sequential(name='Pneumonia Classifier')
    final_model.add(base_model)
    final_model.add(model)
else:
    final_model = model
    
final_model.summary()
# Compile the model
if optimizer_type == 'SGD':
    optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9)
elif optimizer_type == 'Adam':
    optimizer = optimizers.Adam(lr=learning_rate)
else:
    optimizer = optimizers.RMSprop()
    
final_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# create and configure augmented image generator
datagen_train = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# create and configure augmented image generator
datagen_valid = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
# fit augmented image generator on data
datagen_train.fit(train_tensors)
datagen_valid.fit(valid_tensors)
os.mkdir('/kaggle/working/saved-models')
if Augmentation_Indicator:
    model_weights_name = 'weights.best.{}_wAug_{}_{}_{}_{}_{}_{}.hd5'.format(learning_name, model_type, dropout, optimizer_type,learning_rate, epochs, batch_size)
else:
    model_weights_name = 'weights.best.{}_woAug_{}_{}_{}_{}_{}_{}.hd5'.format(learning_name, model_type, dropout, optimizer_type,learning_rate, epochs, batch_size)
print(model_weights_name)
from keras.callbacks import ModelCheckpoint  

checkpointer = ModelCheckpoint(filepath='/kaggle/working/saved-models/{}'.format(model_weights_name), 
                               verbose=1, save_best_only=True)

if Augmentation_Indicator:
    final_model.fit_generator(datagen_train.flow(train_tensors, train_targets, batch_size=batch_size),
                                                steps_per_epoch=train_tensors.shape[0] // batch_size,
                                                epochs=epochs, verbose=1, callbacks=[checkpointer],
                                                validation_data=datagen_valid.flow(valid_tensors, valid_targets, batch_size=batch_size),
                                                validation_steps=valid_tensors.shape[0] // batch_size)
else:
    final_model.fit(train_tensors, train_targets, 
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)



final_model.load_weights('/kaggle/working/saved-models/{}'.format(model_weights_name))
# get index of predicted value for each image in test set
predictions = [np.argmax(final_model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
from sklearn.metrics import classification_report
target_names = ['Lung Opacity', 'No Lung Opacity / Not Normal', 'Normal']
print(classification_report(np.array(predictions), np.argmax(test_targets, axis=1), target_names = target_names))
test_accuracy_dict = {}
for model_weights in os.listdir('/kaggle/working/saved-models/'):
    print(model_weights)
    test_accuracy_dict[model_weights] = test_accuracy
test_accuracy_dict
