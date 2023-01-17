# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import pathlib
import random
import numpy as np
from numpy import save
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.applications import imagenet_utils  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
# Data train
# Take image path
data_train = pathlib.Path('../input/my-data-waste/DATASET/DATASET/TRAIN')
image_path = list(data_train.glob('*/*'))
image_path = [str(path) for path in image_path]
random.shuffle(image_path)
# Take labels train
labels = [p.split(os.path.sep)[-2] for p in image_path]
# convert labels to number
lab = LabelEncoder()
labels = lab.fit_transform(labels)

print(len(image_path))
# Data val
# Take image path
data_val = pathlib.Path('../input/my-data-waste/DATASET/DATASET/VAL')
image_path_val = list(data_val.glob('*/*'))
image_path_val = [str(path) for path in image_path_val]
random.shuffle(image_path)
# Take labels train
labels = [p.split(os.path.sep)[-2] for p in image_path_val]
# convert labels to number
lab = LabelEncoder()
labels_val = lab.fit_transform(labels)

print(len(image_path_val))
# Data test
# Take image path
data_test = pathlib.Path('../input/my-data-waste/DATASET/DATASET/TEST')
image_path_test = list(data_test.glob('*/*'))
image_path_test = [str(path) for path in image_path_test]
random.shuffle(image_path)
# Take labels train
labels = [p.split(os.path.sep)[-2] for p in image_path_test]
# convert labels to number
lab = LabelEncoder()
labels_test = lab.fit_transform(labels)

print(len(image_path_test))
def load_resize_image(image_path):
  list_image = []
  for (j, imagePath) in enumerate(image_path):
      image = load_img(imagePath, target_size=(224, 224))
      image = img_to_array(image)
      image = np.expand_dims(image, 0)
      image = imagenet_utils.preprocess_input(image)
      list_image.append(image)
  list_image = np.vstack(list_image)
  return list_image
x_train = np.load('../input/my-data-waste/x_train.npy')
print(x_train.shape)
x_val = load_resize_image(image_path_val)
x_test = load_resize_image(image_path_test)
y_train = np.load('../input/my-data-waste/y_train.npy')
y_val = labels_val
y_test = labels_test
model = VGG16(weights='imagenet', include_top=False)
features2 = model.predict(x_val)
x_val1 = features2.reshape((features2.shape[0], 512*7*7))
features3 = model.predict(x_test)
x_test1 = features3.reshape((features3.shape[0], 512*7*7))
print('X train: ',x_train.shape)
print('Y train: ',y_train.shape)
print('X valid: ',x_val1.shape)
print('Y valid: ',y_val.shape)
print('X test: ',x_test1.shape)
print('Y test: ',y_test.shape)
# Train model by Logistic Regression
params = {'C' : [0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LogisticRegression(), params)
model.fit(x_train, y_train)
print('Best parameter for the model {}'.format(model.best_params_))
y_pred = model.predict(x_val1)
print(classification_report(y_val, y_pred))
from sklearn.metrics import roc_auc_score

print('AUC: ',roc_auc_score(y_val, y_pred))
y_predict = model.predict(x_test1)
print(classification_report(y_test, y_predict))
from sklearn.metrics import roc_auc_score

print('AUC: ',roc_auc_score(y_test, y_predict))
import pickle

pickle.dump(model, open('vgg16_logistic', 'wb'))
train_gen = ImageDataGenerator()
valid_gen = ImageDataGenerator()

train_dir = '../input/my-data-waste/DATASET/DATASET/TRAIN/'
valid_dir = '../input/my-data-waste/DATASET/DATASET/VAL'

bs = 128
train_generator = train_gen.flow_from_directory(train_dir, batch_size = bs, target_size = (224, 224), class_mode = 'binary')
valid_generator = valid_gen.flow_from_directory(valid_dir, batch_size = bs, target_size = (224, 224), class_mode = 'binary')


print(train_generator)
print(valid_generator)
baseModel = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Built new model base on VGG16
fcHead = baseModel.output
# Add flatten layer
fcHead = BatchNormalization(name='BN')(fcHead)
fcHead = Flatten(name='flatten')(fcHead)
# Add FC
fcHead = Dense(512, activation='relu')(fcHead)
fcHead = Dense(256, activation='relu')(fcHead)
fcHead = Dense(128, activation='relu')(fcHead)
fcHead = Dropout(0.5)(fcHead)
# Output layer with softmax activation
fcHead = Dense(2, activation='softmax')(fcHead)

model = Model(inputs=baseModel.input, outputs=fcHead)
model.summary()
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

for layer in baseModel.layers:
    layer.trainable = False
    
opt = SGD(lr=1e-4, momentum=0.7)
model.compile(opt, 'sparse_categorical_crossentropy', ['accuracy'])
numOfEpoch = 40
len_train = 20064
len_valid = 2500

checkpoint = ModelCheckpoint(filepath = 'vgg16_ft.hdf5', verbose = 1, save_best_only = True)
earlystop = EarlyStopping(monitor = 'val_loss', # value being monitored for improvement
                          min_delta = 0, #Abs value and is the min change required before we stop
                          patience = 15, #Number of epochs we wait before stopping 
                          verbose = 1,
                          restore_best_weights = True) #keeps the best weigths once stopped
ReduceLR = ReduceLROnPlateau(patience=3, verbose=1)
callbacks = [earlystop, checkpoint, ReduceLR]                          
H = model.fit_generator(train_generator, 
                        steps_per_epoch=len_train//bs,
                        validation_data=valid_generator,
                        validation_steps=len_valid//bs,
                        epochs=numOfEpoch,
                        callbacks = callbacks,
                        use_multiprocessing=True,
                        workers=0)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validate'], loc='upper left')
plt.show()
y_predict = model.predict(x_test)
y_pred = list()
for i in range(0,2513):
  if y_predict[i][0]>y_predict[i][1]:
    y_pred.append(0)
  else:
    y_pred.append(1)
y = np.asarray(y_pred, dtype=np.float32)
from sklearn.metrics import roc_auc_score

print('AUC: ',roc_auc_score(y_test, y))
print(classification_report(y_test, y))
