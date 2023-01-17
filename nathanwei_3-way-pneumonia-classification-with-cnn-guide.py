#import stuff

import os

import glob

import h5py

import shutil

import imgaug as aug #augment data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt #plot stuff/show images

import matplotlib.image as mimg

import imgaug.augmenters as iaa #augment data(data is imbalanced)

from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path #get data

from skimage.io import imread

from skimage.transform import resize

from keras.models import Sequential, Model

from keras.applications.vgg16 import VGG16, preprocess_input #transfer learning

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D

#convolutional nueral networks

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

import cv2

from keras import backend as K

color = sns.color_palette()

%matplotlib inline

print(os.listdir("../input"))
import tensorflow as tf



device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))

tf.config.experimental.list_physical_devices('GPU')
#get data

data_dir1 = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')

data_dir2 = Path('../input/covid19-detection-xray-dataset')

data_dir3 = Path('../input/covid19-radiography-database/COVID-19 Radiography Database')

data_dir4 = Path('../input/pneumonia-virus-vs-covid19/Pneumonia_and_COVID19')
data = []

normal_data = []

bacterial_data = []

viral_data = []
loop_dir1 = ['test', 'train', 'val']

for i in loop_dir1:

    normal_dir1 = data_dir1 / i / 'NORMAL'

    pneumonia_dir1 = data_dir1 / i / 'PNEUMONIA'

    normal_cases1 = normal_dir1.glob('*.jpeg')

    pneumonia_cases1 = pneumonia_dir1.glob('*.jpeg')

    for img in normal_cases1:

        normal_data.append((img,0))

    for img in pneumonia_cases1:

        if "bacteria" in str(img):

            bacterial_data.append((img,1))

        elif "virus" in str(img):

            viral_data.append((img,2))

    
loop_dir2 = ['NonAugmentedTrain', 'ValData']

for i in loop_dir2:

    normal_dir2 = data_dir2 / i / 'Normal'

    bacterial_dir2 = data_dir2 / i / 'BacterialPneumonia'

    viral_dir2 = data_dir2 / i / 'ViralPneumonia'

    normal_cases2 = normal_dir2.glob('*.jpeg')

    bacterial_cases2 = bacterial_dir2.glob('*.jpeg')

    viral_cases2 = viral_dir2.glob('*.jpeg')

    for img in normal_cases2:

        normal_data.append((img,0))

    for img in bacterial_cases2:

        bacterial_data.append((img,1))

    for img in viral_cases2:

        viral_data.append((img,2))
# normal_dir3 = data_dir3 / 'NORMAL'

# viral_dir3 = data_dir3 / 'Viral Pneumonia'

# normal_cases3 = normal_dir3.glob('*.png')

# viral_cases3 = viral_dir3.glob('*.png')

# for i, img in enumerate(normal_cases3):

#     im = Image.open(img)

#     rgb_img = im.convert('RGB')

#     rgb_img.save('normal('+str(i)+').jpeg', quality=100, subsampling = 0)

#     data.append((rgb_img,0))

# for i, img in enumerate(viral_cases3):

#     im = Image.open(img)

#     rgb_img = im.convert('RGB')

#     rgb_img.save('viral('+str(i)+').jpeg', quality=100, subsampling = 0)

#     data.append((rgb_img,2))

    

    

# for img in normal_cases2:

#     rgb_img = cv2.imread(img)

#     cv2.imwrite(img[:-3] + 'jpeg', rgb_img)

#     data.append((img,0))

# for img in viral_cases2:

#     rgb_img = cv2.imread(img)

#     cv2.imwrite(img[:-3] + 'jpeg', rgb_img)

#     data.append((img,2))
loop_dir4 = ['TEST', 'TRAIN']

for i in loop_dir4:

    viral_dir4 = data_dir4 / i / 'PNEUMONIA (VIRUS)'

    viral_cases4 = viral_dir4.glob('*.jpeg')

    for img in viral_cases4:

        viral_data.append((img,2))
print('number of normal cases: ' + str(len(normal_data)))

print('number of bacterial cases: ' + str(len(bacterial_data)))

print('number of viral cases: ' + str(len(viral_data)))
data = normal_data + bacterial_data + viral_data

del normal_data, bacterial_data, viral_data, data_dir1, data_dir2, data_dir3, data_dir4
data = pd.DataFrame(data, columns=['image', 'label'],index=None)

#shuffle

data = data.sample(frac=1.).reset_index(drop=True)

#print

data.head()        
# Get the counts for each class

cases_count = data['label'].value_counts()

print(cases_count)



# Plot the results 

plt.figure(figsize=(10,8))

sns.barplot(x=cases_count.index, y= cases_count.values)

plt.title('Number of cases', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Bacterial(1)', 'Viral(2)'])

plt.show()

del cases_count
#show sample

viral_samples = (data[data['label']==2]['image'].iloc[:5]).tolist()

bacterial_samples = (data[data['label']==1]['image'].iloc[:5]).tolist()

normal_samples = (data[data['label']==0]['image'].iloc[:5]).tolist()



# Concat the data in a single list and del the above 3 lists

samples = viral_samples + bacterial_samples + normal_samples

del viral_samples, normal_samples, bacterial_samples



# Plot the data 

f, ax = plt.subplots(3,5, figsize=(30,15))

for i in range(15):

    img = imread(samples[i])

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("Viral")

    elif i>=5 and i<10:

        ax[i//5, i%5].set_title("Bacterial")

    else:

        ax[i//5, i%5].set_title("Normal")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
data = data.iloc[: int(data.shape[0]/2)]

print(data.shape)
image_data = []

label_data = []

for index, d in data.iterrows():

    img = d['image']

    l = d['label']

    img = cv2.imread(str(img))

    img = cv2.resize(img, (224,224))

    if img.shape[2] ==1:

        img = np.dstack([img, img, img])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    label = to_categorical(l, num_classes=3)

    image_data.append(img)

    label_data.append(label)

image_data = np.array(image_data)

label_data = np.array(label_data)

del data
image_train, image_validate, image_test = np.split(image_data, [int(.7*len(image_data)), int(.85*len(image_data))])

label_train, label_validate, label_test = np.split(label_data, [int(.7*len(label_data)), int(.85*len(label_data))])

print('number of training images and labels: ' + str(len(image_train)) + ' and ' + str(len(label_train)))

print('number of validation images and labels: ' + str(len(image_validate)) + ' and ' + str(len(label_validate)))

print('number of test images and labels: ' + str(len(image_test)) + ' and ' + str(len(label_test)))

del image_data, label_data
from keras.applications.vgg16 import VGG16

vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

vgg16_model = VGG16(weights=vgg16_weights)

new_vgg16 = Sequential()

for layer in vgg16_model.layers[:-1]:

    new_vgg16.add(layer)

new_vgg16.add(Dense(3, activation='softmax'))

del vgg16_model, vgg16_weights

#new_vgg16.summary()

new_vgg16.compile(loss='categorical_crossentropy',

              optimizer= Adam(lr=0.0001, decay=1e-5),

              metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(

    monitor='val_acc',

    min_delta=0,

    patience=4,

    verbose=0,

    mode='max',

    baseline=None,

    restore_best_weights=True

)
batch_size = 16

with tf.device('/gpu:0'):

    new_vgg16.fit(image_train, label_train, \

                    validation_data=(image_validate, label_validate), \

                    epochs=20, callbacks=[early_stopping_monitor], batch_size=batch_size, verbose = 0)

test_loss, test_score = new_vgg16.evaluate(image_test, label_test, batch_size=16)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
# Get predictions

preds = new_vgg16.predict(image_test, batch_size=16)

preds = np.argmax(preds, axis=-1)



# Original labels

orig_test_labels = np.argmax(label_test, axis=-1)



print(orig_test_labels.shape)

print(preds.shape)
# Get the confusion matrix

cm  = confusion_matrix(orig_test_labels, preds)

plt.figure()

plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(3), ['Normal', 'Bacterial', 'Viral'], fontsize=16)

plt.yticks(range(3), ['Normal', 'Bacterial', 'Viral'], fontsize=16)

plt.show()
nn, nb, nv, bn, bb, bv, vn, vb, vv = cm.ravel()

normal_precision = nn/(nn+bn+vn)

normal_recall = nn/(nn+nb+nv)

bacterial_precision = bb/(nb+bb+vb)

bacterial_recall = nn/(bn+bb+bv)

viral_precision = nn/(nv+bv+vv)

viral_recall = nn/(vn+vb+vv)

print("Recall of the model when dealing with heathly lungs is {:.2f}".format(normal_recall))

print("Precision of the model when dealing with heathly lungs is {:.2f}".format(normal_precision))

print("Recall of the model when dealing with bacterial pneumonia is {:.2f}".format(bacterial_recall))

print("Precision of the model when dealing with bacterial pneumonia is {:.2f}".format(bacterial_precision))

print("Recall of the model when dealing with viral pneumonia is {:.2f}".format(viral_recall))

print("Precision of the model when dealing with viral pneumonia is {:.2f}".format(viral_precision))