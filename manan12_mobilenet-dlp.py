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
import os
import random
import numpy as np
import pandas as pd 
from skimage import io
from skimage import color
from PIL import Image
import PIL.Image
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from dask.array.image import imread
from dask import bag, threaded
from dask.diagnostics import ProgressBar
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from IPython.display import Image


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image 
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras import optimizers

import os
os.getcwd()
driver_details = pd.read_csv('../input/state-farm-distracted-driver-detection/driver_imgs_list.csv',na_values='na')
print(driver_details.head(5))


train_image = []
image_label = []


for i in range(10):
    print('now we are in the folder C',i)
    imgs = os.listdir("../input/state-farm-distracted-driver-detection/imgs/train/c"+str(i))
    for j in range(1300):
    #for j in range(100):
        img_name = "../input/state-farm-distracted-driver-detection/imgs/train/c"+str(i)+"/"+imgs[j]
        img = cv2.imread(img_name)
        #img = color.rgb2gray(img)
        img = img[50:,120:-50]
        img = cv2.resize(img,(224,224))
        label = i
        driver = driver_details[driver_details['img'] == imgs[j]]['subject'].values[0]
        train_image.append([img,label,driver])
        image_label.append(i)



import random
random.shuffle(train_image)


driv_selected = ['p050', 'p015', 'p022', 'p056']


X_train= []
y_train = []
X_test = []
y_test = []
D_train = []
D_test = []
true_test = []

for features,labels,drivers in train_image:
    if drivers in driv_selected:
        X_test.append(features)
        y_test.append(labels)
        D_test.append(drivers)
        true_test.append(labels)
    
    else:
        X_train.append(features)
        y_train.append(labels)
        D_train.append(drivers)
    
print (len(X_train),len(X_test))
print (len(y_train),len(y_test))
X_train = np.array(X_train).reshape(-1,224,224,3)
X_test = np.array(X_test).reshape(-1,224,224,3)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print (X_train.shape)
base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model.summary()
x=base_model.output #Getting the base model
# Adding more layers in base model
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #Dense layers 1 to learn more complex functions 
x = Dropout(0.1)(x) 
x=Dense(1024,activation='relu')(x) #dense layer 2
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x=Dense(512,activation='relu')(x) #dense layer 3
x = Dropout(0.2)(x)
x=Dense(512,activation='relu')(x) #dense layer 4
#Experiment#5
x=Dense(512,activation='relu')(x) #dense layer 5
##################
preds=Dense(10,activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)
#Printing the modified model summary
model.summary()
from keras import optimizers  
from tensorflow.keras.losses import CategoricalCrossentropy


#adam = optimizers.Adam(lr=0.001) #tried 0.0005 - too slow and didn't converge
#Experiment#1
# sgd = optimizers.SGD(lr = 0.005)
#Experiment#2
sgd = optimizers.SGD(lr = 0.005) # try 0.01 - didn't converge and 0.005 , 0.001 best acc of 11%
# model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) # create object
#Experiment #3
model.compile(optimizer=sgd,loss=CategoricalCrossentropy(label_smoothing=5e-5),metrics=['accuracy']) # create object
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping

checkpointer = ModelCheckpoint('mobilenet_sgd_extra_layers.hdf5', verbose=1, save_best_only=True)
#Patience = 10 is not good as it wastes training time
# earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)


datagen = ImageDataGenerator(
    height_shift_range=0.5,
    width_shift_range = 0.5,
    zoom_range = 0.5,
    rotation_range=30
        )
#datagen.fit(X_train)
data_generator = datagen.flow(X_train, y_train, batch_size = 64)

#Fits the model on train set without augmentation
# mobilenet_model = model.fit_generator(X_train,steps_per_epoch = len(X_train) / 64, callbacks=[checkpointer, earlystopper],
#                                                             epochs = 25, verbose = 1, validation_data = (X_test, y_test))

# # Fits the model on batches with real-time data augmentation:
mobilenet_model = model.fit_generator(data_generator,steps_per_epoch = len(X_train) / 64, callbacks=[checkpointer, earlystopper],
                                                            epochs = 20, verbose = 1, validation_data = (X_test, y_test))

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(range(1, len(model.history.history['accuracy']) + 1), model.history.history['accuracy'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Accuracy')
axes[0].plot(range(1, len(model.history.history['val_accuracy']) + 1), model.history.history['val_accuracy'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Validation Accuracy')
axes[0].set_xlabel('Epochs', fontsize = 14)
axes[0].set_ylabel('Accuracy',fontsize = 14)
axes[0].set_title('CNN Dropout Accuracy Trainig VS Validation', fontsize = 14)
axes[0].legend(loc = 'best')
axes[1].plot(range(1, len(model.history.history['loss']) + 1), model.history.history['loss'], linestyle = 'solid', marker = 'o', color = 'crimson', label = 'Training Loss')
axes[1].plot(range(1, len(model.history.history['val_loss']) + 1), model.history.history['val_loss'], linestyle = 'solid', marker = 'o', color = 'dodgerblue', label = 'Validation Loss')
axes[1].set_xlabel('Epochs', fontsize = 14)
axes[1].set_ylabel('Loss',fontsize = 14)
axes[1].set_title('CNN Dropout Loss Trainig VS Validation', fontsize = 14)
axes[1].legend(loc = 'best')
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='Modified_Model.png', show_shapes=True, show_layer_names=True)
plot_model(base_model, to_file='Base_Model.png', show_shapes=True, show_layer_names=True)
def myprint(s):
    with open('Modelsummary.txt','w+') as f:
        print(s, file=f)

# model.summary(print_fn=myprint)
base_model.summary(print_fn=myprint)


# labels is the image array
test_image = []
i = 0
fig, ax = plt.subplots(1, 20, figsize = (50,50 ))

files = os.listdir('../input/state-farm-distracted-driver-detection/imgs/test/')
nums = np.random.randint(low=1, high=len(files), size=20)
for i in range(20):
    print ('Image number:',i)
    img = cv2.imread('../input/state-farm-distracted-driver-detection/imgs/test/'+files[nums[i]])
    img = img[50:,120:-50]
    img = cv2.resize(img,(224,224))
    test_image.append(img)
    ax[i].imshow(img,cmap = 'gray')
    plt.show


test = []

from keras.models import load_model

for img in test_image:
    test.append(img)

    
# model.load_weights('mobilenet_sgd_extra_layers.hdf5')

model=load_model('mobilenet_sgd_extra_layers.hdf5')

test = np.array(test).reshape(-1,224,224,3)
prediction = model.predict(test)
prediction[0:1]
tags = { "C0": "safe driving",
"C1": "texting - right",
"C2": "talking on the phone - right",
"C3": "texting - left",
"C4": "talking on the phone - left",
"C5": "operating the radio",
"C6": "drinking",
"C7": "reaching behind",
"C8": "hair and makeup",
"C9": "talking to passenger" }
# labels is the image array
i = 0
fig, ax = plt.subplots(20, 1, figsize = (100,100))

for i in range(20):
    ax[i].imshow(test[i].squeeze())
    predicted_class = 'C'+str(np.where(prediction[i] == np.amax(prediction[i]))[0][0])
    ax[i].set_title(tags[predicted_class])
    plt.show
    
from PIL import Image
from keras.models import load_model
model=load_model('mobilenet_sgd_extra_layers.hdf5')
def get_data(image_path):
    img = Image.open(image_path)
    img = img.resize((240, 240), Image.ANTIALIAS) # resizes image in-place
    return np.asarray(img)/255

test_file = pd.read_csv('../input/state-farm-distracted-driver-detection/sample_submission.csv')
# test_file.head(5)

for i, file in enumerate(test_file['img']):
    image = get_data('../input/state-farm-distracted-driver-detection/imgs/test/' + file)
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
    result = model.predict(image)
    test_file.iloc[i, 1:] = result[0]
    
test_file.to_csv('results.csv', index = False)