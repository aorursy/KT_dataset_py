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
#make sure data is loaded
print(os.listdir("../input"))
import SimpleITK as sitk
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Input, Model
from keras.layers import Conv3D, Concatenate, MaxPooling3D, Reshape
from keras.layers import UpSampling3D, Activation, Permute

from IPython.display import clear_output
!pip install imutils
!pip install tensorflow-gpu==2.2.0rc2
import cv2
import imutils as imutils
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # machine learning
from tqdm import tqdm # make your loops show a smart progress meter 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
seed = 1
imgSize = (224, 224) # size of vgg16 input

#Paths
imagePath = "../input/flatbrain/BrainDataSet/"
trainPath = imagePath + "training_set/"
ValPath = imagePath + "validation_set/"
#process data for model 1
def processData(path):
    add_pixels_value=0
    data = []
    for value in os.listdir(path):
        for image in os.listdir(path + value + "/"):
            file_path = path + value + "/" + image
            
            hemmorhage = 1 if value.lower() == "hemmorhage_data" else 0
            data.append({"path": file_path, 'hemmorhage': hemmorhage})
            
    frame = pd.DataFrame(data=data).sample(frac=1).reset_index(drop=True)
    frame = frame['path'] 

    imgs = []
    for img in frame:
        img = cv2.imread(img)
        grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grayScale = cv2.GaussianBlur(grayScale, (5, 5), 0)

        thresh = cv2.threshold(grayScale, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = max(contours, key=cv2.contourArea)

        leftMax = tuple(c[c[:, :, 0].argmin()][0])
        rightMax = tuple(c[c[:, :, 0].argmax()][0])
        topMax = tuple(c[c[:, :, 1].argmin()][0])
        bottomMax = tuple(c[c[:, :, 1].argmax()][0])

        add_pixels = add_pixels_value
        img = img[topMax[1] - add_pixels:bottomMax[1] + add_pixels, leftMax[0] - add_pixels:rightMax[0] + add_pixels].copy()
        imgs.append(img)
        
    return np.array(imgs)

processData(trainPath)
processData(ValPath)
clear_output()
trainGen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.25],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)
valGen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input
)
x = trainGen.flow_from_directory(
    trainPath,
    color_mode='rgb',
    target_size=imgSize,
    batch_size=32,
    class_mode='binary',
    seed=seed
)
y = valGen.flow_from_directory(
    ValPath,
    color_mode='rgb',
    target_size=imgSize,
    batch_size=16,
    class_mode='binary',
    seed=seed
)
from keras.optimizers import Adam
weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = tf.keras.applications.VGG16(
    weights=weights,
    include_top=False,
    input_shape=imgSize + (3,)
)

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=6
)

history = model.fit_generator(
    x,
    steps_per_epoch=10,
    epochs=25,
    validation_data=y,
    validation_steps=1,
    callbacks=[early_stopping],
    use_multiprocessing=False
)

print("Training Done")
model.save_weights('FlatModel2W.h5')
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#process data for model 2
#use Glob to recursively get .mha file
brains = glob.glob("../input/brats2015/BRATS2015_Training/BRATS2015_Training/HGG"+"/**/*T1c*.mha",recursive=True)#Brains
tumors = glob.glob("../input/brats2015/BRATS2015_Training/BRATS2015_Training/HGG"+"/**/*OT*.mha",recursive=True)#y

#size to resize images to
size=(32,32,32)

brainImgs= []
tumorImgs = []
for file in brains:
    i = io.imread(file, plugin='simpleitk')
    i = (i-i.mean()) / i.std()
    i = trans.resize(i, size, mode='constant')
    brainImgs.append(i)
np.save("x",np.array(brainImgs)[..., np.newaxis].astype('float32'))#
    
for file in tumors:
    i = io.imread(file, plugin='simpleitk')
    i[i == 4] = 1
    i[i != 1] = 0
    i = i.astype('float32')
    i = trans.resize(i, size, mode='constant')
    tumorImgs.append(i)
np.save("y",np.array(tumorImgs)[..., np.newaxis].astype('float32'))
x = np.load('x.npy')
print('x: ', x.shape)
y = np.load('y.npy')
print('y:', y.shape)
inputs = Input(shape=x.shape[1:])#input shape

layer1 = Conv3D(16, 3, activation='elu', padding='same')(inputs)
layer2 = MaxPooling3D()(layer1)

layer3 = Conv3D(16, 3, activation='elu', padding='same')(layer2)
layer4 = MaxPooling3D()(layer3)

layer5 = Conv3D(16, 3, activation='elu', padding='same')(layer4)
layer6 = MaxPooling3D()(layer5)

layer7 = Conv3D(16, 3, activation='elu', padding='same')(layer6)
layer8 = MaxPooling3D()(layer7)

layer9 = Conv3D(16, 3, activation='elu', padding='same')(layer8)

layer10 = UpSampling3D()(layer9)
layer11 = Concatenate(axis=4)([layer7, layer10])

layer12 = Conv3D(16, 3, activation='elu', padding='same')(layer11)

layer13 = UpSampling3D()(layer12)
layer14 = Concatenate(axis=4)([layer5, layer13])

layer15 = Conv3D(16, 3, activation='elu', padding='same')(layer14)

layer16 = UpSampling3D()(layer15)
layer17 = Concatenate(axis=4)([layer3, layer16])

layer18 = Conv3D(16, 3, activation='elu', padding='same')(layer17)

layer19 = UpSampling3D()(layer18)
layer20 = Concatenate(axis=4)([layer1, layer19])

layer21 = Conv3D(16, 3, activation='elu', padding='same')(layer20)

outputs=Conv3D(1, 1, activation='sigmoid')(layer20)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
#import os # to find file without having to run first code block
#print(os.listdir("../input/savedweights/")) #used just to find files

model.load_weights("../input/savedweights/model2W.h5")#not used unless there are weights for this model
#from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.000001), loss='binary_crossentropy',metrics = ['accuracy'])
model.fit(x, y, validation_split=0.2, epochs=5000, batch_size=8)
print("Training Done")
model.save_weights('model2W.h5')
import random as r
pred = model.predict(x[:50])

num = int(x.shape[1]/2)
for n in range(3):
    i = int(r.random() * pred.shape[0])
    plt.figure(figsize=(15,10))

    plt.subplot(131)
    plt.title('Input')
    plt.imshow(x[i, num, :, :, 0])

    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(y[i, num, :, :, 0])

    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(pred[i, num, :, :, 0])

    plt.show()