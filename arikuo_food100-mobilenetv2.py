import os

print(os.listdir('../input/uecfood100'))
train_dir = '../input/uecfood100/UECFOOD100'
dirNames = os.listdir(train_dir)

dirNames.remove('category_ja_sjis.txt')



for dirName in dirNames:

    if '.txt' in dirName:

        print(dirName)

        dirNames.remove(dirName)





print(dirNames)

print(len(dirNames))
import pandas as pd

df = pd.read_csv(train_dir+'/category.txt', sep='\t')

print(df)



labels = df['name'].tolist()
print(dirNames)
trainFiles = []

trainClasses = []



for dirName in dirNames:

    for file in os.listdir(train_dir+"/"+dirName):

        trainFiles.append(train_dir+"/"+dirName+"/"+file)

        trainClasses.append(dirName)



print(len(trainFiles), len(trainClasses))
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
plt.imshow(mpimg.imread(trainFiles[0]))
from collections import Counter

import seaborn as sns

sns.set_style("whitegrid")



def plot_equilibre(categories, counts):



    plt.figure(figsize=(12, 8))



    sns_bar = sns.barplot(x=categories, y=counts)

    sns_bar.set_xticklabels(categories, rotation=45)

    plt.title('Equilibre of Training Dataset')

    plt.show()
categories = dirNames

counts = []

[counts.append(trainClasses.count(dirName)) for dirName in dirNames]



plot_equilibre(categories, counts)
import numpy as np

import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.utils import to_categorical



from IPython.display import Image

import matplotlib.pyplot as plt
target_size=(224,224)

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',    

    shuffle=True,

    seed=42,

    class_mode='categorical')

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.models import Model, save_model

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix
num_classes = 100

input_shape = (224,224,3)
# Build Model

net = MobileNetV2(input_shape=input_shape, weights='imagenet', include_top=False)



# add two FC layers (with L2 regularization)

x = net.output

x = GlobalAveragePooling2D()(x) 

#x = BatchNormalization()(x)



#x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

x = Dense(1024)(x)

#x = Dropout(0.2)(x)



#x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

x = Dense(256)(x)

#x = Dropout(0.2)(x)



# Output layer

out = Dense(num_classes, activation="softmax")(x)



model = Model(inputs=net.input, outputs=out)

model.summary()
# Compile Model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
## set Checkpoint : save best only, verbose on

#checkpoint = ModelCheckpoint("food100_mobilenetv2.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

num_epochs = 50
# Train Model

history = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=num_epochs) #, callbacks=[checkpoint])
## Save Model

save_model(model, 'food100_mobilenetv2.h5')
## load best model weights if using callback (save-best-only)

#model.load_weights("food100_mobilenetv2.hdf5")
#score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)

#print(score)
#predY=model.predict_generator(test_generator)

#y_pred = np.argmax(predY,axis=1)

##y_label= [labels[k] for k in y_pred]

#y_actual = test_generator.classes

#cm = confusion_matrix(y_actual, y_pred)

#print(cm)
#print(classification_report(y_actual, y_pred, target_names=labels))
dir_name  = '1'

file_name = '1.jpg'

testfile  = train_dir+'/'+dir_name+'/'+file_name

plt.imshow(mpimg.imread(testfile))
from PIL import Image

def prepare_image(filepath):

    img = Image.open(filepath)  

    out = img.resize((224, 224)) # (width, height), resample

    return out
img = prepare_image(testfile)

testData = np.array(img).reshape(1,224,224,3)

testData = testData / 255.0

predictions = model.predict(testData)

print(predictions[0])
maxindex = int(np.argmax(predictions[0]))

print('Predicted: %s, Probability = %f' %(labels[maxindex], predictions[0][maxindex]) )