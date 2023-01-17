# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
import os



url = "../input/dogs-vs-cats-redux-kernels-edition/"

print(os.listdir(url +""))
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
FAST_RUN = False

width = 128

height = 128

size = (width,height)

img_channels = 3
filenames = os.listdir(url + "train/train/")

categories = []

for filename in filenames:

    cate = filename.split('.')[0]

    categories.append(int(cate=='dog'))

df = pd.DataFrame({

    'filename':filenames,

    'category':categories

})
df.head()
df.tail()
df['category'].value_counts().plot.bar()
import keras

from keras.layers import Dense,GlobalAveragePooling2D, BatchNormalization,Dropout

from keras.applications import inception_resnet_v2,inception_v3

from keras.preprocessing import image

from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input

from keras.models import Model

from keras.optimizers import Adam
sample = random.choice(filenames)

image=load_img(url + "train/train/"+sample)

plt.imshow(image)
mode_1 = MobileNet(weights = 'imagenet', include_top = False, input_shape = (width, height, 3))

mode_2 = inception_resnet_v2.InceptionResNetV2(weights = 'imagenet',include_top = False,input_shape = (width, height, 3))

mode_3 = inception_v3.InceptionV3(weights = 'imagenet',include_top=False , input_shape = (width, height, 3))



x_1 = mode_1.output

x_2 = mode_2.output

x_3 = mode_3.output



x_1 = GlobalAveragePooling2D()(x_1)

x_2 = GlobalAveragePooling2D()(x_2)

x_3 = GlobalAveragePooling2D()(x_3)
x_1 = Dense(256,activation='relu')(x_1)

x_2 = Dense(256,activation='relu')(x_2)

x_3 = Dense(256,activation='relu')(x_3)



x_1 = BatchNormalization()(x_1)

x_2 = BatchNormalization()(x_2)

x_3 = BatchNormalization()(x_3)



x_1 = Dropout(0.20)(x_1)

x_2 = Dropout(0.20)(x_2)

x_3 = Dropout(0.20)(x_3)



preds_1 = Dense(2,activation='softmax')(x_1)

preds_2 = Dense(2,activation='softmax')(x_2)

preds_3 = Dense(2,activation='softmax')(x_3)
model_1 = Model(inputs = mode_1.input,outputs = preds_1)

model_2 = Model(inputs = mode_2.input,outputs = preds_2)

model_3 = Model(inputs = mode_3.input,outputs = preds_3)
for layer in model_1.layers:

  layer.trainable = False

for layer in model_1.layers[87:]:

  layer.trainable = True
for layer in model_2.layers:

  layer.trainable = False

for layer in model_2.layers[780:]:

  layer.trainable = True
for layer in model_3.layers:

  layer.trainable = False

for layer in model_3.layers[311:]:

  layer.trainable = True
model_1.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model_2.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model_3.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
earlystop=EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',

                                            patience=2,

                                            verbose=1,

                                            factor=0.5,

                                            min_lr=0.00001)
callbacks= [earlystop,learning_rate_reduction]
df["category"] = df["category"].replace({0:'cat',1:'dog'})
train_df,validate_df = train_test_split(df,test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True) #drops dog.blahblah from dataset

validate_df=validate_df.reset_index(drop=True)
train_df['category'].value_counts().plot.bar()
validate_df['category'].value_counts().plot.bar()
total_train = train_df.shape[0]

total_validate = validate_df.shape[0]

batch_size = 15
train_datagen=ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)

train_generator=train_datagen.flow_from_dataframe(

    train_df,

    url+"train/train/",

    x_col='filename',

    y_col="category",

    target_size=size,

    class_mode='categorical',

    batch_size=batch_size

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df,

    url+"train/train/",

    x_col='filename', 

    y_col='category',

    target_size=size,

    class_mode='categorical',

    batch_size=batch_size

)
example_df = train_df.sample(n=1).reset_index(drop=True)

example_generator = train_datagen.flow_from_dataframe(

    example_df,

    url+"train/train/",

    x_col='filename',

    y_col='category',

    target_size=size,

    class_mode='categorical'

)
plt.figure(figsize=(12,12))

for i in range(0,15):

    plt.subplot(5,3,i+1)

    for X_batch,Y_batch in example_generator:

        image=X_batch[0]

        plt.imshow(image)

        break

plt.tight_layout()

plt.show()
print(device_lib.list_local_devices())
FAST_RUN = False

epochs=3 if FAST_RUN else 12

history_1 = model_1.fit_generator(

    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model_1.save_weights("w1.h5")

model_1.save('m1.h5')
history_2 = model_2.fit_generator(

    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model_2.save_weights("w2.h5")

model_2.save('m2.h5')
history_3 = model_3.fit_generator(





    train_generator,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=total_validate//batch_size,

    steps_per_epoch=total_train//batch_size,

    callbacks=callbacks

)
model_3.save_weights("w3.h5")

model_3.save('m3.h5')
history = history_1

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12))

ax1.plot(history.history['loss'],color='b',label='Training Loss')

ax1.plot(history.history['val_loss'],color='r',label='Validation Loss')

ax1.set_xticks(np.arange(1,epochs,1))

ax1.set_yticks(np.arange(0,1,0.1))



ax2.plot(history.history['accuracy'],color='b',label='Training Accuracy')

ax2.plot(history.history['val_accuracy'],color='r',label='Validation Accuracy')

ax2.set_xticks(np.arange(1,epochs,1))



legend = plt.legend(loc='best',shadow=True)

plt.tight_layout()

plt.savefig('p1.png')

history = history_2

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12))

ax1.plot(history.history['loss'],color='b',label='Training Loss')

ax1.plot(history.history['val_loss'],color='r',label='Validation Loss')

ax1.set_xticks(np.arange(1,epochs,1))

ax1.set_yticks(np.arange(0,1,0.1))



ax2.plot(history.history['accuracy'],color='b',label='Training Accuracy')

ax2.plot(history.history['val_accuracy'],color='r',label='Validation Accuracy')

ax2.set_xticks(np.arange(1,epochs,1))



legend = plt.legend(loc='best',shadow=True)

plt.tight_layout()

plt.savefig('p2.png')
history = history_3

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12))

ax1.plot(history.history['loss'],color='b',label='Training Loss')

ax1.plot(history.history['val_loss'],color='r',label='Validation Loss')

ax1.set_xticks(np.arange(1,epochs,1))

ax1.set_yticks(np.arange(0,1,0.1))



ax2.plot(history.history['accuracy'],color='b',label='Training Accuracy')

ax2.plot(history.history['val_accuracy'],color='r',label='Validation Accuracy')

ax2.set_xticks(np.arange(1,epochs,1))



legend = plt.legend(loc='best',shadow=True)

plt.tight_layout()

plt.savefig('p3.png')
test_filenames = os.listdir(url+"test/test/")

test_df = pd.DataFrame({

    'filename' : test_filenames

})

nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df,

    url+"test/test/",

    x_col = 'filename',

    y_col = None,

    class_mode=None,

    target_size=size,

    batch_size=batch_size,

    shuffle=False

)
predict_1 = model_1.predict_generator(test_generator,steps = np.ceil(nb_samples/batch_size))

predict_2 = model_2.predict_generator(test_generator,steps = np.ceil(nb_samples/batch_size))

predict_3 = model_3.predict_generator(test_generator,steps = np.ceil(nb_samples/batch_size))

test_df['category'] = np.argmax(predict_1,axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({'dog' : 1, 'cat' : 0})
test_df['category'].value_counts().plot.bar()
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename','category'],axis = 1, inplace=True)

submission_df.to_csv('submitthis1.csv',index=False)
test_df['category'] = np.argmax(predict_2,axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'dog' : 1, 'cat' : 0})



submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename','category'],axis = 1, inplace=True)

submission_df.to_csv('submitthis2.csv',index=False)
test_df['category'] = np.argmax(predict_3,axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)

test_df['category'] = test_df['category'].replace({'dog' : 1, 'cat' : 0})



submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename','category'],axis = 1, inplace=True)

submission_df.to_csv('submitthis3.csv',index=False)