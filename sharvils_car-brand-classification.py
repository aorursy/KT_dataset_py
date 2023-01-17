from tensorflow.keras.layers import Input,Lambda,Dense,Flatten

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

import numpy as np

import pandas as pd

from glob import glob

import matplotlib.pyplot as plt
IMAGE_SIZE=[224,224]

train_path='/kaggle/input/car-brand-classification-dataset/Datasets/Train'

test_path='/kaggle/input/car-brand-classification-dataset/Datasets/Test'
resnet=ResNet50(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
# resnet.summary()
for layer in resnet.layers:

    layer.trainable=False
folders=glob('/kaggle/input/car-brand-classification-dataset/Datasets/Train/*')
folders
x=Flatten()(resnet.output)
pred=Dense(len(folders),activation='softmax')(x)

model=Model(inputs=resnet.input, outputs=pred)
# model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
train_datagen=ImageDataGenerator(rescale=1./255,

                                shear_range=0.2,

                                zoom_range=0.2,

                                horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory(train_path,

                                            target_size=(224,224),

                                            batch_size=32,

                                            class_mode='categorical')
test_set=test_datagen.flow_from_directory(test_path,

                                          target_size=(224,224),

                                          batch_size=32,

                                         class_mode='categorical')
r=model.fit_generator(train_set,

                     validation_data=test_set,

                     epochs=50,

                     steps_per_epoch=len(train_set),

                     validation_steps=len(test_set))
print(r.history)
plt.plot(r.history['loss'],label='train loss')

plt.plot(r.history['val_loss'],label='val loss')

plt.legend()

plt.show()

plt.plot(r.history['accuracy'],label='train accuracy')

plt.plot(r.history['val_accuracy'],label='val accuracy')

plt.legend()

plt.show()
model.save('model_resnet50.h5')
y_pred=model.predict(test_set)
y_pred
y_pred=np.argmax(y_pred,axis=1)
y_pred
img=image.load_img('../input/car-brand-classification-dataset/Datasets/Test/mercedes/34.jpg',target_size=(224,224))
img
X=image.img_to_array(img)
X.shape
X=X/255
X=np.expand_dims(X,axis=0)

img_data=preprocess_input(X)

img_data.shape
model.predict(img_data)
a=np.argmax(model.predict(img_data),axis=1)
a