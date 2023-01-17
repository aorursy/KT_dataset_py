from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D ,Dropout, MaxPooling2D

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Activation , Conv2D

from keras.applications.inception_v3 import preprocess_input

from keras import optimizers
base_model = InceptionV3(weights='imagenet', include_top=False)

base_model.summary()
print(base_model.trainable_weights)
print(len(base_model.trainable_weights))
import os

print(len(next(os.walk('../input/images/Images/'))[1]))
# basedir = '../input/images/Images/'

# def main(): 

#     i = 0

      

#     for filename in os.listdir("../input/images/Images/"): 

#         name = filename.split("-")

#         print(name[0])

#         print(name[1])

#         print(filename)



#         os.rename(os.path.join(basedir,filename), os.path.join(basedir,str(name[1]))) 

  

# if __name__ == '__main__': 

      

#     main() 
CLASSES = 120

x = base_model.output

print(x)

print(x.shape)

x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dense(512,activation='relu')(x) #dense layer 3

preds=Dense(CLASSES,activation='softmax')(x) #final layer with softmax activation



# x = Dropout(0.4)(x)

# x= Dense(64)(x)

# x = Activation('relu')(x)



model = Model(inputs=base_model.input, outputs=preds)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

for layer in base_model.layers:

    layer.trainable = True
model.compile(optimizer=sgd,

              loss='categorical_crossentropy',

              metrics=['accuracy'])
train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')
data_dir = '../input/images/Images/'

train_dir = data_dir

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(299,299), batch_size=20)

from keras.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=8)



import h5py

EPOCHS = 25

BATCH_SIZE = 20

STEPS_PER_EPOCH = 320

VALIDATION_STEPS = 64

MODEL_FILE = 'Dogs.h5'

history = model.fit_generator(

    train_generator,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks=[early_stopping_callback])

  

model.save(MODEL_FILE)
model.summary()
import matplotlib.pyplot as plt



print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

import numpy as np

from keras.preprocessing import image

from keras.models import load_model

def predict(model, img):

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    print(preds)

    return preds[0]

img = image.load_img('../input/images/Images/n02104029-kuvasz/n02104029_110.jpg', target_size=(299, 299))

preds = predict(load_model(MODEL_FILE), img)

print(len(preds))

print(preds.argmax())

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())
labels
print(labels[preds.argmax()])
!pip install keras-vis
from vis.utils import utils

from matplotlib import pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (18, 6)

img1 = utils.load_img('../input/images/Images/n02085620-Chihuahua/n02085620_10074.jpg', target_size=(224, 224))



img2 = utils.load_img('../input/images/Images/n02085936-Maltese_dog/n02085936_10073.jpg', target_size=(224, 224))



f, ax = plt.subplots(1, 2)

ax[0].imshow(img1)

ax[1].imshow(img2)
model.layers[1]

weight_conv2d_1 = model.layers[1].get_weights()[0][:,:,0,:]



col_size = 6

row_size = 5

filter_index = 0

fig, ax = plt.subplots(row_size, col_size, figsize=(12,8))

for row in range(0,row_size): 

  for col in range(0,col_size):

    ax[row][col].imshow(weight_conv2d_1[:,:,filter_index])

    filter_index += 1