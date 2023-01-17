from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.optimizers import RMSprop

from sklearn.utils import shuffle

from tensorflow.keras import optimizers

from sklearn.neighbors import KNeighborsClassifier



from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.nasnet import NASNetLarge

from tensorflow.keras import models



import numpy as np



from datetime import datetime



BATCH = 128

image_size = 150

def generator():

    datagen = ImageDataGenerator(rescale=1./255)

    

    train_data = datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_train/seg_train/',

                                        target_size=(image_size, image_size),

                                        batch_size=BATCH,

                                        class_mode='categorical',

                                        shuffle=True)



    test_data = datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_test/seg_test/',

                                        target_size=(image_size, image_size),

                                        batch_size=BATCH,

                                        class_mode='categorical',

                                        shuffle=True)



    return train_data, test_data

train_data, test_data = generator()

train_images, train_labels = train_data.next()

test_images, test_labels = test_data.next()
print(len(test_data))

print(test_data[23][0].shape)



print(len(train_data))

print(train_data[109][0].shape)
train_data[0][0].shape
train_data.class_indices
import subprocess

import pprint



sp = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)



out_str = sp.communicate()

out_list = str(out_str[0]).split('\\n')



out_dict = {}



for item in out_list:

    print(item)
ACTIVATION='relu'

KERNEL_INIT='he_normal'

vgg_model = Sequential()

vgg_model.add(VGG16(include_top=False, weights=None, input_shape=(image_size, image_size, 3), classes=6))

vgg_model.add(layers.Flatten())

vgg_model.add(layers.Dense(1000, activation=ACTIVATION, input_dim=4*4*512, kernel_initializer=KERNEL_INIT))

vgg_model.add(layers.Dense(6, activation='softmax'))



vgg_model.summary()
EPOCHS = 20

vgg_model.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

time_start = datetime.now()

history = vgg_model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

time = datetime.now() - time_start

print('Time: ', time)
vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3), classes=1000)

vgg_model.summary()
def extract_features(sample_count, vgg_model, generat):

    features = np.zeros(shape=(sample_count, 4, 4, 512))

    labels = np.zeros(shape=(sample_count, 6))

    i=0

    for inputs_batch, labels_batch in generat:

        #выделяем признаки из изображений

        features_batch = vgg_model.predict(inputs_batch)

        features[i * BATCH : (i + 1) * BATCH] = features_batch

        labels[i * BATCH : (i + 1) * BATCH] = labels_batch

        i = i + 1

        if (i + 1) * BATCH >= sample_count:

            return features, labels

    return features, labels
time_start = datetime.now()

train_sample_count = 14034

test_sample_count = 3000

generat = generator()

#выделяем признаки

train_features, train_y = extract_features(train_sample_count, vgg_model, generat[0])

test_features, test_y = extract_features(test_sample_count, vgg_model, generat[1])



time = datetime.now() - time_start

print('Time: ', time)
print(train_features.shape)

print(train_y.shape)



print(test_features.shape) 

print(test_y.shape)
train_features = np.reshape(train_features, (train_sample_count, 4*4*512))

test_features = np.reshape(test_features, (test_sample_count, 4*4*512))
print(train_features.shape)

print(train_y.shape)
#передадим полученные признаки на вход полносвязному классификатору

EPOCHS = 20

ACTIVATION='relu'

KERNEL_INIT='he_normal'



model = Sequential()

model.add(layers.Dense(1000, activation=ACTIVATION, input_dim=4*4*512, kernel_initializer=KERNEL_INIT))



model.add(layers.Dense(6, activation='softmax'))

model.summary()



model.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

time_start = datetime.now()

history = model.fit(train_features, train_y, epochs=EPOCHS, batch_size = 128, shuffle=True, validation_data=(test_features, test_y))

time = datetime.now() - time_start

print('Time: ', time)
vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3), classes=1000)

vgg_model.summary()
EPOCHS = 20

ACTIVATION='relu'

KERNEL_INIT='he_normal'



model = Sequential()

model.add(vgg_model)

vgg_model.trainable = False

model.add(layers.Flatten())

model.add(layers.Dense(1000, activation=ACTIVATION, input_dim=4*4*512, kernel_initializer=KERNEL_INIT))



model.add(layers.Dense(6, activation='softmax'))

model.summary()



model.compile(optimizer='rmsprop', 

              loss='categorical_crossentropy',

              metrics=['accuracy'])

time_start = datetime.now()

history =  model.fit_generator(train_data, steps_per_epoch=len(train_data), shuffle=True, epochs=EPOCHS, validation_steps=len(test_data), validation_data=test_data)

time = datetime.now() - time_start

print('Time: ', time)