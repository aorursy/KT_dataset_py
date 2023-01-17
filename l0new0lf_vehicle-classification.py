import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import OrderedDict

import cv2

from PIL import Image

import keras

# For one-hot-encoding

from keras.utils import np_utils

# For creating sequenttial model

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# For saving and loading models

from keras.models import load_model





import random
# Simple EDA
classes = os.listdir('/kaggle/input/vehicle/train/train')

BASE = '/kaggle/input/vehicle/train/train/'



# create dict of list of images per class

dataset = {}

for vehicle in classes:

    dataset[vehicle] = [i for i in os.listdir(BASE + vehicle)]
# verify

print(dataset.keys())

print(len(dataset.keys()))
# convert dict to pandas df

df = pd.DataFrame.from_dict(dataset, orient='index')

df = df.transpose()
# train set

df.head()
df.info()
cols = []

col_imgs = []

for col in df.columns:

    cols.append(col)

    col_imgs.append(df[col].count())



plt.figure(figsize=(10,6))

plt.barh(cols, col_imgs)

plt.show()
print("="*70)

print("Displaying 4 ranndom image per vehicle class")

print("="*70)



# for every class in `cols`

for j in range(17):

    plt.figure(j)

    plt.figure(figsize=(20,20))

    

    # 4 images per every class

    for i in range(4):

        id = "14{}".format(i+1)

        plt.subplot(int(id))

        random_file = random.choice(os.listdir(BASE + cols[j] + "/"))

        img = cv2.imread(BASE + cols[j] + "/" + random_file)

        plt.title(cols[j])

        plt.imshow(img)

plt.show()
data = []

labels = []
'''

cols = sorted(cols)



# Creating trainable 224x224 images

#                    -------

for vehicle_class in cols:

    print(vehicle_class + " started .....")

    for filename in df[vehicle_class]:

        try:

            # for empty cols

            if filename == None:

                pass

            else:

                image = cv2.imread("/kaggle/input/vehicle/train/train/{}/".format(vehicle_class) + filename)

                image_from_numpy_array = Image.fromarray(image, "RGB")

                resized_image = image_from_numpy_array.resize((224, 224))

                data.append(np.array(resized_image))



                if vehicle_class == 'Ambulance':

                    labels.append(0)

                elif vehicle_class == 'Barge':

                    labels.append(1)

                elif vehicle_class == 'Bicycle':

                    labels.append(2)

                elif vehicle_class == 'Boat':

                    labels.append(3)

                elif vehicle_class == 'Bus':

                    labels.append(4)

                elif vehicle_class == 'Car':

                    labels.append(5)

                elif vehicle_class == 'Cart':

                    labels.append(6)

                elif vehicle_class == 'Caterpillar':

                    labels.append(7)

                elif vehicle_class == 'Helicopter':

                    labels.append(8)

                elif vehicle_class == 'Limousine':

                    labels.append(9)

                elif vehicle_class == 'Motorcycle':

                    labels.append(10)

                elif vehicle_class == 'Segway':

                    labels.append(11)

                elif vehicle_class == 'Snowmobile':

                    labels.append(12)

                elif vehicle_class == 'Tank':

                    labels.append(13)

                elif vehicle_class == 'Taxi':

                    labels.append(14)

                elif vehicle_class == 'Truck':

                    labels.append(15)

                elif vehicle_class == 'Van':

                    labels.append(16)

                else:

                    print("Something is wrong.")

                

        except AttributeError:

            print("Attribute error occured for "+filename)

'''
'''

vehicle_images_224x224 = np.array(data)

labels_224x224 = np.array(labels)



# save

np.save("all-vehicle-224x224-images-as-arrays", vehicle_images_224x224)

np.save("corresponding-labels-for-all-224x224-images", labels_224x224)

'''
#data = np.load("all-vehicle-224x224-images-as-arrays.npy")

#labels = np.load("corresponding-labels-for-all-224x224-images.npy")
'''

print(vehicle_images_224x224.shape)

print(labels_224x224.shape)

print(np.unique(labels_224x224))

'''
!rm -r /kaggle/working/data
# Move images to `test` and `train` dir

import shutil

import os



os.mkdir("/kaggle/working/data")

os.mkdir("/kaggle/working/data/test")

os.mkdir("/kaggle/working/data/train")



classes = ['Bicycle', 'Boat', 'Bus', 'Car', 'Motorcycle', 'Truck', 'Van']



for dir in ["test", "train"]:

    for _class in classes:

        os.mkdir("/kaggle/working/data/{}/{}".format(dir, _class))



for _class in classes:

    images = os.listdir("/kaggle/input/vehicle/train/train/{}".format(_class))



    test = images[:300]

    

    # downsample to 1.5k images

    if len(images) < 1500:

      train = images[300:]

    else:

      train = images[300:1500]



    # move images to test-set folder

    for image in test:

        shutil.copy("/kaggle/input/vehicle/train/train/{}/{}".format(_class, image), "/kaggle/working/data/test/{}/{}".format(_class, image))



    # move images to train-set folder

    for image in train:

        shutil.copy("/kaggle/input/vehicle/train/train/{}/{}".format(_class, image), "/kaggle/working/data/train/{}/{}".format(_class, image))
import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

from PIL import Image
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import ResNet50

from keras.applications.resnet50 import preprocess_input

from keras import Model, layers

from keras.models import load_model, model_from_json
input_path = "/kaggle/working/data/"
train_datagen = ImageDataGenerator(

    shear_range=10,

    zoom_range=0.2,

    horizontal_flip=True,

    preprocessing_function=preprocess_input)



train_generator = train_datagen.flow_from_directory(

    input_path + 'train',

    batch_size=32,

    #class_mode='binary',

    target_size=(224,224))



validation_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input)



validation_generator = validation_datagen.flow_from_directory(

    input_path + 'test',

    shuffle=False,

    #class_mode='binary',

    target_size=(224,224))
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False
x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dense(512, activation='relu')(x)

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)
optimizer = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=30,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history
x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x) 

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



optimizer = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=30,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False



x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x) 

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



optimizer = keras.optimizers.Adam(learning_rate=0.000001)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=30,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False



x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x) 

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



optimizer = keras.optimizers.Adam(lr=1e-2, beta_1=1e-2/60)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=60,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False



x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x) 

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=60,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False



x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(128, activation='relu')(x) 

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



# Note sgd 

optimizer = keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/60)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=60,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history          
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False



x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu')(x)

x = layers.Dropout(0.5)(x)

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



optimizer = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=347 // 32,  # added in Kaggle

                              epochs=60,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()
del model

del history          
conv_base = ResNet50(

    include_top=False,

    weights='imagenet')



for layer in conv_base.layers:

    layer.trainable = False



x = conv_base.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(1024, activation='relu')(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation='relu')(x)

x = layers.Dropout(0.5)(x)

predictions = layers.Dense(7, activation='softmax')(x)

model = Model(conv_base.input, predictions)



from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



earlystop = EarlyStopping(monitor = 'val_loss', 

                          min_delta = 0, 

                          patience = 5,

                          verbose = 1,

                          restore_best_weights = True)



reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',

                              factor = 0.2,

                              patience = 3,

                              verbose = 1,

                              min_delta = 0.00001)



checkpoint = ModelCheckpoint("/kaggle/working/ckp/resnet.h5",

                             monitor="val_loss",

                             mode="min",

                             save_best_only = True,

                             verbose=1)





callbacks = [earlystop, checkpoint, reduce_lr]



optimizer = keras.optimizers.Adam(lr = 0.0001)

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])



nb_train_samples = 8011

nb_validation_samples = 2100

epochs = 60

batch_size = 7



history = model.fit_generator(generator=train_generator,

                              steps_per_epoch=nb_train_samples // batch_size,  # added in Kaggle

                              epochs=epochs,

                              callbacks = callbacks,

                              validation_data=validation_generator,

                              validation_steps=10  # added in Kaggle

                             )
# plot loss chart

import matplotlib.pyplot as plt



history_dict = history.history



loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)



line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')

line2 = plt.plot(epochs, loss_values, label='Training Loss')

plt.setp(line1, linewidth=2.0, marker = '+', markersize=10.0)

plt.setp(line2, linewidth=2.0, marker = '4', markersize=10.0)

plt.xlabel('Epochs') 

plt.ylabel('Loss')

plt.grid(True)

plt.legend()

plt.show()