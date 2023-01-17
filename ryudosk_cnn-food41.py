import h5py

import os

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.applications.vgg16 import VGG16

import keras

from keras.utils.io_utils import HDF5Matrix

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.models import Model

from keras.layers import Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, Dropout, Activation

base_path = os.path.join('..', '/kaggle/working/')

train_h5_path = os.path.join(base_path, 'food5_train.hdf5')

test_h5_path = os.path.join(base_path, 'food5_test.hdf5')

%matplotlib inline

#!pip install --upgrade keras

!pip install gdown

import gdown; 



os.remove("food5_train.hdf5")

os.remove("food5_test.hdf5")



url = 'https://drive.google.com/uc?id=1--CQs5t4NhajbFnfJsXGi1h2CEEn_UbK' 

output = 'food5_train.hdf5' 

gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1-5YJJTz0YPzUi25vOEmt0736eKNHQfIi' 

output = 'food5_test.hdf5' 

gdown.download(url, output, quiet=False)

X_train = HDF5Matrix(train_h5_path,'images')

y_train = HDF5Matrix(train_h5_path, dataset='category')



print('In Data',X_train.shape,'=>', y_train.shape)
X_test = HDF5Matrix(test_h5_path, 'images')

y_test = HDF5Matrix(test_h5_path, 'category')

print('In Data',X_test.shape,'=>', y_test.shape)
sample_imgs = 25

with h5py.File(train_h5_path, 'r') as n_file:

    total_imgs = n_file[[0].shape[0]]

    print(n_file.items())

    read_idxs = slice(0,sample_imgs)

    im_data = n_file['images'][read_idxs]

    im_label = n_file['category'].value[read_idxs]

    label_names = [x.decode() for x in n_file['category_names'].value]

fig, m_ax = plt.subplots(5, 5, figsize = (12, 12))

for c_ax, c_label, c_img in zip(m_ax.flatten(), im_label, im_data):

    c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

    c_ax.axis('off')

    c_ax.set_title(label_names[np.argmax(c_label)])
#https://keras.io/preprocessing/image/

#https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad

#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb

BATCH_SIZE = 16

IMAGE_HT_WID=300

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(

                               rotation_range=15,

#                                width_shift_range=0.1,

#                                height_shift_range=0.1,

#                                shear_range=0.01,

#                                zoom_range=[0.9, 1.25],

#                                horizontal_flip=True,

#                                vertical_flip=False,

#                                #data_format='channels_last',

                              fill_mode='reflect',

                              channel_shift_range = 30,

#                               brightness_range=[0.5, 1.5],

                               validation_split=0.4,

                              # rescale=1./255

                              samplewise_center = True,

                              samplewise_std_normalization = True,

                              preprocessing_function = tf.keras.applications.inception_resnet_v2.preprocess_input

                               )







train_generator=train_datagen.flow_from_directory(

                   

                    directory="../input/images/",

                

                   subset="training",

                    batch_size=BATCH_SIZE,

                    seed=55551,

                    shuffle=True,

                    

                    class_mode="categorical",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator=train_datagen.flow_from_directory(

                    

                    directory="../input/images/",

                   subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=55551,

                    shuffle=True,

                  

                    class_mode="categorical",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
# Creating model

model = Sequential()



model.add(Conv2D(64, (8, 8), padding='same',input_shape=(300, 300, 3)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (6, 6)))

model.add(Activation('relu'))

model.add(Dropout(0.25))





model.add(Conv2D(86, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(86, (3, 3)))

model.add(Activation('relu'))

model.add(AvgPool2D(pool_size=(3,3)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (2, 2)))

model.add(Activation('relu'))

model.add(Conv2D(128, (2, 2)))

model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(4021))

model.add(Activation('relu'))

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(0.5))

# total class here

model.add(Dense(6))

model.add(Activation('softmax'))

# initiate RMSprop optimizer

opt = keras.optimizers.Adam(lr=0.00008, beta_1=0.9, beta_2=0.97, epsilon=1e-7)



# Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



loss_history = []

model.summary()
baseModel = VGG16(weights="imagenet", include_top=False,input_shape=(300,300,3),classes=1000)

#model = VGG16(weights=None,input_shape=(64,64,3),classes=101)

# initiate RMSprop optimizer

opt = keras.optimizers.Adam(lr=0.00008, beta_1=0.9, beta_2=0.97, epsilon=1e-7)



# Let's train the model using RMSprop

#model.compile(loss='categorical_crossentropy',

#              optimizer=opt,

#              metrics=['accuracy'])



loss_history = []



# construct the head of the model that will be placed on top of the

# the base model

headModel = baseModel.output

headModel = Flatten(name="flatten")(headModel)

headModel = Dense(1024, activation="relu")(headModel)

headModel = Dropout(0.5)(headModel)

headModel = Dense(6, activation="softmax")(headModel)



# place the head FC model on top of the base model (this will become

# the actual model we will train)

model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will

# *not* be updated during the first training process

for layer in baseModel.layers:

	layer.trainable = False



opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics="binary_accuracy")

#model.summary()
for layer in baseModel.layers[15:]:

	layer.trainable = True



# loop over the layers in the model and show which ones are trainable

# or not

for layer in baseModel.layers:

	print("{}: {}".format(layer, layer.trainable))



opt = keras.optimizers.SGD(lr=1e-4, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=opt,

	metrics=["accuracy"])
def schedule(epoch):

    if epoch < 5:

        return .01

    elif epoch < 7:

        return .002

    else:

        return .0004

lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)

checkpointer = keras.callbacks.ModelCheckpoint(filepath='modelFood_dataset.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)

csv_logger = keras.callbacks.CSVLogger('modelFood_dataset.log')
EPOCHS=8

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1 

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    callbacks=[lr_scheduler,checkpointer,csv_logger],              

                    verbose=1

)
for i in range(7):

    print("")

    print("Training epoch:", i+1)

    loss_history += [model.fit(X_train, y_train,

                               validation_data=(X_test, y_test), 

                               batch_size = 32,

                               epochs = 1, shuffle="batch")]
epich = np.cumsum(np.concatenate(

    [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

_ = ax1.plot(epich,

             np.concatenate([mh.history['loss'] for mh in loss_history]),

             'b-',

             epich, np.concatenate(

        [mh.history['val_loss'] for mh in loss_history]), 'r-')

ax1.legend(['Training', 'Validation'])

ax1.set_title('Loss')



_ = ax2.plot(epich, np.concatenate(

    [mh.history['acc'] for mh in loss_history]), 'b-',

                 epich, np.concatenate(

        [mh.history['val_acc'] for mh in loss_history]),

                 'r-')

ax2.legend(['Training', 'Validation'])

ax2.set_title('Accuracy')
sample_imgs = 16

with h5py.File(test_h5_path, 'r') as n_file:

    total_imgs = n_file['images'].shape[0]

    read_idxs = slice(0,sample_imgs)

    im_data = n_file['images'][read_idxs]

    im_label = n_file['category'].value[read_idxs]

    label_names = ["RICE","EELS ON RICE","PILAF","CHICKEN-'N'-EGG ON RICE","PORK CUTLET ON RICE","UNCATEGORIZED"]

pred_label = model.predict(im_data)

fig, m_ax = plt.subplots(4, 4, figsize = (20, 20))

for c_ax, c_label, c_pred, c_img in zip(m_ax.flatten(), im_label, pred_label, im_data):

    c_ax.imshow(c_img if c_img.shape[2]==3 else c_img[:,:,0], cmap = 'gray')

    c_ax.axis('off')

    c_ax.set_title('Predicted:{}\nActual:{}'.format(label_names[np.argmax(c_pred)],

                                                  label_names[np.argmax(c_label)]))
model.save('my_model.h5')