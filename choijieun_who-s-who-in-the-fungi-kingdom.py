import numpy as np

import matplotlib.pyplot as plt

import os

import glob

import random

import math

import pandas as pd

import seaborn as sns



from keras import preprocessing, optimizers, utils, layers, models

from keras.models import Sequential

from keras.layers import Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization

from keras.callbacks import ModelCheckpoint, EarlyStopping



from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess_input

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input



%matplotlib inline



from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
image_dir = '../input/mushrooms-classification-common-genuss-images/mushrooms/Mushrooms/'

g_type_list = [x.split('/')[-1] for x in glob.glob(os.path.join(image_dir, '[A-Z]*'))]

print(g_type_list)
num_classes = len(g_type_list)

print('{:<20}'.format('Mushroom Genuses:')+'{:>90}'.format(', '.join(g_type_list)))

print('{:<40}'.format('Number of Mushroom Genuses/Classes:')+'{:>70}'.format('{:2n}'.format(num_classes)))



image_size = 299 #299 for InceptionV3, 224 for ResNet50 and VGG16

batch_size = 32

print('-'*110)

print('{:<40}'.format('Input Image:')+'{:>70}'.format('{:3n} x {:3n} x {:1n}'.format(*[image_size, image_size, 3])))

print('{:<40}'.format('Batch Size:')+'{:>70}'.format('{:2n}'.format(batch_size)))



n_images = len(glob.glob(os.path.join(image_dir, '*/*jpg')))

train_val_split = 0.25

n_val = int(n_images*train_val_split)

n_train = n_images - n_val

print('-'*110)

print('{:<40}'.format('Total Number of Training Images:')+'{:>70}'.format('{:3n}'.format(n_train)))

print('{:<40}'.format('Total Number of Validation Images:')+'{:>70}'.format('{:3n}'.format(n_val)))
inceptionv3_weights = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

base = InceptionV3(weights=inceptionv3_weights, include_top=False, input_shape=(image_size, image_size, 3))



#resnet50_weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

#base = ResNet50(weights=resnet50_weights, include_top=False, input_shape=(image_size, image_size, 3))

#

#vgg16_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

#base = VGG16(weights=vgg16_weights, include_top=False, input_shape=(image_size, image_size, 3))
for layer in base.layers:

    layer.trainable = False



X = base.output

X = GlobalAveragePooling2D()(X)

X = BatchNormalization()(X)



X = Dense(128, activation=None)(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)



X = Dense(128, activation=None)(X)

X = BatchNormalization()(X)

X = Activation('relu')(X)

X = Dropout(0.3)(X)



X = Dense(num_classes, activation=None)(X)

X = BatchNormalization()(X)

Y = Activation('softmax')(X)



model = models.Model(inputs=base.input, outputs=Y)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(), metrics=['accuracy'])
idg = preprocessing.image.ImageDataGenerator(preprocessing_function=inceptionv3_preprocess_input, validation_split=train_val_split)



train_generator = idg.flow_from_directory(

        image_dir,

        target_size=(image_size, image_size),

        batch_size=batch_size,

        class_mode='categorical',

        subset='training',

        shuffle=True)



val_generator = idg.flow_from_directory(

        image_dir,

        target_size=(image_size, image_size),

        batch_size=batch_size,

        class_mode='categorical',

        subset='validation',

        shuffle=True)
print(val_generator.class_indices)
num_per_genus = 3

random_images = []

random_images_labels = []



for g in g_type_list:

    g_img_list = glob.glob(os.path.join(image_dir, g+'/*jpg'))

    num_img = len(g_img_list)

    rand_img_ind = np.random.choice(np.arange(0, num_img), size=num_per_genus, replace=False)

    random_images += [g_img_list[i] for i in rand_img_ind]

    random_images_labels += [g]*num_per_genus



f, ax = plt.subplots(num_per_genus, num_classes, figsize=(20,5))

for i, sample in enumerate(random_images):

    ax[i%num_per_genus, i//num_per_genus].imshow(plt.imread(sample))

    ax[i%num_per_genus, i//num_per_genus].axis('off')

    ax[i%num_per_genus, i//num_per_genus].set_title(random_images_labels[i])



plt.show()    
training_df = pd.DataFrame(train_generator.classes, columns=['classes'])

val_df = pd.DataFrame(val_generator.classes, columns=['classes'])



plt.figure(1, figsize=(7, 7))



for i in range(2):

    if i == 0:

        df = training_df

        ax = plt.subplot(211)



    else:

        df = val_df

        ax = plt.subplot(212)



    s = sns.countplot(x='classes', data=df, ax=ax)

    _ = plt.xticks(s.get_xticks(), g_type_list, rotation=60, ha='right')

    s.set_xlabel(None)

    s.set_ylim(0, 1200)

    if i == 0:

        _ = s.set_title('Training Set')

    else:

        _ = s.set_title('Validation Set')



plt.tight_layout()
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

earlystop = EarlyStopping(monitor = 'val_loss', patience = 5)



history = model.fit_generator(train_generator,\

                                  steps_per_epoch=math.ceil(n_train/batch_size),\

                                  epochs=20,\

                                  validation_data=val_generator,\

                                  validation_steps=math.ceil(n_val/batch_size),\

                                  callbacks=[checkpoint, earlystop])

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

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
model.save('mushroom_inceptionv3.h5')