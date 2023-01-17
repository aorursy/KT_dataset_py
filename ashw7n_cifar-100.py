from __future__ import print_function

import keras

from keras.layers import Dense, Conv2D, BatchNormalization, Activation

from keras.layers import AveragePooling2D, Input, Flatten

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from keras import backend as K

from keras.models import Model

from keras.datasets import cifar100

import numpy as np

import os
LABEL = [

    'APPLE',

    'AQUARIUM_FISH',

    'BABY',

    'BEAR',

    'BEAVER',

    'BED',

    'BEE',

    'BEETLE',

    'BICYCLE',

    'BOTTLE',

    'BOWL',

    'BOY',

    'BRIDGE',

    'BUS',

    'BUTTERFLY',

    'CAMEL',

    'CAN',

    'CASTLE',

    'CATERPILLAR',

    'CATTLE',

    'CHAIR',

    'CHIMPANZEE',

    'CLOCK',

    'CLOUD',

    'COCKROACH',

    'COUCH',

    'CRAB',

    'CROCODILE',

    'CUP',

    'DINOSAUR',

    'DOLPHIN',

    'ELEPHANT',

    'FLATFISH',

    'FOREST',

    'FOX',

    'GIRL',

    'HAMSTER',

    'HOUSE',

    'KANGAROO',

    'COMPUTER_KEYBOARD',

    'LAMP',

    'LAWN_MOWER',

    'LEOPARD',

    'LION',

    'LIZARD',

    'LOBSTER',

    'MAN',

    'MAPLE_TREE',

    'MOTORCYCLE',

    'MOUNTAIN',

    'MOUSE',

    'MUSHROOM',

    'OAK_TREE',

    'ORANGE',

    'ORCHID',

    'OTTER',

    'PALM_TREE',

    'PEAR',

    'PICKUP_TRUCK',

    'PINE_TREE',

    'PLAIN',

    'PLATE',

    'POPPY',

    'PORCUPINE',

    'POSSUM',

    'RABBIT',

    'RACCOON',

    'RAY',

    'ROAD',

    'ROCKET',

    'ROSE',

    'SEA',

    'SEAL',

    'SHARK',

    'SHREW',

    'SKUNK',

    'SKYSCRAPER',

    'SNAIL',

    'SNAKE',

    'SPIDER',

    'SQUIRREL',

    'STREETCAR',

    'SUNFLOWER',

    'SWEET_PEPPER',

    'TABLE',

    'TANK',

    'TELEPHONE',

    'TELEVISION',

    'TIGER',

    'TRACTOR',

    'TRAIN',

    'TROUT',

    'TULIP',

    'TURTLE',

    'WARDROBE',

    'WHALE',

    'WILLOW_TREE',

    'WOLF',

    'WOMAN',

    'WORM',

]



MAPPING = {

    'AQUATIC MAMMALS': ['BEAVER', 'DOLPHIN', 'OTTER', 'SEAL', 'WHALE'],

    'FISH': ['AQUARIUM_FISH', 'FLATFISH', 'RAY', 'SHARK', 'TROUT'],

    'FLOWERS': ['ORCHID', 'POPPY', 'ROSE', 'SUNFLOWER', 'TULIP'],

    'FOOD CONTAINERS': ['BOTTLE', 'BOWL', 'CAN', 'CUP', 'PLATE'],

    'FRUIT AND VEGETABLES': ['APPLE', 'MUSHROOM', 'ORANGE', 'PEAR', 'SWEET_PEPPER'],

    'HOUSEHOLD ELECTRICAL DEVICE': ['CLOCK', 'COMPUTER_KEYBOARD', 'LAMP', 'TELEPHONE', 'TELEVISION'],

    'HOUSEHOLD FURNITURE': ['BED', 'CHAIR', 'COUCH', 'TABLE', 'WARDROBE'],

    'INSECTS': ['BEE', 'BEETLE', 'BUTTERFLY', 'CATERPILLAR', 'COCKROACH'],

    'LARGE CARNIVORES': ['BEAR', 'LEOPARD', 'LION', 'TIGER', 'WOLF'],

    'LARGE MAN-MADE OUTDOOR THINGS': ['BRIDGE', 'CASTLE', 'HOUSE', 'ROAD', 'SKYSCRAPER'],

    'LARGE NATURAL OUTDOOR SCENES': ['CLOUD', 'FOREST', 'MOUNTAIN', 'PLAIN', 'SEA'],

    'LARGE OMNIVORES AND HERBIVORES': ['CAMEL', 'CATTLE', 'CHIMPANZEE', 'ELEPHANT', 'KANGAROO'],

    'MEDIUM-SIZED MAMMALS': ['FOX', 'PORCUPINE', 'POSSUM', 'RACCOON', 'SKUNK'],

    'NON-INSECT INVERTEBRATES': ['CRAB', 'LOBSTER', 'SNAIL', 'SPIDER', 'WORM'],

    'PEOPLE': ['BABY', 'BOY', 'GIRL', 'MAN', 'WOMAN'],

    'REPTILES': ['CROCODILE', 'DINOSAUR', 'LIZARD', 'SNAKE', 'TURTLE'],

    'SMALL MAMMALS': ['HAMSTER', 'MOUSE', 'RABBIT', 'SHREW', 'SQUIRREL'],

    'TREES': ['MAPLE_TREE', 'OAK_TREE', 'PALM_TREE', 'PINE_TREE', 'WILLOW_TREE'],

    'VEHICLES 1': ['BICYCLE', 'BUS', 'MOTORCYCLE', 'PICKUP_TRUCK', 'TRAIN'],

    'VEHICLES 2': ['LAWN_MOWER', 'ROCKET', 'STREETCAR', 'TANK', 'TRACTOR']

}
batch_size = 32

epochs = 10

num_classes = 100

subtract_pixel_mean = True

depth = 20
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

input_shape = x_train.shape[1:]

x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255



if subtract_pixel_mean:

    x_train_mean = np.mean(x_train, axis=0)

    x_train -= x_train_mean

    x_test -= x_train_mean



print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')

print('y_train shape:', y_train.shape)



y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):

    lr = 1e-3

    if epoch > 180:

        lr *= 0.5e-3

    elif epoch > 160:

        lr *= 1e-3

    elif epoch > 120:

        lr *= 1e-2

    elif epoch > 80:

        lr *= 1e-1

    print('Learning rate: ', lr)

    return lr
def resnet_layer(inputs,

                 num_filters=16,

                 kernel_size=3,

                 strides=1,

                 activation='relu',

                 batch_normalization=True,

                 conv_first=True):

    conv = Conv2D(num_filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-4))



    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

    return x
def resnet(input_shape, depth, num_classes=100):

    if (depth - 2) % 6 != 0:

        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    num_filters = 16

    num_res_blocks = int((depth - 2) / 6)



    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:

                strides = 2

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None)

            if stack > 0 and res_block == 0:

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])

            x = Activation('relu')(x)

        num_filters *= 2

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)

    return model



model = resnet(input_shape=input_shape, depth=depth)



model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=lr_schedule(0)),

              metrics=['accuracy'])

model.summary()
lr_scheduler = LearningRateScheduler(lr_schedule)



lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),

                               cooldown=0,

                               patience=5,

                               min_lr=0.5e-6)



callbacks = [lr_reducer, lr_scheduler]



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_data=(x_test, y_test),

          shuffle=True,

          callbacks=callbacks)
scores = model.evaluate(x_test, y_test, verbose=2)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
predictions = model.predict([x_test])
from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    

    #classes = classes[unique_labels(y_true, y_pred).astype(int)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    fig, ax = plt.subplots(figsize=(20, 10))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(20),

           yticks=np.arange(20),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(20):

        for j in range(20):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    #fig.tight_layout()

    return ax



def map(x):

    j = 0

    for i in MAPPING:

        if LABEL[x] in MAPPING[i]:

            k = j

            break

        else:

            j += 1

    return k
y_pred = np.zeros(shape=(10000,1))

y_true = np.zeros(shape=(10000,1))

for i in range (0,10000):

    y_pred[i] = map(np.argmax(predictions[i]))

    y_true[i] = map(np.argmax(y_test[i]))
plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=MAPPING, normalize=False, title="CONFUSION MATRIX WITHOUT NORMALIZATION")

plt.show()
plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=MAPPING, normalize=True, title="CONFUSION MATRIX WITH NORMALIZATION",

                     cmap = plt.cm.Greens)

plt.show()
# METHOD NO 3
from __future__ import print_function

import keras

from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os



batch_size = 32

num_classes = 100

epochs = 10

data_augmentation = True

num_predictions = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10_trained_model.h5'



# The data, split between train and test sets:

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# Convert class vectors to binary class matrices.

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# initiate RMSprop optimizer

opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)



# Let's train the model using RMSprop

model.compile(loss='categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255



if not data_augmentation:

    print('Not using data augmentation.')

    model.fit(x_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              validation_data=(x_test, y_test),

              shuffle=True)

else:

    print('Using real-time data augmentation.')

    # This will do preprocessing and realtime data augmentation:

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        zca_epsilon=1e-06,  # epsilon for ZCA whitening

        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

        # randomly shift images horizontally (fraction of total width)

        width_shift_range=0.1,

        # randomly shift images vertically (fraction of total height)

        height_shift_range=0.1,

        shear_range=0.,  # set range for random shear

        zoom_range=0.,  # set range for random zoom

        channel_shift_range=0.,  # set range for random channel shifts

        # set mode for filling points outside the input boundaries

        fill_mode='nearest',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False,  # randomly flip images

        # set rescaling factor (applied before any other transformation)

        rescale=None,

        # set function that will be applied on each input

        preprocessing_function=None,

        # image data format, either "channels_first" or "channels_last"

        data_format=None,

        # fraction of images reserved for validation (strictly between 0 and 1)

        validation_split=0.0)



    # Compute quantities required for feature-wise normalization

    # (std, mean, and principal components if ZCA whitening is applied).

    datagen.fit(x_train)



    # Fit the model on the batches generated by datagen.flow().

    model.fit_generator(datagen.flow(x_train, y_train,

                                     batch_size=batch_size),

                                    epochs=epochs,

                                    validation_data=(x_test, y_test),

                                    workers=4)



# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)



# Score trained model.

scores = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
model.save('cifar100.h5')
from google.colab import files

files.download('cifar100.h5')
predictions = model.predict([x_test])
y_pred = np.zeros(shape=(10000,1))

y_true = np.zeros(shape=(10000,1))

for i in range (0,10000):

    y_pred[i] = map(np.argmax(predictions[i]))

    y_true[i] = map(np.argmax(y_test[i]))
plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=MAPPING, normalize=False, title="CONFUSION MATRIX WITHOUT NORMALIZATION")

plt.show()
plot_confusion_matrix(y_true=y_true, y_pred=y_pred, classes=MAPPING, normalize=True, title="CONFUSION MATRIX WITH NORMALIZATION",

                     cmap = plt.cm.Greens)

plt.show()