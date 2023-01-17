import tensorflow

import keras

# If you get a FutureWarning from running this c
from keras.preprocessing.image import ImageDataGenerator



generator = ImageDataGenerator()

batches = generator.flow_from_directory('../input/100-bird-species/train', batch_size=4)



batches
indices = batches.class_indices

labels = [None] * 225



for key in indices:

    labels[indices[key]] = key



labels
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np



for X, y in batches:

    fig, ax = plt.subplots(1, 4, figsize=(10, 10))

    

    for i in range(len(X)):

        img = X[i].astype(np.uint8)

        label = labels[np.argmax(y[i])]



        ax[i].imshow(img)

        ax[i].set_title(label)

        ax[i].set_xticks([])

        ax[i].set_yticks([])

    

    plt.show()

    break # We only need the first batch
from keras.applications.vgg19 import VGG19
model = VGG19(weights='imagenet', include_top=True, 

              input_shape=(224, 224, 3))

model.summary()
np.random.seed(1234)

generator = ImageDataGenerator()

batches = generator.flow_from_directory('../input/100-bird-species/train', 

                                        target_size=(224, 224), 

                                        batch_size=1)
from keras.applications.vgg19 import decode_predictions



for X, y in batches:

    preds = model.predict(X)

    decoded_preds = decode_predictions(preds, top=1)

    fig = plt.figure()

    

    img = X[0].astype(np.uint8)

    label = labels[np.argmax(y[0])]

    predicted = decoded_preds[0]

    

    plt.imshow(img)

    fig.suptitle('Truth: {}, Predicted: {}'.format(label, predicted))

    plt.show()

    

    break
from keras.applications.vgg19 import preprocess_input



np.random.seed(1234)

generator = ImageDataGenerator(preprocessing_function=preprocess_input)

batches = generator.flow_from_directory('../input/100-bird-species/train', 

                                        target_size=(224, 224),

                                        batch_size=1)
for X, y in batches:

    preds = model.predict(X)

    decoded_preds = decode_predictions(preds, top=1)

    fig = plt.figure()

    

    img = X[0].astype(np.uint8)

    label = labels[np.argmax(y[0])]

    predicted = decoded_preds[0]

    

    plt.imshow(img)

    fig.suptitle('Truth: {}, Predicted: {}'.format(label, predicted))

    plt.show()

    

    break
pretrained = VGG19(include_top=False, input_shape=(224, 224, 3), 

                   weights='imagenet', pooling='max')

inputs = pretrained.input

outputs = pretrained.output
for layer in pretrained.layers:

    layer.trainable = False
from keras.layers import Dense



hidden = Dense(128, activation='relu')(outputs)

preds = Dense(225, activation='softmax')(hidden)
from keras.engine import Model

from keras.optimizers import Adam



model = Model(inputs, preds)

model.compile(loss='categorical_crossentropy', 

              optimizer=Adam(lr=1e-4),

              metrics=['acc'])



model.summary()
np.random.seed(1234)



# If you run into memory errors, try reducing this

batch_size = 32



train_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input)

train_batches = train_generator.flow_from_directory('../input/100-bird-species/train',

                                                    target_size=(224, 224), 

                                                    batch_size=batch_size)



val_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input)

val_batches = val_generator.flow_from_directory('../input/100-bird-species/valid',

                                                target_size=(224, 224),

                                                batch_size=batch_size)



# Note that training is set to 1 epoch, 

# to avoid unintentionally locking up computers

model.fit_generator(train_batches, 

                    epochs=1, 

                    validation_data=val_batches, 

                    steps_per_epoch=len(train_batches), 

                    validation_steps=len(val_batches))
from keras.layers import Dropout



hidden = Dense(128, activation='relu')(outputs)

dropout = Dropout(.3)(hidden)

preds = Dense(225, activation='softmax')(dropout)



model = Model(inputs, preds)

model.compile(loss='categorical_crossentropy', 

              optimizer=Adam(lr=1e-4),

              metrics=['acc'])



model.summary()
np.random.seed(1234)



# If you run into memory errors, try reducing this

batch_size = 32



train_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input)

train_batches = train_generator.flow_from_directory('../input/100-bird-species/train',

                                                    target_size=(224, 224),

                                                    batch_size=batch_size)



val_generator = ImageDataGenerator(

    preprocessing_function=preprocess_input)

val_batches = val_generator.flow_from_directory('../input/100-bird-species/train',

                                                target_size=(224, 224),

                                                batch_size=batch_size)



# Note that training is set to 1 epoch, 

# to avoid unintentionally locking up computers

model.fit_generator(train_batches, 

                    epochs=1, 

                    validation_data=val_batches,

                    steps_per_epoch=len(train_batches), 

                    validation_steps=len(val_batches))
np.random.seed(1234)



generator = ImageDataGenerator(horizontal_flip=True)

batches = generator.flow_from_directory('../input/100-bird-species/train',

                                        batch_size=1,

                                        shuffle=False)
fig, ax = plt.subplots(1, 5, figsize=(15, 10))



for i in range(5):

    batches = generator.flow_from_directory('../input/100-bird-species/train', 

                                            batch_size=1,

                                            shuffle=False)

    for X, y in batches:

        ax[i].imshow(X[0].astype(np.uint8))

        ax[i].set_title('Run {}'.format(i + 1))

        ax[i].set_xticks([])

        ax[i].set_yticks([])

        break



plt.show()