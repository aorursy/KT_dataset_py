import pickle



%matplotlib inline

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd



from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from lrutils import *
IMG_SHAPE = (128, 128, 3)
# Load data

train_df = pd.read_pickle('../input/cat-vs-dog-data/train_df.pkl')

train_df.filename = train_df.filename.map(lambda s: s.split('\\')[1])

validate_df = pd.read_pickle('../input/cat-vs-dog-data/validate_df.pkl')

validate_df.filename = validate_df.filename.map(lambda s: s.split('\\')[1])

train_dir = '../input/dogs-vs-cats/train/train'
train_datagen = ImageDataGenerator(rescale = 1./255.,

                                   rotation_range=20,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    train_dir,

                                                    x_col='filename',

                                                    y_col='category',

                                                    batch_size=50,

                                                    class_mode='binary',

                                                    target_size=(128,128))     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (128,128))
def plot_history(history):

    #-----------------------------------------------------------

    # Retrieve a list of list results on training and test data

    # sets for each training epoch

    #-----------------------------------------------------------

    acc      = history.history[     'acc' ]

    val_acc  = history.history[ 'val_acc' ]

    loss     = history.history[    'loss' ]

    val_loss = history.history['val_loss' ]



    epochs   = range(len(acc)) # Get number of epochs



    #------------------------------------------------

    # Plot training and validation accuracy per epoch

    #------------------------------------------------

    plt.plot  ( epochs,     acc , label='Training')

    plt.plot  ( epochs, val_acc , label='Validation')

    plt.xlabel ('Epoch')

    plt.ylabel ('Accuracy')

    plt.legend ()

    plt.title ('Training and validation accuracy')

    plt.figure()



    #------------------------------------------------

    # Plot training and validation loss per epoch

    #------------------------------------------------

    plt.plot  ( epochs,     loss, label='Training')

    plt.plot  ( epochs, val_loss, label='Validation')

    plt.xlabel ('Epoch')

    plt.ylabel ('Loss')

    plt.legend ()

    plt.title ('Training and validation loss'   )
def report(model, history=None, validation_generator=None):

    if history is not None:

        plot_history(history)

    

    if validation_generator is not None:

        # Evaluate trained model on validation set

        validation_generator.reset()

        [val_loss, val_acc] = model.evaluate_generator(validation_generator)

        print('Model evaluation')

        print(f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')

        print()
mobilenet = keras.applications.MobileNetV2(input_shape = IMG_SHAPE,

                                                   include_top = False,

                                                   weights = 'imagenet')
mobilenet.trainable = False
classifier_mobilenet = keras.Sequential([

    mobilenet,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(1, activation='sigmoid')

])
classifier_mobilenet.summary()
base_learning_rate = 0.0001
classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=base_learning_rate),

              loss='binary_crossentropy',

              metrics=['accuracy'])
savebest = keras.callbacks.ModelCheckpoint('mobilenet_weights.h5', monitor='val_loss', mode='min',

                                             save_best_only=True, save_weights_only=True,

                                             verbose=1, save_freq='epoch')
history = classifier_mobilenet.fit_generator(train_generator,

                              epochs=10,

                              validation_data=validation_generator,

                                callbacks=[savebest])
classifier_mobilenet.load_weights('mobilenet_weights.h5')

report(classifier_mobilenet, history, validation_generator)
classifier_mobilenet = keras.Sequential([

    mobilenet,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(1, activation='sigmoid')

])
classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(),

              loss='binary_crossentropy',

              metrics=['accuracy'])
lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 2**(epoch/2))
history = classifier_mobilenet.fit_generator(train_generator,

                              epochs=30,

                              validation_data=validation_generator,

                                callbacks=[lr_schedule],

                                steps_per_epoch=1)
# Plot the tuning history

lrs = 1e-6 * 2**(np.arange(30)/2)

plt.semilogx(lrs, history.history["val_loss"])

plt.axis([1e-5, 1e-2, 0.2, 1.0])
for epoch in range(20,25):

    print(f'lr = {history.history["lr"][epoch]}, val_loss: {history.history["val_loss"][epoch]}')
tuned_learning_rate = 0.0014
classifier_mobilenet = keras.Sequential([

    mobilenet,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(1, activation='sigmoid')

])
classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=tuned_learning_rate),

              loss='binary_crossentropy',

              metrics=['accuracy'])
savebest = keras.callbacks.ModelCheckpoint('tunedlr_weights.h5', monitor='val_loss', mode='min',

                                             save_best_only=True, save_weights_only=True,

                                             verbose=1, save_freq='epoch')
history = classifier_mobilenet.fit_generator(train_generator,

                              epochs=10,

                              validation_data=validation_generator,

                                callbacks=[savebest])
classifier_mobilenet.load_weights('tunedlr_weights.h5')

report(classifier_mobilenet, history, validation_generator)
def compute_lrs(max_lr, epochs, batches_per_epoch, cycle_mult=1):

    lrs = []

    cycle_iterations = 0

    epoch_iterations = batches_per_epoch

    for _ in range(epochs*batches_per_epoch):

        decay_phase = np.pi*cycle_iterations/epoch_iterations

        decay = (np.cos(decay_phase) + 1.) / 2.

        lrs.append(max_lr*decay)

        cycle_iterations += 1

        if cycle_iterations == epoch_iterations:

            cycle_iterations = 0

            epoch_iterations *= cycle_mult

    return lrs
lrs = compute_lrs(0.0014, 10, 450)

plt.plot(np.arange(10*450), lrs)

plt.xlabel('Batch number')

plt.ylabel('Learning rate')

plt.show()
classifier_mobilenet = keras.Sequential([

    mobilenet,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(1, activation='sigmoid')

])
classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=tuned_learning_rate),

              loss='binary_crossentropy',

              metrics=['accuracy'])
cyclic_lr = LR_Cycle(450)
savebest = keras.callbacks.ModelCheckpoint('cyclic_weights.h5', monitor='val_loss', mode='min',

                                             save_best_only=True, save_weights_only=True,

                                             verbose=1, save_freq='epoch')
history = classifier_mobilenet.fit_generator(train_generator,

                              epochs=10,

                              validation_data=validation_generator,

                                callbacks=[cyclic_lr, savebest])
classifier_mobilenet.load_weights('cyclic_weights.h5')

report(classifier_mobilenet, history, validation_generator)
lrs = compute_lrs(0.0014, 15, 450, 2)

plt.plot(np.arange(15*450), lrs)

plt.xlabel('Batch number')

plt.ylabel('Learning rate')

plt.show()
classifier_mobilenet = keras.Sequential([

    mobilenet,

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(1, activation='sigmoid')

])
classifier_mobilenet.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0014),

              loss='binary_crossentropy',

              metrics=['accuracy'])
cyclic_lr = LR_Cycle(450, cycle_mult=2)
savebest = keras.callbacks.ModelCheckpoint('cycleMult2_weights.h5', monitor='val_loss', mode='min',

                                             save_best_only=True, save_weights_only=True,

                                             verbose=1, save_freq='epoch')
history = classifier_mobilenet.fit_generator(train_generator,

                              epochs=15,

                              validation_data=validation_generator,

                                callbacks=[cyclic_lr, savebest])
classifier_mobilenet.load_weights('cycleMult2_weights.h5')

report(classifier_mobilenet, history, validation_generator)