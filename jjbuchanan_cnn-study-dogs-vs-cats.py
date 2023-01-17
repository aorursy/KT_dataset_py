import os

import pickle



%matplotlib inline

import matplotlib.image as mpimg

import matplotlib.pyplot as plt



from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd



from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation='relu'), 

    tf.keras.layers.Dense(1, activation='sigmoid')  

])
model.summary()
model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
# Load data

train_df = pd.read_pickle('../input/cat-vs-dog-data/train_df.pkl')

train_df.filename = train_df.filename.map(lambda s: s.split('\\')[1])

validate_df = pd.read_pickle('../input/cat-vs-dog-data/validate_df.pkl')

validate_df.filename = validate_df.filename.map(lambda s: s.split('\\')[1])

train_dir = '../input/dogs-vs-cats/train/train'
train_datagen = ImageDataGenerator(rescale = 1./255.)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    train_dir,

                                                    x_col='filename',

                                                    y_col='category',

                                                    batch_size=50,

                                                    class_mode='binary',

                                                    target_size=(150, 150),

                                                       seed=42)     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150),

                                                            seed=42)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, restore_best_weights=True)
# history = model.fit_generator(train_generator,

#                               epochs=15,

#                               validation_data=validation_generator,

#                              callbacks=[earlystopping])
def save_history(history, filename):

    with open(filename, 'wb') as file_pi:

        pickle.dump((history.epoch, history.history), file_pi)
def load_history(filename):

    history = keras.callbacks.History()

    with open(filename, 'rb') as file_pi:

        (history.epoch, history.history) = pickle.load(file_pi)

    return history
def save_earlystopping(earlystopping, filename):

    with open(filename, 'wb') as file_pi:

        pickle.dump((earlystopping.stopped_epoch, earlystopping.patience,

                    earlystopping.monitor, earlystopping.min_delta,

                    earlystopping.monitor_op, earlystopping.restore_best_weights,

                    earlystopping.wait, earlystopping.baseline), file_pi)
def load_earlystopping(filename):

    earlystopping = keras.callbacks.EarlyStopping()

    with open(filename, 'rb') as file_pi:

        (earlystopping.stopped_epoch, earlystopping.patience,

         earlystopping.monitor, earlystopping.min_delta,

         earlystopping.monitor_op, earlystopping.restore_best_weights,

         earlystopping.wait, earlystopping.baseline) = pickle.load(file_pi)

    return earlystopping
# # Save model

# model.save('initial_model.h5')

# save_history(history, 'initial_history.pkl')

# save_earlystopping(earlystopping, 'initial_earlystopping.pkl')
# Load model

model = keras.models.load_model('../input/cat-vs-dog-cnn-models/initial_model.h5')

history = load_history('../input/cat-vs-dog-cnn-models/initial_history.pkl')

earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/initial_earlystopping.pkl')
def check_early_stopping(history, earlystopping):

    print('Early stopping')

    monitor = history.history[earlystopping.monitor]

    fun1, fun2 = np.min, np.argmin

    best = 'Lowest'

    if earlystopping.monitor_op == np.greater:

        fun1, fun2 = np.max, np.argmax

        best = 'Highest'

    print(f'  Monitor: {earlystopping.monitor}')

    print(f'    {best} value: {fun1(monitor)}')

    print(f'    Epoch: {fun2(monitor)+1}')

    stopped_epoch = earlystopping.stopped_epoch - earlystopping.patience + 1

    if earlystopping.stopped_epoch == 0:

        stopped_epoch = 'None (stopped_epoch==0)'

    print(f'  Epoch detected by early stopping: {stopped_epoch}')

    if not earlystopping.restore_best_weights or earlystopping.stopped_epoch == 0:

        print('  Best weights NOT returned')

    else:

        print('  Best weights returned')
def plot_history(history):

    acc      = history.history[     'acc' ]

    val_acc  = history.history[ 'val_acc' ]

    loss     = history.history[    'loss' ]

    val_loss = history.history['val_loss' ]



    epochs   = range(len(acc))



    plt.plot  ( epochs,     acc , label='Training')

    plt.plot  ( epochs, val_acc , label='Validation')

    plt.xlabel ('Epoch')

    plt.ylabel ('Accuracy')

    plt.legend ()

    plt.title ('Training and validation accuracy')

    plt.figure()



    plt.plot  ( epochs,     loss, label='Training')

    plt.plot  ( epochs, val_loss, label='Validation')

    plt.xlabel ('Epoch')

    plt.ylabel ('Loss')

    plt.legend ()

    plt.title ('Training and validation loss'   )
def report(model, validation_generator, history=None, earlystopping=None):

    if earlystopping is not None:

        check_early_stopping(history, earlystopping)

        print()

    

    if history is not None:

        plot_history(history)

    

    # Evaluate trained model on validation set

    validation_generator.reset()

    [val_loss, val_acc] = model.evaluate_generator(validation_generator)

    print('Model evaluation')

    print(f'val_loss: {val_loss}, val_acc: {val_acc}')

    print()

    

    # Compute ROC curve

    validation_generator.reset()

    validation_set = [validation_generator.next() for _ in range(len(validation_generator))]

    val_images = np.concatenate([validation_set[i][0] for i in range(len(validation_set))])

    val_y = np.concatenate([validation_set[i][1] for i in range(len(validation_set))])

    fpr, tpr, thresholds = roc_curve(val_y, model.predict(val_images))

    

    return fpr, tpr
class ROCCurveParams():

    def __init__(self, fpr, tpr, color, linestyle):

        self.fpr, self.tpr, self.color, self.linestyle = fpr, tpr, color, linestyle
def plot_roc_curves(roc_curves):

    plt.figure()

    for curvename in roc_curves:

        c = roc_curves[curvename]

        plt.plot(c.fpr, c.tpr, color=c.color, lw=2, label=curvename, linestyle=c.linestyle)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Validation set ROC')

    plt.legend(loc="lower right")

    plt.show()
roc_curves = dict()
fpr, tpr = report(model, validation_generator, history, earlystopping)

roc_curves['Initial model'] = ROCCurveParams(fpr, tpr, 'darkorange', '-')
plot_roc_curves(roc_curves)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
model.summary()
train_datagen = ImageDataGenerator(rescale = 1./255.)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    train_dir,

                                                    x_col='filename',

                                                    y_col='category',

                                                    batch_size=50,

                                                    class_mode='binary',

                                                    target_size=(150, 150),

                                                       seed=42)     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150),

                                                            seed=42)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, restore_best_weights=True)
# history = model.fit_generator(train_generator,

#                               epochs=30,

#                               validation_data=validation_generator,

#                              callbacks=[earlystopping])
# # Save model

# model.save('dropout_model.h5')

# save_history(history, 'dropout_history.pkl')

# save_earlystopping(earlystopping, 'dropout_earlystopping.pkl')
# Load model

model = keras.models.load_model('../input/cat-vs-dog-cnn-models/dropout_model.h5')

history = load_history('../input/cat-vs-dog-cnn-models/dropout_history.pkl')

earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/dropout_earlystopping.pkl')
fpr, tpr = report(model, validation_generator, history, earlystopping)

roc_curves['Dropout'] = ROCCurveParams(fpr, tpr, 'green', '-')

plot_roc_curves(roc_curves)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
train_datagen = ImageDataGenerator(rescale = 1./255.)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    train_dir,

                                                    x_col='filename',

                                                    y_col='category',

                                                    batch_size=50,

                                                    class_mode='binary',

                                                    target_size=(150, 150),

                                                       seed=42)     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150),

                                                            seed=42)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)

reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)
# history = model.fit_generator(train_generator,

#                               epochs=40,

#                               validation_data=validation_generator,

#                              callbacks=[earlystopping, reducelr])
# # Save model

# model.save('redlr_model.h5')

# save_history(history, 'redlr_history.pkl')

# save_earlystopping(earlystopping, 'redlr_earlystopping.pkl')
# Load model

model = keras.models.load_model('../input/cat-vs-dog-cnn-models/redlr_model.h5')

history = load_history('../input/cat-vs-dog-cnn-models/redlr_history.pkl')

earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/redlr_earlystopping.pkl')
fpr, tpr = report(model, validation_generator, history, earlystopping)

roc_curves['Reduce LR'] = ROCCurveParams(fpr, tpr, 'red', '--')

plot_roc_curves(roc_curves)
roc_curves['Dropout'] = roc_curves.pop('Reduce LR')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
train_datagen = ImageDataGenerator(rescale = 1./255.)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    train_dir,

                                                    x_col='filename',

                                                    y_col='category',

                                                    batch_size=50,

                                                    class_mode='binary',

                                                    target_size=(150, 150),

                                                       seed=42)     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150),

                                                            seed=42)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)

reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)
# history = model.fit_generator(train_generator,

#                               epochs=40,

#                               validation_data=validation_generator,

#                              callbacks=[earlystopping, reducelr])
# # Save model

# model.save('bnorm_model.h5')

# save_history(history, 'bnorm_history.pkl')

# save_earlystopping(earlystopping, 'bnorm_earlystopping.pkl')
# Load model

model = keras.models.load_model('../input/cat-vs-dog-cnn-models/bnorm_model.h5')

history = load_history('../input/cat-vs-dog-cnn-models/bnorm_history.pkl')

earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/bnorm_earlystopping.pkl')
fpr, tpr = report(model, validation_generator, history, earlystopping)

roc_curves['Batch Norm'] = ROCCurveParams(fpr, tpr, 'Blue', '-')

plot_roc_curves(roc_curves)
roc_curves['Dropout'] = roc_curves.pop('Batch Norm')
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
model.summary()
train_datagen = ImageDataGenerator(rescale = 1./255.)

validation_datagen = ImageDataGenerator(rescale = 1./255.)



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                    train_dir,

                                                    x_col='filename',

                                                    y_col='category',

                                                    batch_size=50,

                                                    class_mode='binary',

                                                    target_size=(150, 150),

                                                       seed=42)     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150),

                                                            seed=42)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)

reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)
# history = model.fit_generator(train_generator,

#                               epochs=50,

#                               validation_data=validation_generator,

#                              callbacks=[earlystopping, reducelr])
# # Save model

# model.save('big_model.h5')

# save_history(history, 'big_history.pkl')

# save_earlystopping(earlystopping, 'big_earlystopping.pkl')
# Load model

model = keras.models.load_model('../input/cat-vs-dog-cnn-models/big_model.h5')

history = load_history('../input/cat-vs-dog-cnn-models/big_history.pkl')

earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/big_earlystopping.pkl')
fpr, tpr = report(model, validation_generator, history, earlystopping)

roc_curves['Double filters'] = ROCCurveParams(fpr, tpr, 'Green', '-')

plot_roc_curves(roc_curves)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(optimizer=RMSprop(lr=0.001),

              loss='binary_crossentropy',

              metrics = ['acc'])
model.summary()
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

                                                    target_size=(150, 150),

                                                       seed=42)     

validation_generator =  validation_datagen.flow_from_dataframe(validate_df,

                                                          train_dir,

                                                          x_col='filename',

                                                          y_col='category',

                                                         batch_size=50,

                                                         class_mode  = 'binary',

                                                         target_size = (150, 150),

                                                              seed=42)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=9, restore_best_weights=True)

reducelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=0.00001)
# history = model.fit_generator(train_generator,

#                               epochs=50,

#                               validation_data=validation_generator,

#                              callbacks=[earlystopping, reducelr])
# # Save model

# model.save('daug_model.h5')

# save_history(history, 'daug_history.pkl')

# save_earlystopping(earlystopping, 'daug_earlystopping.pkl')
# Load model

model = keras.models.load_model('../input/cat-vs-dog-cnn-models/daug_model.h5')

history = load_history('../input/cat-vs-dog-cnn-models/daug_history.pkl')

earlystopping = load_earlystopping('../input/cat-vs-dog-cnn-models/daug_earlystopping.pkl')
fpr, tpr = report(model, validation_generator, history, earlystopping)

roc_curves['Data augment'] = ROCCurveParams(fpr, tpr, 'Purple', '-')

plot_roc_curves(roc_curves)