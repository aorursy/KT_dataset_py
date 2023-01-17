from petal_helper import *

import tensorflow.compat.v1 as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.xception import Xception

from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
ds_train = get_training_dataset()

ds_valid = get_validation_dataset()

ds_test = get_test_dataset()
model_path = 'model.h5'

checkpoint = ModelCheckpoint(model_path,

                            monitor='val_sparse_categorical_accuracy',

                            save_best_only=True,

                            verbose=1)

reduce_callback = ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',

                                   patience=3,

                                   factor=0.5,

                                   min_lr=0.00001,

                                   verbose=1)

callbacks_list = [reduce_callback, checkpoint]
with strategy.scope():

    xcept = Xception(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

    

    model = Sequential()

    model.add(xcept)

    model.add(Dropout(0.3))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(len(CLASSES), activation='softmax'))

    

    model.compile(optimizer=Adam(lr=1e-5),

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy'])
model.summary()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
history = model.fit(ds_train,

    validation_data=ds_valid,

    epochs=20,

    steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,

    validation_steps=NUM_VALIDATION_IMAGES // BATCH_SIZE,

    verbose=1,

    callbacks=callbacks_list)
model.load_weights(model_path)
plt.plot(history.history['sparse_categorical_accuracy'], 

         label='Accuracy')

plt.plot(history.history['val_sparse_categorical_accuracy'],

         label='Validation accuracy')

plt.xlabel('Epoch')

plt.ylabel('Percentage of correct responses')

plt.title('Accuracy')

plt.legend()

plt.show()
plt.plot(history.history['loss'], 

         label='Loss')

plt.plot(history.history['val_loss'],

         label='Validation loss')

plt.xlabel('Epoch')

plt.ylabel('Percentage of loss')

plt.title('Loss')

plt.legend()

plt.show()
test_ds = get_test_dataset(ordered=True)

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')



np.savetxt(

    'submission.csv',

    np.rec.fromarrays([test_ids, predictions]),

    fmt=['%s', '%d'],

    delimiter=',',

    header='id,label',

    comments='',

)