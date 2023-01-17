import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
IMG_SIZE = 224



TRAINING_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train"

training_datagen = ImageDataGenerator(rescale = 1./255 ,

                                      shear_range=0.2,

                                      zoom_range=0.2)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR,

                                                       target_size=(IMG_SIZE,IMG_SIZE) ,class_mode='categorical',

                                                       batch_size=64,shuffle=True )





TEST_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test"

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(TEST_DIR,target_size=(IMG_SIZE,IMG_SIZE), class_mode = None , batch_size = 64,

                                                  shuffle = False)





VAL_DIR = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val"

val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = val_datagen.flow_from_directory(TEST_DIR,target_size=(IMG_SIZE,IMG_SIZE),class_mode='categorical',

                                                       batch_size=64,shuffle= False)

x,y = train_generator.next()

for i in range(0,1):

    image = x[i]

    plt.imshow(image)

    plt.show()
import tensorflow_hub as hub



URL = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4' 

feature_extractor = hub.KerasLayer(URL, input_shape=(224,224,3))

feature_extractor.trainable = False
model = tf.keras.models.Sequential([    

    feature_extractor,

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(2,activation='softmax')

    

])

model.summary()
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self,epoch,logs={}):

    if(logs['accuracy']>=0.95):

      self.model.stop_training=True



callbacks=myCallback()

METRICS = [

        'accuracy',

        tf.keras.metrics.Precision(name='precision'),

        tf.keras.metrics.Recall(name='recall')]

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=METRICS )



history = model.fit(train_generator , epochs=15 , callbacks=[callbacks], validation_data=val_generator)
fig, ax = plt.subplots(1, 4, figsize=(20, 3))

ax = ax.ravel()



for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
model.evaluate(val_generator)