import numpy as np

import tensorflow as tf

import math, cv2, os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib inline

import matplotlib.image  as mpimg

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from random import randint

from sklearn.utils import shuffle



print(tf.__version__)



train_dataset_path = "../input/seg_train/seg_train"

test_dataset_path = "../input/seg_test/seg_test"

pred_dataset_path = "../input/seg_pred/seg_pred"



IMG_SIZE      = 150

data_list     = os.listdir(train_dataset_path)

NUM_CLASSES   = len(data_list)

BATCH_SIZE    = 12  

EPOCHS        = 15


train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dataset_path,

        target_size=(IMG_SIZE, IMG_SIZE),

        color_mode='rgb',

        batch_size=BATCH_SIZE,

        shuffle=True,

        seed=None,

        class_mode="categorical")



test_generator = test_datagen.flow_from_directory(

        test_dataset_path,

        target_size=(IMG_SIZE, IMG_SIZE),

        color_mode='rgb',

        batch_size=BATCH_SIZE,

        shuffle=True,

        seed=None,

        class_mode="categorical")
l = tf.keras.layers

bnmomemtum=0.9

def fire(x, squeeze, expand):

  y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=3, activation='relu', padding='same')(x)

  y  = tf.keras.layers.BatchNormalization(momentum=bnmomemtum, center=True, scale=False)(y)

  y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)

  y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum, center=True, scale=False)(y1)

  y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=5, activation='relu', padding='same')(y)

  y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum, center=True, scale=False)(y3)

  return tf.keras.layers.concatenate([y1, y3])



def fire_module(squeeze, expand):

  return lambda x: fire(x, squeeze, expand)



x = l.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(x)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Conv2D(kernel_size=2, filters=6, padding='same', use_bias=True, activation='relu')(y)

y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum, center=True, scale=False)(y)

y = fire_module(12, 24)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(24, 48)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(24, 48)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(12, 24)(y)



y = tf.keras.layers.GlobalAveragePooling2D()(y)

y = tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')(y)

model = tf.keras.Model(x, y)



model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),  # 2e-5

              loss='categorical_crossentropy',

              metrics=['acc'])



model.summary()
def lr_decay(epoch):

  return 0.01 * math.pow(0.666, epoch)

lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)



class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.96):

      print("\nReached 96% accuracy so cancelling training!")

      self.model.stop_training = True



callbacks_max_acc = myCallback()



def get_images(directory):

    Images = []

        

    for image_file in all_image_paths:

        image=cv2.imread(directory+'/'+image_file)

        image=cv2.resize(image,(IMG_SIZE, IMG_SIZE))

        Images.append(image)

    

    return shuffle(Images,random_state=81732)





callback_early_stop = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, mode='auto', baseline=None, restore_best_weights=False)



# checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weights_only=True,verbose=1)



# model.load_weights(checkpoint_path) # Load ,   

   
history = model.fit_generator(

      train_generator,

      epochs=EPOCHS,

      validation_data=test_generator,

      callbacks=[callbacks_max_acc, lr_decay_callback])
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,max(plt.ylim())])

plt.title('Training and Validation Loss')

plt.show()
all_image_paths=os.listdir(pred_dataset_path)



print(all_image_paths[:10])



pred_images = get_images(pred_dataset_path)

pred_images = np.array(pred_images)

pred_images.shape





# classifications = model.predict(pred_images)

# print(classifications[7])
fig = plt.figure(figsize=(30, 30))

outer = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)



for i in range(25):

    inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    rnd_number = randint(0,len(pred_images))

    pred_image = np.array([pred_images[rnd_number]])

    pred_prob = model.predict(pred_image).reshape(6)

    for j in range(2):

        if (j%2) == 0:

            ax = plt.Subplot(fig, inner[j])

            ax.imshow(pred_image[0])

            ax.set_xticks([])

            ax.set_yticks([])

            fig.add_subplot(ax)

        else:

            ax = plt.Subplot(fig, inner[j])

            ax.bar([0,1,2,3,4,5],pred_prob)

            fig.add_subplot(ax)





fig.show()