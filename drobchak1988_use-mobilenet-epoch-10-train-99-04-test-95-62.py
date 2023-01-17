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

%load_ext tensorboard.notebook

%tensorboard --logdir logs

print(tf.__version__)



train_dataset_path = "../input/seg_train/seg_train"

test_dataset_path = "../input/seg_test/seg_test"

pred_dataset_path = "../input/seg_pred/seg_pred"



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dataset_path,

        target_size=(150, 150),

        color_mode='rgb',

        batch_size=32,

        shuffle=True,

        seed=None,

        class_mode="categorical")



test_generator = test_datagen.flow_from_directory(

        test_dataset_path,

        target_size=(150, 150),

        color_mode='rgb',

        batch_size=32,

        shuffle=True,

        seed=None,

        class_mode="categorical")
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3),

                                               include_top=False)

base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:

  layer.trainable =  False

model = tf.keras.Sequential([

  base_model,

  tf.keras.layers.GlobalAveragePooling2D(),

  tf.keras.layers.Dense(6, activation='sigmoid')

])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()
class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.99):

      print("\nReached 99% accuracy so cancelling training!")

      self.model.stop_training = True



callbacks_max_acc = myCallback()



def get_images(directory):

    Images = []

        

    for image_file in all_image_paths:

        image=cv2.imread(directory+'/'+image_file)

        image=cv2.resize(image,(150,150))

        Images.append(image)

    

    return shuffle(Images,random_state=817328462)



tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")
history = model.fit_generator(

      train_generator,

      epochs=10,

      validation_data=test_generator,

      callbacks=[callbacks_max_acc, tensorboard_callback])
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