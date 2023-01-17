import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
x_train, x_validation, y_train, y_validation = train_test_split(train, 

    train['label'], 

    test_size=0.2, 

    random_state=42,

    stratify=train['label'])
print(train.shape)

train.head()
# We'll create a "split" of training, for cross-validation

train_images_split = []

train_labels_split = []

validation_images = []

validation_labels = []



# We'll create a full training set

train_images_full = []

train_labels_full = []



# And the test set

test_images = []



for index, row in x_train.iterrows():

    train_images_split.append(row.values[1 : ].reshape((28, 28)))

    train_labels_split.append(row['label'])

    train_images_full.append(row.values[1 : ].reshape((28, 28)))

    train_labels_full.append(row['label'])

    

for index, row in x_validation.iterrows():

    validation_images.append(row.values[1 : ].reshape((28, 28)))

    validation_labels.append(row['label'])

    train_images_full.append(row.values[1 : ].reshape((28, 28)))

    train_labels_full.append(row['label'])



for index, row in test.iterrows():

    test_images.append(row.values.reshape((28, 28)))

    

# Convert numpy array, while normalizing the image data

train_labels_split = np.array(train_labels_split)

train_images_split = np.array(train_images_split) / 255.

validation_labels = np.array(validation_labels)

validation_images = np.array(validation_images) / 255.

train_labels_full = np.array(train_labels_full)

train_images_full = np.array(train_images_full) / 255.

test_images = np.array(test_images) / 255.



print(train_labels_full.shape)

print(train_images_full.shape)

print(train_labels_split.shape)

print(train_images_split.shape)

print(validation_labels.shape)

print(validation_images.shape)

print(test_images.shape)
fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(train_images_full[42], cmap='gray')

axs[0, 1].imshow(train_images_full[7], cmap='gray')

axs[1, 0].imshow(train_images_full[2020], cmap='gray')

axs[1, 1].imshow(train_images_full[0], cmap='gray')
# Callback function so we can stop training once we've reached a desired level of accuracy.

class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('accuracy')>=0.995):

      print("\n99.5% accuracy reached, stopping!")

      self.model.stop_training = True
# Add an extra dimension

train_images = np.expand_dims(train_images_split, axis=3)

validation_images = np.expand_dims(validation_images, axis=3)
# Model definition

model = tf.keras.models.Sequential([

    # Convolutional layer 1

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # Convolutional layer 2

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    # Final layers

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])



# Model compiler

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



# Generators

train_gen = ImageDataGenerator().flow(train_images, train_labels_split, batch_size=32)

validation_gen = ImageDataGenerator().flow(validation_images, validation_labels, batch_size=32)



# Model fit

history = model.fit(train_gen,

                    validation_data = validation_gen,

                    steps_per_epoch = len(train_images) / 32,

                    validation_steps = len(validation_images) / 32,    

                    epochs=10, 

                    callbacks=[myCallback()])
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()
# Redefine training images and labels, using full sets

train_images = np.expand_dims(train_images_full, axis=3)

train_labels = train_labels_full



# Re-define generator

train_gen = ImageDataGenerator().flow(train_images, train_labels, batch_size=32)



# Model fit

history = model.fit(train_gen,

                    steps_per_epoch = len(train_images) / 32,  

                    epochs=10, 

                    callbacks=[myCallback()])
test_images = np.expand_dims(test_images, axis=3)

preds = model.predict_classes(test_images)
my_submission = pd.concat([pd.Series(range(1,28001), name = "ImageId")],axis = 1)

my_submission['Label'] = preds

my_submission.to_csv("my_submission.csv", index=False)