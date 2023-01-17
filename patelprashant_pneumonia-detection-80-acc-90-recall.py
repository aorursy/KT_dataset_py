import os

import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, Input, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint



from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report



tf.random.set_seed(42)
input_dir = '../input/chest-xray-pneumonia/chest_xray/'



train_dir = os.path.join(input_dir, 'train')

val_dir = os.path.join(input_dir, 'val')

test_dir = os.path.join(input_dir, 'test')



pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')

pneumonia_val_dir = os.path.join(val_dir, 'PNEUMONIA')

pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')



normal_train_dir = os.path.join(train_dir, 'NORMAL')

normal_val_dir = os.path.join(val_dir, 'NORMAL')

normal_test_dir = os.path.join(test_dir, 'NORMAL')



pneumonia_train_images = len(os.listdir(pneumonia_train_dir))

pneumonia_val_images = len(os.listdir(pneumonia_val_dir))

pneumonia_test_images = len(os.listdir(pneumonia_test_dir))



normal_train_images = len(os.listdir(normal_train_dir))

normal_val_images = len(os.listdir(normal_val_dir))

normal_test_images = len(os.listdir(normal_test_dir))



train_size = pneumonia_train_images + normal_train_images

test_size = pneumonia_test_images + normal_test_images

val_size = pneumonia_val_images + normal_val_images



print(f'Total training images: {pneumonia_train_images + normal_train_images}')

print(f'Pneumonia: {pneumonia_train_images}')

print(f'Normal: {normal_train_images}')

print('---')

print(f'Total testing images: {pneumonia_test_images + normal_test_images}')

print(f'Pneumonia: {pneumonia_test_images}')

print(f'Normal: {normal_test_images}')

print('---')

print(f'Total validation images: {pneumonia_val_images + normal_val_images}')

print(f'Pneumonia: {pneumonia_val_images}')

print(f'Normal: {normal_val_images}')
IMG_HEIGHT = 150

IMG_WIDTH = 150

BATCH_SIZE = 16

EPOCHS = 15

CHECKPOINT_FILEPATH = '/tmp/checkpoint'
train_datagen = ImageDataGenerator(rescale=1./255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(directory=train_dir,

                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                    batch_size=BATCH_SIZE,

                                                    class_mode='binary')



test_generator = test_datagen.flow_from_directory(directory=test_dir,

                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                  batch_size=BATCH_SIZE,

                                                  class_mode='binary')



val_generator = test_datagen.flow_from_directory(directory=val_dir,

                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                 batch_size=BATCH_SIZE,

                                                 class_mode='binary')
CLASS_NAMES_DICT = {value: name for name, value in train_generator.class_indices.items()}

def show_batch(image_batch, label_batch):

    plt.figure(figsize=(10,10))

    for n in range(15):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        plt.title(CLASS_NAMES_DICT[label_batch[n]])

        plt.axis('off')
image_batch, label_batch = next(train_generator)

show_batch(image_batch, label_batch)
def create_model():

    model = Sequential()



    model.add(Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))



    model.add(Conv2D(16, (3,3), activation='relu'))

    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.2))



    model.add(Conv2D(32, (3,3), activation='relu'))

    model.add(MaxPooling2D(2,2))



    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    

    model.compile(optimizer='rmsprop', 

              loss='binary_crossentropy', 

              metrics=['accuracy'])

    

    return model
model = create_model()

model.summary()
# clear session variables

tf.keras.backend.clear_session()



# checkpoint callback

model_checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINT_FILEPATH,

                                            save_weights_only=True,

                                            monitor='val_accuracy',

                                            mode='max',

                                            save_best_only=True)



# training

history = model.fit(train_generator,

                    steps_per_epoch=train_size//BATCH_SIZE,

                    validation_data=val_generator, 

                    validation_steps=val_size//BATCH_SIZE,

                    epochs=EPOCHS,

                    verbose=1,

                    callbacks=[model_checkpoint_callback])
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epoch_range = range(len(acc))



plt.plot(epoch_range, acc)

plt.plot(epoch_range, val_acc)

plt.title('Training and Validation Accuracy')



plt.figure()



plt.plot(epoch_range, loss)

plt.plot(epoch_range, val_loss)

plt.title('Training and Validation Loss')
# load weights of the best checkpoint

model.load_weights(CHECKPOINT_FILEPATH)



# save best checkpoint as hdf5

model.save('/tmp/best_model.hdf5')



# load the best model

best_model = tf.keras.models.load_model('/tmp/best_model.hdf5')



# calculate test accuracy using the best model

loss, acc = best_model.evaluate_generator(test_generator)

print(f'Testing Loss: {loss} | Testing Accuracy: {acc}')
actual_labels, predicted_labels = [],[]

for _ in range(len(test_generator)):

    test_images, test_labels = next(test_generator)

    actual_labels.append(test_labels)

    predicted_labels.append(best_model.predict_classes(test_images).ravel())



actual_labels = np.array(actual_labels, dtype=float).ravel()

predicted_labels = np.array(predicted_labels, dtype=float).ravel()
accuracy = accuracy_score(actual_labels, predicted_labels)

precision = precision_score(actual_labels, predicted_labels)

recall = recall_score(actual_labels, predicted_labels)



print(f'Accuracy: {accuracy} | Precision: {precision} | Recall: {recall}')
print(classification_report(actual_labels, predicted_labels))