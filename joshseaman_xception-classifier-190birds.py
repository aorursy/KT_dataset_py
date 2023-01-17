# Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import Xception
from sklearn.metrics import confusion_matrix
parent_dir = '/kaggle/input/100-bird-species/consolidated'
cats = os.listdir(path=parent_dir)

subcats = cats[0:15]
fig = plt.figure(figsize = [16,12])
for category in subcats:
    img = os.listdir(path=os.path.join(parent_dir,subcats[1]))[1]
    plt.subplot(3,5,subcats.index(category)+1, title = category)
    path = os.path.join(parent_dir, category)
    img_array = cv2.imread(os.path.join(path,img))
    plt.imshow(img_array)
plt.show()
base_dir = '/kaggle/input/100-bird-species/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir,'test')
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

epoch = 100

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=20,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=20,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=20,
    class_mode='categorical')
backend.clear_session()
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(224,224,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(BatchNormalization())
model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(len(cats), activation='softmax'))

model.compile(optimizer = 'adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit_generator(train_generator,
                              epochs = epoch,
                              validation_data = validation_generator,
                              verbose = 1,
                              callbacks = [EarlyStopping(monitor='val_accuracy', 
                                                         patience = 5,
                                                         restore_best_weights=True)])

test_loss, test_acc = model.evaluate_generator(test_generator)
print('base_model_test_acc:', test_acc)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
epoch = 50
train_datagen2 = ImageDataGenerator(rescale=1./255,
                                    rotation_range=10,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip = True,
                                    fill_mode='nearest')
test_datagen2 = ImageDataGenerator(rescale=1./255)
train_generator2 = train_datagen2.flow_from_directory(train_dir,
                                                     target_size=(224,224),
                                                     batch_size=20,
                                                     class_mode='categorical')
validation_generator2 = train_datagen2.flow_from_directory(validation_dir,
                                                          target_size=(224,224),
                                                          batch_size=20,
                                                          class_mode='categorical')
test_generator2 = test_datagen2.flow_from_directory(test_dir,
                                                   target_size=(224,224),
                                                   batch_size=20,
                                                   class_mode='categorical')
backend.clear_session()
conv_base = Xception (weights = 'imagenet',
                    include_top = False,
                    input_shape = (224,224,3))
for layer in conv_base.layers[:-6]:
    layer.trainable = False

modelx = models.Sequential()
modelx.add(conv_base)
modelx.add(layers.Flatten())
modelx.add(layers.Dense(512, activation = 'relu'))
modelx.add(layers.Dense(len(cats), activation = 'softmax'))

modelx.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = modelx.fit_generator(train_generator2,
                              epochs = epoch,
                              validation_data = validation_generator2,
                              verbose = 1,
                              callbacks = [EarlyStopping(monitor='val_accuracy',
                                                        patience = 5,
                                                        restore_best_weights = True)])

test_loss, test_acc = modelx.evaluate_generator(test_generator2, steps = 48)
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print('Xception_test_acc:', test_acc)