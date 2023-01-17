%pylab inline

import time # used for timing the training

import os # operating system commands

import shutil # High level oprations for files

import random # Random generators

import pandas as pd # Pandas data read

import tensorflow as tf # Tensorflow

from tensorflow.keras import layers, models, optimizers # Neural network stuff

from tensorflow.keras.preprocessing.image import ImageDataGenerator # For image processing with keras

from tensorflow.keras.metrics import Accuracy, FalsePositives, FalseNegatives, SensitivityAtSpecificity # Specific metrics

from sklearn.metrics import classification_report, confusion_matrix, roc_curve # For final results analysis

print(f"Tensorflow version: {tf.__version__}")

random.seed(10) # Set seed for the random generators
orig_train_dir = r'/kaggle/input/chest_xray/train'

normal_images=[x for x in os.listdir(os.path.join(orig_train_dir,'NORMAL')) if x.endswith(".jpeg")]

pneumonia_images=[x for x in os.listdir(os.path.join(orig_train_dir,'PNEUMONIA')) if x.endswith(".jpeg")]

print(len(normal_images))

print(len(pneumonia_images))
orig_test_dir = r'/kaggle/input/chest_xray/test'

normal_t_images=[x for x in os.listdir(os.path.join(orig_test_dir,'NORMAL')) if x.endswith(".jpeg")]

pneumonia_t_images=[x for x in os.listdir(os.path.join(orig_test_dir,'PNEUMONIA')) if x.endswith(".jpeg")]

print(len(normal_t_images))

print(len(pneumonia_t_images))
train_dir='./train'

valid_dir='./validation'

test_dir='./test'

all_dirs=[train_dir,valid_dir,test_dir]

try:

    for d in all_dirs:

        os.mkdir(d)

        os.mkdir(os.path.join(d,'NORMAL'))

        os.mkdir(os.path.join(d,'PNEUMONIA'))

except:

    pass
for fname in normal_images[:944]:

    src=os.path.join(orig_train_dir,'NORMAL',fname)

    dst=os.path.join(train_dir,'NORMAL',fname)

    shutil.copyfile(src,dst)

    

for fname in normal_images[945:]:

    src=os.path.join(orig_train_dir,'NORMAL',fname)

    dst=os.path.join(valid_dir,'NORMAL',fname)

    shutil.copyfile(src,dst)

    

for fname in normal_t_images[:234]:

    src=os.path.join(orig_test_dir,'NORMAL',fname)

    dst=os.path.join(test_dir,'NORMAL',fname)

    shutil.copyfile(src,dst)

    

    

for fname in pneumonia_images[:2718]:

    src=os.path.join(orig_train_dir,'PNEUMONIA',fname)

    dst=os.path.join(train_dir,'PNEUMONIA',fname)

    shutil.copyfile(src,dst)

    

for fname in pneumonia_images[2719:]:

    src=os.path.join(orig_train_dir,'PNEUMONIA',fname)

    dst=os.path.join(valid_dir,'PNEUMONIA',fname)

    shutil.copyfile(src,dst)

    

for fname in pneumonia_t_images[:390]:

    src=os.path.join(orig_test_dir,'PNEUMONIA',fname)

    dst=os.path.join(test_dir,'PNEUMONIA',fname)

    shutil.copyfile(src,dst)
tg=ImageDataGenerator(rescale=1./255)

train_generator=tg.flow_from_directory(

    train_dir,

    target_size=(150,150),

    batch_size=128,

    shuffle=True,

    class_mode='binary')

vg=ImageDataGenerator(rescale=1./255)

valid_generator=vg.flow_from_directory(

    valid_dir,

    target_size=(150,150),

    batch_size=128,

    shuffle=True,

    class_mode='binary')

testg=ImageDataGenerator(rescale=1./255)

test_generator=testg.flow_from_directory(

    test_dir,

    target_size=(150,150),

    batch_size=128,

    shuffle=True,

    class_mode='binary')
# Check the indices

print(f" Train image classes: {train_generator.class_indices}")

print(f" Validation image classes: {valid_generator.class_indices}")

print(f" Test image classes: {test_generator.class_indices}")
#Check image data of the generator that correct or not 

for data_batch, labels_batch in train_generator:break

imshow(data_batch[0])

show()

print("Figure 1: The first image from the batch")
#Check labe of the generator that correct or not 

print(f"Check the last labels batch:\n {labels_batch}")
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))

          

model.summary()

print("Model created")
#Comply model

M = ['acc', FalsePositives(), FalseNegatives(), SensitivityAtSpecificity(0.90)] # Define model metrics



model.compile(loss = 'binary_crossentropy',

              optimizer = optimizers.RMSprop(lr = 1e-4),

              metrics = M)

print("Model compiled")
E = 10 # Number of epochs



time_start = time.time() # Start the clock



history = model.fit_generator(

    train_generator,

    steps_per_epoch = 10,

    verbose = 0,

    epochs = E,

    validation_data = valid_generator,

    validation_steps = None)



time_end = time.time()

time_elapsed = time_end - time_start

time_per_epoch = time_elapsed / E

print("Done")

print(f"Time elapsed: {time_elapsed:.0f} seconds")

print(f"Time per epoch: {time_per_epoch:.2f} seconds")



model.save('case_2_run_1.h5') # Save the model

print("Model saved")
accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

false_positives = history.history['false_positives']

val_false_positives = history.history['val_false_positives']

false_negatives = history.history['false_negatives']

val_false_negatives = history.history['val_false_negatives']

sensitivity_at_specificity = history.history['sensitivity_at_specificity']

val_sensitivity_at_specificity = history.history['val_sensitivity_at_specificity']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plot(epochs, accuracy, 'bo-', label='Training accuracy')

plot(epochs, val_accuracy, 'r*-', label='Validation accuracy')

title('Training and validation accuracy')

grid ()

legend()



figure()

plot(epochs, loss, 'bo-', label='Training loss')

plot(epochs, val_loss, 'r*-', label='Validation loss')

title('Training and validation loss')

grid()

legend()



show()
plot(epochs, sensitivity_at_specificity, 'bo-', label='Training sensitivity at specificity')

plot(epochs, val_sensitivity_at_specificity, 'r*-', label='Validation sensitivity at specificity')

title('Training and validation sensitivity at specificity')

grid ()

legend()



show()
tg=ImageDataGenerator(rescale=1./255,rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

train_generator=tg.flow_from_directory(

    train_dir,

    target_size=(150,150),

    batch_size=128,

    shuffle=True,

    class_mode='binary')
M = ['acc', FalsePositives(), FalseNegatives(), SensitivityAtSpecificity(0.90)] # Define model metrics



model.compile(loss = 'binary_crossentropy',

              optimizer = optimizers.RMSprop(lr = 1e-4),

              metrics = M)

print("Model compiled")
E = 10 # Number of epochs



time_start = time.time() # Start the clock



history = model.fit_generator(

    train_generator,

    steps_per_epoch = 10,

    verbose = 0,

    epochs = E,

    validation_data = valid_generator,

    validation_steps = None)



time_end = time.time()

time_elapsed = time_end - time_start

time_per_epoch = time_elapsed / E

print("Done")

print(f"Time elapsed: {time_elapsed:.0f} seconds")

print(f"Time per epoch: {time_per_epoch:.2f} seconds")



model.save('case_2_run_1.h5') # Save the model

print("Model saved")
accuracy = history.history['acc']

val_accuracy = history.history['val_acc']

false_positives = history.history['false_positives_1']

val_false_positives = history.history['val_false_positives_1']

false_negatives = history.history['false_negatives_1']

val_false_negatives = history.history['val_false_negatives_1']

sensitivity_at_specificity = history.history['sensitivity_at_specificity_1']

val_sensitivity_at_specificity = history.history['val_sensitivity_at_specificity_1']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))
plot(epochs, accuracy, 'bo-', label='Training accuracy')

plot(epochs, val_accuracy, 'r*-', label='Validation accuracy')

title('Training and validation accuracy')

grid ()

legend()



figure()

plot(epochs, loss, 'bo-', label='Training loss')

plot(epochs, val_loss, 'r*-', label='Validation loss')

title('Training and validation loss')

grid()

legend()



show()
plot(epochs, sensitivity_at_specificity, 'bo-', label='Training sensitivity at specificity')

plot(epochs, val_sensitivity_at_specificity, 'r*-', label='Validation sensitivity at specificity')

title('Training and validation sensitivity at specificity')

grid ()

legend()



show()
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False,

input_shape=(150,150,3))

conv_base.summary()
for inputs_batch, labels_batch in train_generator:

    break



feature_batch=conv_base.predict(inputs_batch)

feature_batch.shape
model = models.Sequential()



model.add(tf.keras.layers.Flatten(input_shape=(4,4,512)))

model.add(tf.keras.layers.Dense(512, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='softmax'))

          

model.summary()

print("Model created")
M = ['acc', FalsePositives(), FalseNegatives(), SensitivityAtSpecificity(0.90)] # Define model metrics



model.compile(loss = 'binary_crossentropy',

              optimizer = optimizers.RMSprop(lr = 1e-4),

              metrics = M)

print("Model compiled")
E = 10 # Number of epochs



time_start = time.time() # Start the clock





history = model.fit(feature_batch, 

    labels_batch,

    epochs=10,

    verbose=0,

   

    steps_per_epoch=2,)

time_end = time.time()

time_elapsed = time_end - time_start

time_per_epoch = time_elapsed / E

print("Done")

print(f"Time elapsed: {time_elapsed:.0f} seconds")

print(f"Time per epoch: {time_per_epoch:.2f} seconds")



accuracy = history.history['acc']

false_positives = history.history['false_positives_2']

false_negatives = history.history['false_negatives_2']

sensitivity_at_specificity = history.history['sensitivity_at_specificity_2']

loss = history.history['loss']

epochs = range(len(accuracy))
print(accuracy[9])

print(false_positives[9])

print(false_negatives[9])

print(sensitivity_at_specificity[9])

print(loss[9])

epochs