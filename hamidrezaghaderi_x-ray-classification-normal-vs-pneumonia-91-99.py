import os
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import pandas as pd
from sklearn.utils import class_weight

%matplotlib inline
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.95):
            self.model.stop_training = True
            print('\n The model reached to 95% accuracy on the training set.')
EPOCHS = 10
BATCH_SIZE = 16
INPUT_SHAPE=[200, 200, 3]
IMAGE_SIZE = [200, 200]

# Define the bas edirectory of the dataset
base_dir =  os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray')
train_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/')
validation_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/val/')
test_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/')


# Training normal pictures directory
train_normal_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/')

# Training pneumonia pictures directory
train_pneumonia_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')

# Validation normal pictures directory
validation_normal_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/')

# Validation pneumonia pictures directory
validation_pneumonia_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/')

# Test normal pictures directory
test_normal_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/')

# Test pneumonia pictures directory
test_pneumonia_dir = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')


# Get files' name
train_normal_names = os.listdir(train_normal_dir)
train_pneumonia_names = os.listdir(train_pneumonia_dir)
val_normal_names = os.listdir(validation_normal_dir)
val_pneumonia_names = os.listdir(validation_pneumonia_dir)
test_normal_names = os.listdir(test_normal_dir)
test_pneumonia_names = os.listdir(test_pneumonia_dir)

# Show the content of the base directory
print('Dataset folder content:')
print(os.listdir(base_dir))
print('')

# Print Dataset size
print('Train Normal Images:',  len(os.listdir(train_normal_dir)))
print('Train Pneumonia Images:',  len(os.listdir(train_pneumonia_dir)))
print('Total Train Images:',  len(os.listdir(train_normal_dir)) + len(os.listdir(train_pneumonia_dir)))
print('')

print('Validation Normal Images:',  len(os.listdir(validation_normal_dir)))
print('Validation Pneumonia Images:',  len(os.listdir(validation_pneumonia_dir)))
print('Total Validation Images:',  len(os.listdir(validation_normal_dir)) + len(os.listdir(validation_pneumonia_dir)))
print('')

print('Test Normal Images:',  len(os.listdir(test_normal_dir)))
print('Test Pneumonia Images:',  len(os.listdir(test_pneumonia_dir)))
print('Total test Images:',  len(os.listdir(test_normal_dir)) + len(os.listdir(test_pneumonia_dir)))
print('')
# 

train_list = []
test_list = []
val_list = []

for x in train_normal_names:
    train_list.append([x, 'Normal'])
    
for x in train_pneumonia_names:
    train_list.append([x, 'Pneumonia'])
    
for x in val_normal_names:
    val_list.append([x, 'Normal'])
    
for x in val_normal_names:
    val_list.append([x, 'Pneumonia'])
    
for x in test_normal_names:
    test_list.append([x, 'Normal'])
    
for x in test_pneumonia_names:
    test_list.append([x, 'Pneumonia'])


# Creates train, validation, and test dataframes
train_df = pd.DataFrame(train_list, columns=['image', 'label'])
val_df = pd.DataFrame(val_list, columns=['image', 'label'])
test_df = pd.DataFrame(test_list, columns=['image', 'label'])

# Plot each set
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.countplot(train_df['label'])
plt.title('Train data')

plt.subplot(1,3,2)
sns.countplot(val_df['label'])
plt.title('Validation data')

plt.subplot(1,3,3)
sns.countplot(test_df['label'])
plt.title('Test data')

plt.show()
# Display some of the training images
fig = plt.gcf()
fig.set_size_inches(16, 16)

next_normal_pix = [os.path.join(train_normal_dir, fname) 
                for fname in train_normal_names[0:8]]
next_pneumonia_pix = [os.path.join(train_pneumonia_dir, fname) 
                for fname in train_pneumonia_names[0:8]]

for i, img_path in enumerate(next_normal_pix+next_pneumonia_pix):
  sp = plt.subplot(4, 4, i + 1)
  sp.axis('off') 

  imgage = mpimg.imread(img_path)
  plt.imshow(imgage, cmap='gray')

plt.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu', input_shape=INPUT_SHAPE),
    #tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(3,3),
    
    # CNN Layer 1
    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    #tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(3,3),

    # CNN Layer 2
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    #tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # CNN Layer 3
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    #tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # CNN Layer 4
    tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    #tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),


    # Layer 6
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.add(tf.keras.layers.Dropout(0.1))

model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])
base_model = tf.keras.applications.InceptionV3(input_shape=INPUT_SHAPE,include_top=False, weights='imagenet')

for layers in base_model.layers[:200]:
    layers.trainable = False

model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid') 
        ])


model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['acc'])

model.summary()
# Using data augmentation to expand the dataset size 
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   fill_mode='nearest',
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='binary', 
                                                    target_size=IMAGE_SIZE,
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(train_dir, 
                                                         target_size=IMAGE_SIZE,
                                                         batch_size=BATCH_SIZE,
                                                         class_mode='binary',
                                                         subset='validation') 
callbacks = myCallback()

cw = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)

class_weights = {0: cw[0], 1: cw[1]}

print(class_weights)

history = model.fit_generator(train_generator,
                              class_weight=class_weights,
                              epochs=EPOCHS,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks=[callbacks])
test_datagen = ImageDataGenerator(rescale=1.0/255)


test_generator = validation_datagen.flow_from_directory(test_dir, 
                                                         batch_size=BATCH_SIZE, 
                                                         class_mode='binary', 
                                                         target_size=IMAGE_SIZE)

# evaluate the model by Test data
scores = model.evaluate_generator(test_generator)

print("\n Test accuracy: %.2f%%" % (scores[1]*100))
# Plot Model Accuracy and Loss
training_acc=history.history['acc']
validation_acc=history.history['val_acc']
training_loss=history.history['loss']
validation_loss=history.history['val_loss']

epochs=range(len(training_acc)) # Get number of epochs

# Plot accuracy of training and validation per epoch
plt.plot(epochs, training_acc, 'r', "Training Accuracy")
plt.plot(epochs, validation_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

# Plot loss of training and validation per epoch
plt.plot(epochs, training_loss, 'r', "Training Loss")
plt.plot(epochs, validation_loss, 'b', "Validation Loss")