import tensorflow as tf
print(tf.__version__)
import os
base_dir = '/kaggle/input/rockpaperscissors/rps-cv-images'
print(os.listdir(base_dir))

rock_dir = os.path.join(base_dir,'rock')
paper_dir = os.path.join(base_dir,'paper')
scissors_dir = os.path.join(base_dir,'scissors')
print("Rock : ",len(os.listdir(rock_dir)))
print("Paper : ",len(os.listdir(paper_dir)))
print("Scissors : ",len(os.listdir(scissors_dir)))
# define data augmentation configuration
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size=32
img_rows,img_cols=120,120
num_classes=3

datagen = ImageDataGenerator(
        rescale=1/255.0,
        zoom_range=0.25,
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.4)

# setup generator
train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(img_rows,img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        subset='training')
validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(img_rows,img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        subset='validation')

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Create a Sequential model by passing a list of layers to the Sequential constructor
from tensorflow.keras.layers import Dropout
img_rows,img_cols=120,120
model = Sequential([Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', 
                           input_shape=(img_rows,img_cols,3)),
                    MaxPooling2D(pool_size=(2,2)),
                    Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Dropout(rate=0.25),
                    Conv2D(filters=64, kernel_size=(3,3), padding='same',activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Conv2D(filters=128, kernel_size=(3,3), padding='same',activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Dropout(rate=0.25),
                    Flatten(),
                    Dropout(rate=0.5),
                    Dense(units=512, activation='relu'),
                    Dense(units=num_classes, activation='softmax')])
print(model.summary())
accuracythreshold=96e-2

class AccCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') >= accuracythreshold):   
          print("\nReached %2.2f%% accuracy, stop training!" %(accuracythreshold*100))   
          self.model.stop_training = True
# Compile the Model
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

nb_train_samples=1314
nb_validation_samples=874
epochs=55

history=model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[AccCallback()],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

from matplotlib.pyplot import figure,subplot,plot,legend,title,show
figure(figsize=(8,8))
subplot(1, 2, 1)
plot(epochs, acc, 'r', label='Training Accuracy')
plot(epochs, val_acc, 'b', label='Validation Accuracy')
legend(loc='lower right')
title('Training and Validation Accuracy')
show

subplot(1, 2, 2)
plot(epochs, loss, 'r', label='Training Loss')
plot(epochs, val_loss, 'b', label='Validation Loss')
legend(loc='upper right')
title('Training and Validation Loss')
show