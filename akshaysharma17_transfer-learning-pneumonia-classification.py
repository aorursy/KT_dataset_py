import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import os
print(os.listdir("../input"))
#Libraries imported
from tensorflow import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Input, Lambda, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix

train_dir = '../input/chest-xray-pneumonia/chest_xray/train/'
test_dir = '../input/chest-xray-pneumonia/chest_xray/test/'
val_dir = '../input/chest-xray-pneumonia/chest_xray/val/'
train_normal = train_dir + 'NORMAL/'
train_pneumonia = train_dir + 'PNEUMONIA/'
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
#learning rate reduced when val_acc or loss is becomes constant
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=1, verbose=1)
# normal pic
rand_norm = np.random.randint(0, len(os.listdir(train_normal)))
normal_pic = os.listdir(train_normal)[rand_norm]
normal_pic_address = train_normal+normal_pic

# pneumonia pic
rand_norm = np.random.randint(0, len(os.listdir(train_pneumonia)))
pneumonia_pic = os.listdir(train_pneumonia)[rand_norm]
pneumonia_pic_address = train_pneumonia+pneumonia_pic

# load the images
normal_load = Image.open(normal_pic_address)
pneumonia_load = Image.open(pneumonia_pic_address)

# plot
f = plt.figure(figsize= (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(normal_load, cmap='gray')
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(pneumonia_load, cmap='gray')
a2.set_title('Pneumonia')
vgg = VGG16(input_shape=[150, 150, 3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
#added a few layers after this 
flatten1 = Flatten()(vgg.output)
dense1 = Dense(256, activation="relu")(flatten1)
dropout1 = Dropout(0.5)(dense1)
prediction = Dense(2, activation="softmax")(dropout1)

model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
#model compilation using adam optimizer
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# loading the images
train_datagen = ImageDataGenerator(rescale=1./255,
                            rotation_range=30,
                            zoom_range=0.15,
                            horizontal_flip=True,
                            fill_mode="nearest")


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                    target_size=(150, 150),
                                    color_mode='rgb',
                                    batch_size=32,
                                    class_mode='categorical',
                                    shuffle=True)

validation_set = test_datagen.flow_from_directory(val_dir,
                                    target_size=(150, 150),
                                    color_mode='rgb',
                                    batch_size=32,
                                    class_mode='categorical',
                                    shuffle=True)

test_set = test_datagen.flow_from_directory(test_dir,
                                    target_size=(150, 150),
                                    color_mode='rgb',
                                    batch_size = 32,
                                    class_mode='categorical')
#early stopping used if the validation loss does not change for long
early_stop = EarlyStopping(monitor='val_loss',patience=3)

#fitting the model with training set and checking accuracy on validation set simultaneously
history = model.fit_generator(training_set,
                               steps_per_epoch=128,
                               epochs=6,
                               validation_data=validation_set,
                               validation_steps=len(validation_set))
# plot the loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
model.save('pneumonia_model.h5')
img=image.load_img('../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0005-0001.jpeg',
                   target_size=(150,150))
x=image.img_to_array(img)
x
x=x/255
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape
pred = np.argmax(model.predict(img_data), axis=1)
pred
if(pred==1):
    print("Uninfected")
else:
    print("Infected")
