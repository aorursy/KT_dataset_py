import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import os

train_dir='../input/intel-image-classification/seg_train/seg_train'
test_dir='../input/intel-image-classification/seg_test/seg_test'

print(len(train_dir))
print(len(test_dir))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1.0/255.0)

trainDatagen=datagen.flow_from_directory(train_dir,
                                        target_size=(150,150),
                                         batch_size=40,
                                         class_mode='categorical')
valDatagen=datagen.flow_from_directory(test_dir,
                                      target_size=(150,150),
                                       batch_size=40,
                                       class_mode='categorical')

model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6,activation='softmax')])
model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs,logs={}):
        if(logs.get('accuracy')>0.99):
            self.model.stop_training=True
callback=mycallbacks()
history=model.fit_generator(trainDatagen,validation_data=valDatagen,epochs=30,verbose=1,callbacks=[callback])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from keras.preprocessing import image
test_image = image.load_img('../input/intel-image-classification/seg_pred/seg_pred/10021.jpg', target_size = (150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

print(result)
classes=trainDatagen.class_indices



classes=['buildings','forest','glacier','mountain','sea','street']


classes[np.argmax(result)]
