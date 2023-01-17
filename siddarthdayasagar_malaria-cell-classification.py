import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import os
print(os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images"))

infected_data=os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")
uninfected_data=os.listdir("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")
print(len(infected_data))
print(len(uninfected_data))
model = tf.keras.models.Sequential([
   
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                           target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='training')
validationDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',
                                           target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='validation')
model.summary()
class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs,logs={}):
        if(logs.get('accuracy')>0.98):
            self.model.stop_training=True
callbacks=mycallbacks()
history=model.fit_generator(trainDatagen,validation_data=validationDatagen,epochs=10,verbose=1,callbacks=[callbacks])
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
test_image = image.load_img('../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144823_cell_160.png', target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

print(result)
classes=trainDatagen.class_indices



classes=['Parasitized','Uninfected']


classes[np.argmax(result)]
