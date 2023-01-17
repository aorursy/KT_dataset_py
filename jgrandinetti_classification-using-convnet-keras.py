import numpy as np
from keras import layers
from keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Split images into Training and Validation (20%)

train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

img_size = 128
batch_size = 20
t_steps = 3462/batch_size
v_steps = 861/batch_size

train_gen = train.flow_from_directory("../input/flowers/flowers", target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory("../input/flowers/flowers/", target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')
# Model

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=20, validation_data=valid_gen, validation_steps=v_steps)
model.save('flowers_model.h5')
acc = model_hist.history['acc']
val_acc = model_hist.history['val_acc']
loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15, 6));
plt.subplot(1,2,1)
plt.plot(epochs, acc, color='#0984e3',marker='o',linestyle='none',label='Training Accuracy')
plt.plot(epochs, val_acc, color='#0984e3',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss, color='#eb4d4b', marker='o',linestyle='none',label='Training Loss')
plt.plot(epochs, val_loss, color='#eb4d4b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()