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
# Model 2

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)))
model2.add(layers.MaxPooling2D(2,2))
model2.add(layers.Conv2D(64, (3,3), activation='relu'))
model2.add(layers.MaxPooling2D(2,2))
model2.add(layers.Conv2D(128, (3,3), activation='relu'))
model2.add(layers.MaxPooling2D(2,2))
model2.add(layers.Conv2D(128, (3,3), activation='relu'))
model2.add(layers.MaxPooling2D(2,2))
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(5, activation='softmax'))

model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model2_hist = model2.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=20, validation_data=valid_gen, validation_steps=v_steps)
model2.save('flowers_model2.h5')
acc = model2_hist.history['acc']
val_acc = model2_hist.history['val_acc']
loss = model2_hist.history['loss']
val_loss = model2_hist.history['val_loss']

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
# Model 3

model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)))
model3.add(layers.MaxPooling2D(2,2))
model3.add(layers.Conv2D(64, (3,3), activation='relu'))
model3.add(layers.MaxPooling2D(2,2))
model3.add(layers.Conv2D(32, (3,3), activation='relu'))
model3.add(layers.MaxPooling2D(2,2))
model3.add(layers.Conv2D(64, (3,3), activation='relu'))
model3.add(layers.MaxPooling2D(2,2))
model3.add(layers.Flatten())
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(512, activation='relu'))
model3.add(layers.Dense(5, activation='softmax'))

model3.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model3_hist = model3.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=20, validation_data=valid_gen, validation_steps=v_steps)
model3.save('flowers_model3.h5')
acc = model3_hist.history['acc']
val_acc = model3_hist.history['val_acc']
loss = model3_hist.history['loss']
val_loss = model3_hist.history['val_loss']

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
# Model 4

model4 = models.Sequential()
model4.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)))
model4.add(layers.MaxPooling2D(2,2))
model4.add(layers.Conv2D(64, (3,3), activation='relu'))
model4.add(layers.MaxPooling2D(2,2))
model4.add(layers.Conv2D(128, (3,3), activation='relu'))
model4.add(layers.MaxPooling2D(2,2))
model4.add(layers.Conv2D(256, (3,3), activation='relu'))
model4.add(layers.MaxPooling2D(2,2))
model4.add(layers.Flatten())
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(512, activation='relu'))
model4.add(layers.Dense(5, activation='softmax'))

model4.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model4_hist = model2.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=20, validation_data=valid_gen, validation_steps=v_steps)
model4.save('flowers_model4.h5')
acc = model4_hist.history['acc']
val_acc = model4_hist.history['val_acc']
loss = model4_hist.history['loss']
val_loss = model4_hist.history['val_loss']

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
# Model 5

model5 = models.Sequential()
model5.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(img_size,img_size,3)))
model5.add(layers.MaxPooling2D(2,2))
model5.add(layers.Conv2D(64, (3,3), activation='relu'))
model5.add(layers.MaxPooling2D(2,2))
model5.add(layers.Conv2D(96, (3,3), activation='relu'))
model5.add(layers.MaxPooling2D(2,2))
model5.add(layers.Conv2D(96, (3,3), activation='relu'))
model5.add(layers.MaxPooling2D(2,2))
model5.add(layers.Flatten())
model5.add(layers.Dropout(0.5))
model5.add(layers.Dense(512, activation='relu'))
model5.add(layers.Dense(5, activation='softmax'))

model5.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model5_hist = model5.fit_generator(train_gen, steps_per_epoch=t_steps, epochs=30, validation_data=valid_gen, validation_steps=v_steps)
model5.save('flowers_model5.h5')
acc = model5_hist.history['acc']
val_acc = model5_hist.history['val_acc']
loss = model5_hist.history['loss']
val_loss = model5_hist.history['val_loss']

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