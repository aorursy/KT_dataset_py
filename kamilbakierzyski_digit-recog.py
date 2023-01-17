import numpy as np
import pandas as pd
import tensorflow
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

y_train = np.array(df_train['label'])
df_train.drop('label', axis=1, inplace=True)

x_train = np.expand_dims(df_train, axis=-1)
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28))
x_train = np.expand_dims(x_train, axis=-1)

x_test = np.expand_dims(df_test, axis=-1)
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28))
x_test = np.expand_dims(x_test, axis=-1)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,stratify=y_train,test_size=0.1)
print(f"Train: {x_train.shape}, Labels: {y_train.shape}, Val: {x_val.shape} Test: {x_test.shape}")
y_train = tensorflow.keras.utils.to_categorical(y_train,10)
y_val = tensorflow.keras.utils.to_categorical(y_val,10)
print(x_train.shape)
print(y_train.shape)
hard_list = [2617, 4748, 5276, 9545, 10950, 11272, 14512, 16490, 17244, 19083,
             20043, 20241, 20509, 22009, 22565, 24984, 27336, 27716, 3277, 3279,
             6979, 7461, 8458, 8465, 10434, 14459, 14798, 14992, 15656, 16452,
             17946, 18107, 22823, 24015, 25715, 27352, 27937, 6117, 18649, 11539,
             15047, 15158, 8119, 19542, 20153, 21657, 22766, 24767, 27799, 645]
hard_list_answ = [6, 7, 1, 1, 2, 1, 1, 1, 1, 7,
                 7, 1, 5, 1, 6, 4, 0, 8, 9, 4,
                 9, 4, 9, 6, 9, 4, 9, 9, 4, 9,
                 9, 4, 9, 4, 4, 9, 9, 6, 6, 0,
                 5, 3, 9, 8, 9, 5, 8, 5, 7, 2]

import matplotlib.pyplot as plt
%matplotlib inline
# preview the images first
plt.figure(figsize=(12,10))
x, y = 10, 5
counter = 0
for index, answer in zip(hard_list, hard_list_answ):  
    plt.subplot(y, x, counter+1)
    plt.imshow(x_test[index].reshape((28,28)), cmap='gray')
    plt.title(answer)
    counter += 1
plt.show()
x_train = tensorflow.keras.utils.normalize(x_train, axis=-1)
x_test = tensorflow.keras.utils.normalize(x_test, axis=-1)
x_val = tensorflow.keras.utils.normalize(x_val, axis=-1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1,
                                 rotation_range=15,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.1,
                                 horizontal_flip=False,
                                 vertical_flip=False)
test_datagen = ImageDataGenerator(rescale=1)

train_generator = train_datagen.flow(x_train, y_train,
                                    batch_size=64)

validation_generator = test_datagen.flow(x_val, y_val,
                                        batch_size=64)
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
# model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.25))

# model.add(Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
# model.add(Dropout(0.4))
# model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

mcp = ModelCheckpoint("/kaggle/working/best_model.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0001)

import time

# !rm -R ./logs/ # rf
log_dir="logs/fit/{}-{}".format('CNN-Digit-recog-aug', time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
history_aug = model.fit_generator(train_generator,
                                  steps_per_epoch= (64 // x_train.shape[0]),
                                  epochs=50,
                                  validation_data=validation_generator,
                                  validation_steps=(64 // x_val.shape[0]),
                                  callbacks=[mcp, es, learning_rate_reduction, tensorboard])
import matplotlib.pyplot as plt

acc = history_aug.history['accuracy']
val_acc = history_aug.history['val_accuracy']
val_loss = history_aug.history['val_loss']
epochs = range(1, len(acc) + 1)
loss = history_aug.history['loss']

plt.plot(epochs, loss, 'b', label='Training loss', color='red')
plt.plot(epochs, val_loss, 'b', label='Validation loss', color='blue')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, acc, 'b', label='Training accuracy', color='red')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy', color='blue')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model
best_model = load_model('/kaggle/working/best_model.hdf5')
results=best_model.predict_classes(x_test)
print(results)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
#submission.to_csv("F:\\PYTHON PROGRAM\\JaiShreeRammnist11.csv",index=False)
submission.to_csv("submission.csv",index=False,header=True)
df_check = pd.read_csv('/kaggle/working/submission.csv')
df_check