import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
import tensorflow.keras.applications as app
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import re

%matplotlib inline
warnings.filterwarnings('ignore')
train = glob.glob('../input/100-bird-species/train/*')
test = glob.glob('../input/100-bird-species/test/*')
valid = glob.glob('../input/100-bird-species/valid/*')
tr_total = ts_total = vl_total = 0
for tr, ts, vl in zip(train, test, valid):
    tr_total += len(glob.glob(tr + '/*'))
    ts_total += len(glob.glob(ts + '/*'))
    vl_total += len(glob.glob(vl + '/*'))

print("Train Total = %d\nTest total = %d\nValid Total = %d" %(tr_total, ts_total, vl_total))
birds_list = sorted([re.findall('/train/(.+)', train[i])[0] for i in range(len(train))])
n_classes = len(train)
batch_size = 64
input_shape = (224, 224, 3)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory('../input/100-bird-species/train/', target_size=input_shape[0:2], class_mode='categorical', batch_size=batch_size, shuffle=True)
val_data = datagen.flow_from_directory('../input/100-bird-species/valid/', target_size=input_shape[0:2], class_mode='categorical', batch_size=batch_size, shuffle=True)
test_data = datagen.flow_from_directory('../input/100-bird-species/test/', target_size=input_shape[0:2], class_mode='categorical', batch_size=batch_size, shuffle=True)
index = 16
print('y = ', birds_list[np.argmax(val_data[0][1][index])])
plt.imshow(val_data[0][0][index])
xp = app.xception.Xception(include_top=False, weights='imagenet', input_shape=input_shape)
xp.trainable = False
model = Sequential()

model.add(xp)

model.add(Flatten())

model.add(Dense(units=1098, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=len(train), activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.000005), loss=CategoricalCrossentropy(), metrics=['accuracy'])
history = model.fit(train_data, epochs=30, steps_per_epoch=len(train_data), verbose=1, validation_data=val_data, validation_steps=len(val_data), use_multiprocessing=True, workers=10)
plt.figure(figsize=(6, 4))
plt.plot([i for i in range(len(history.history['accuracy']))], history.history['accuracy'], label='Train Acc.')
plt.plot([i for i in range(len(history.history['val_accuracy']))], history.history['val_accuracy'], label='Val Acc.')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.figure(figsize=(6, 4))
sns.lineplot([i for i in range(len(history.history['loss']))], history.history['loss'], label='Train loss')
sns.lineplot([i for i in range(len(history.history['val_loss']))], history.history['val_loss'], label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
model.save_weights('weights.h5')
model.evaluate(test_data)