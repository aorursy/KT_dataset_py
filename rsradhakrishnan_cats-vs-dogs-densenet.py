from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import  Conv2D ,MaxPool2D
from keras.layers import Activation,Dropout,Flatten, Dense
from keras import backend as k
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
import pandas as pd
import tensorflow as tf
import time
train_dir = "train/"
test_dir= "test1/"

IMG_SIZE = (150, 150)


labels = []
img_path = []
path= '/kaggle/working/train'
for img in os.listdir(train_dir):
    img_path.append(os.path.join(path,img))
    
    if img.startswith("cat"):
        labels.append("cat")
    elif img.startswith("dog"):
        labels.append("dog")
df = pd.DataFrame({
    'image' : img_path,
    'class' : labels
})
df['class'].value_counts().plot.bar()
df.head()
train_datagen = ImageDataGenerator(validation_split = 0.25, rescale = 1. / 255)
train_generator = train_datagen.flow_from_dataframe(dataframe = df, 
                                                    x_col = 'image', 
                                                    y_col = 'class', 
                                                    batch_size = 64, 
                                                    seed = 11, 
                                                    class_mode = 'categorical', 
                                                    target_size = IMG_SIZE,
                                                    shuffle = True,
                                                    subset="training"
                                                   )

val_generator = train_datagen.flow_from_dataframe(dataframe = df, 
                                                    x_col = 'image', 
                                                    y_col = 'class', 
                                                    batch_size = 64, 
                                                    seed = 11, 
                                                    class_mode = 'categorical', 
                                                    target_size = IMG_SIZE,
                                                    shuffle = True,
                                                    subset="validation"
                                                   )

train_steps = train_generator.n // train_generator.batch_size
validation_steps = val_generator.n // val_generator.batch_size

from keras.applications import DenseNet201

base_model = DenseNet201(include_top=False, weights="imagenet",  input_shape=(150,150,3), pooling='avg' )


for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
base_model,

Dense(512, activation='relu'),
Dropout(0.25),
BatchNormalization(),

Dense(512, activation='relu'),
Dropout(0.25),
BatchNormalization(),


    
Dense(2, activation='softmax')])


model.summary()
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(optimizer,loss='binary_crossentropy',metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss',mode='min', verbose=1,patience=10)

lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3,verbose=1, factor=0.25, min_lr=0.0001)

epochs = 10
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch= train_steps, epochs=epochs, callbacks = [es, lr],
                    validation_data = val_generator,validation_steps = validation_steps, max_queue_size=100, use_multiprocessing=True)
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(train_acc))


plt.plot(epochs, train_acc, label = 'Train Accuracy')
plt.plot(epochs, val_acc, label = 'validation Accuracy')
plt.legend(loc=0)

plt.show()
train_loss = history.history['loss']
val_loss = history.history['val_loss']




plt.plot(epochs, train_loss, label = 'Train loss')
plt.plot(epochs, val_loss, label = 'validation loss')
plt.legend(loc=0)
plt.show()
test_img_path = []

test_path = '/kaggle/working/test1'

for img in os.listdir(test_path):
    test_img_path.append(os.path.join(test_path, img))
    
df_test = pd.DataFrame({'image_path': test_img_path})


test_datagen = ImageDataGenerator(rescale = 1./ 255)
test_generator = test_datagen.flow_from_dataframe(dataframe = df_test,
                                                 x_col = 'image_path',
                                                 y_col = None,
                                                 batch_size = 50,
                                                 seed = 11,
                                                 target_size = IMG_SIZE,
                                                   class_mode=None,)





test_steps = test_generator.n // test_generator.batch_size

predictions = model.predict(test_generator,steps = test_steps, verbose= 1)
print(type(predictions))
print(predictions[:10])
predictions = np.argmax(predictions,axis=1)
print(len(predictions))
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (10,10))
fig.suptitle("predictions", fontsize = 16)

for i, img in enumerate(df_test.image_path[:10]):
  plt.subplot(5, 5, i + 1)
  img = mpimg.imread(img)
  plt.imshow(img)
  plt.title(str(predictions[i]))
  plt.xticks([])
  plt.yticks([])


df_test["label"]=predictions
df_test.to_csv("submission.csv",index=False)
