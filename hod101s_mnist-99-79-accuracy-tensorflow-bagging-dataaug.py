import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D , MaxPooling2D , Dense, Flatten, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical

import gc
from tqdm import tqdm
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = np.concatenate((x_train,x_test)).reshape((-1,28,28,1))
y = np.concatenate((y_train,y_test))
y = to_categorical(y)
y[0]
BATCH_SIZE = 16
datagen = ImageDataGenerator(
    rotation_range=10, 
    width_shift_range=2.0,
    height_shift_range=0.5, 
    shear_range=1.0, 
    validation_split = 0.1
)
trainGen = datagen.flow(X,
                   y,
                   subset = 'training',
                    shuffle=True,
                    
    batch_size = BATCH_SIZE,
)

valGen = datagen.flow(X,
                   y,
                   subset = 'validation'
)
model = Sequential()

# NET1
# model.add(Conv2D(32,(4,4),activation = tf.nn.leaky_relu, input_shape=(28,28,1)))
# model.add(BatchNormalization())

# model.add(Conv2D(32,(4,4),activation = tf.nn.leaky_relu))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())

# model.add(Conv2D(64,(3,3),activation = tf.nn.leaky_relu))
# model.add(BatchNormalization())

# model.add(Conv2D(64,(3,3),activation = tf.nn.leaky_relu))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())

# model.add(Conv2D(128,(2,2),activation = tf.nn.leaky_relu))
# model.add(BatchNormalization())

# model.add(Conv2D(256,(2,2),activation = tf.nn.leaky_relu))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())


# NET2
model.add(Conv2D(32,(2,2),activation = tf.nn.leaky_relu, input_shape=(28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(32,(2,2),activation = tf.nn.leaky_relu))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64,(2,2),activation = tf.nn.leaky_relu))
model.add(BatchNormalization())

model.add(Conv2D(64,(2,2),activation = tf.nn.leaky_relu))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(128,(2,2),activation = tf.nn.leaky_relu))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256,activation= tf.nn.leaky_relu))
model.add(Dropout(0.2))
model.add(Dense(64,activation= tf.nn.leaky_relu))
model.add(Dropout(0.1))
model.add(Dense(10,activation = 'softmax'))
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
hist = model.fit(
    trainGen,
    validation_data = valGen,
    epochs = 50,
    batch_size = BATCH_SIZE,
    shuffle = True,
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor = 0.5 ,patience = 3)]
)
gc.collect()
with open('model.json','w') as f:
    f.write(model.to_json())
model.save_weights('weights.h5')
mod2 = model
!pip install kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle competitions download -c digit-recognizer 
import zipfile
with zipfile.ZipFile("/content/test.csv.zip","r") as zip_ref:
    zip_ref.extractall("/content/")
test = pd.read_csv('/content/test.csv')
sample = pd.read_csv('/content/sample_submission.csv')
test_data = test.to_numpy().reshape((-1,28,28,1))
preds = np.zeros(test.shape[0])
i = 0
for x in tqdm(test_data):
    img = x.reshape((1,28,28,1))
    pred = (model.predict(img) + mod2.predict(img))/2
    preds[i] = np.argmax(pred)
    i+=1
gc.collect()
results = pd.DataFrame()
results['ImageId'] = sample['ImageId']
results['Label'] = preds.astype(int)
results.head()
results.to_csv('res.csv',index=False)
! kaggle competitions submit -c digit-recognizer -f "res.csv" -m "TF v2.2"