# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import os
import gc
print(os.listdir("../input"))
data_fer = pd.read_csv('../input/fer2013/fer2013.csv')
data_fer.head()
!pip install -U git+https://github.com/qubvel/efficientnet
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image



from efficientnet.keras import center_crop_and_resize, preprocess_input
!pip install -U git+https://github.com/qubvel/efficientnet
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
from efficientnet.keras import center_crop_and_resize, preprocess_input

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout, Conv2D, MaxPool2D, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
BATCH_SIZE=128


# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
idx_to_emotion_fer = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

X_fer_train, y_fer_train = np.rollaxis(data_fer[data_fer.Usage == "Training"][["pixels", "emotion"]].values, -1)
X_fer_train = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_train]).reshape((-1, 48, 48))
y_fer_train = y_fer_train.astype('int8')

X_fer_test_public, y_fer_test_public = np.rollaxis(data_fer[data_fer.Usage == "PublicTest"][["pixels", "emotion"]].values, -1)
X_fer_test_public = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_public]).reshape((-1, 48, 48))
y_fer_test_public = y_fer_test_public.astype('int8')

X_fer_test_private, y_fer_test_private = np.rollaxis(data_fer[data_fer.Usage == "PrivateTest"][["pixels", "emotion"]].values, -1)
X_fer_test_private = np.array([np.fromstring(x, dtype="uint8", sep=" ") for x in X_fer_test_private]).reshape((-1, 48, 48))
y_fer_test_private = y_fer_test_private.astype('int8')

X_train = X_fer_train.reshape((-1, 48, 48, 1))
X_val = X_fer_test_public.reshape((-1, 48, 48, 1))
X_test = X_fer_test_private.reshape((-1, 48, 48, 1))
y_train = to_categorical(y_fer_train,7)
y_val = to_categorical(y_fer_test_public,7)
y_test = to_categorical(y_fer_test_private,7)

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
)

train_datagen.fit(X_train)
val_datagen.fit(X_train)

train_flow = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_flow = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
test_flow = val_datagen.flow(X_test, y_test, batch_size=1, shuffle=False)


print(f"X_fer_train shape: {X_fer_train.shape}; y_fer_train shape: {y_fer_train.shape}")
print(f"X_fer_test_public shape: {X_fer_test_public.shape}; y_fer_test_public shape: {y_fer_test_public.shape}")
print(f"X_fer_test_private shape: {X_fer_test_private.shape}; y_fer_test_private shape: {y_fer_test_private.shape}")
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Concatenate
from keras.utils import to_categorical
import tensorflow as tf
def one_hot(y):
    return to_categorical(y, 7)
from efficientnet.keras import EfficientNetB1 as Net
width = 48
height = 48
dropout_rate = 0.2
#input_shape = (height, width, 1)
input_shape1 = Input(shape=(height,width,1))
input_shape = Concatenate()([input_shape1, input_shape1, input_shape1]) 
conv_base = Net(weights='imagenet', include_top=False,input_shape=(48, 48, 3))
conv_output = conv_base(input_shape)
conv_output_flattened = Flatten()(conv_output)
dense_out = Dense(128, activation='relu')(conv_output_flattened)
out = Dense(7, activation='softmax')(dense_out)


early_stopping = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=20)
checkpoint_loss = ModelCheckpoint('best_loss_weights.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')
checkpoint_acc = ModelCheckpoint('best_accuracy_weights.h5', verbose=1, monitor='val_categorical_accuracy',save_best_only=True, mode='max')
lr_reduce = ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max', factor=0.5, patience=5, min_lr=1e-7, cooldown=1, verbose=1)




model = Model(inputs=input_shape1, outputs=out)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['categorical_accuracy'])
history = model.fit_generator(
        train_flow, 
        steps_per_epoch= X_train.shape[0] // BATCH_SIZE,
        epochs=50, 
        validation_data=val_flow,
        validation_steps = X_val.shape[0] // BATCH_SIZE,
        callbacks=[early_stopping, checkpoint_acc, checkpoint_loss, lr_reduce]
    )

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()





model.load_weights('best_loss_weights.h5', by_name=True)
y_pred = model.predict_generator(test_flow, steps=X_test.shape[0])
y_pred_cat = np.argmax(y_pred, axis=1)
y_true_cat = np.argmax(test_flow.y, axis=1)
report = classification_report(y_true_cat, y_pred_cat)
print(report)

conf = confusion_matrix(y_true_cat, y_pred_cat, normalize="true")

labels = idx_to_emotion_fer.values()
_, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(conf, annot=True, cmap='YlGnBu', 
                 xticklabels=labels, 
                 yticklabels=labels)

plt.show()
# best acc
model.load_weights('best_accuracy_weights.h5')
y_pred = model.predict_generator(test_flow, steps=X_test.shape[0])
y_pred_cat = np.argmax(y_pred, axis=1)
y_true_cat = np.argmax(test_flow.y, axis=1)
report = classification_report(y_true_cat, y_pred_cat)
print(report)

conf = confusion_matrix(y_true_cat, y_pred_cat, normalize="true")

labels = idx_to_emotion_fer.values()
_, ax = plt.subplots(figsize=(8, 6))
ax = sns.heatmap(conf, annot=True, cmap='YlGnBu', 
                 xticklabels=labels, 
                 yticklabels=labels)

plt.show()