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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
test.head()
sample_submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")

sample_submission.head()
images = train.iloc[:, 1:].values.reshape(train.shape[0],28, 28).astype( 'float32' )/255.0
# images=np.stack((images,)*3, axis=-1).astype('float32')
images=images.reshape(train.shape[0],28, 28,1)

images_test = test.iloc[:, 1:].values.reshape(test.shape[0],28, 28).astype( 'float32' )/255.0
# images_test=np.stack((images_test,)*3, axis=-1).astype('float32')
images_test=images_test.reshape(test.shape[0],28, 28,1)
label = train.iloc[:,0].astype('int').values
ids_test = test.iloc[:,0].values
plt.imshow(images[0].reshape(28,28),cmap='gray')
plt.title(label[0])

images.shape,images_test.shape
plt.imshow(images_test[0].reshape(28,28),cmap = 'gray')
from sklearn.model_selection import train_test_split
images_train, images_validation,label_train,label_validation = train_test_split(images,label,test_size=0.1)
classes = train.iloc[:,0].unique()
classes
label_train
from tensorflow.keras.utils import to_categorical
label_train = to_categorical(label_train)
label_validation = to_categorical(label_validation)
label_train,label_validation
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Conv2D, MaxPooling2D, Flatten,BatchNormalization,Dropout
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D())
model.add(Dropout(0.4))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.summary()
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction,ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
from tensorflow.keras.optimizers import Adam
model.compile(
loss="categorical_crossentropy",
optimizer=Adam(lr=0.001),
metrics=['accuracy'])
model.fit(images_train,label_train, validation_data=(images_validation,label_validation),batch_size=64, epochs=50,callbacks=callbacks)
test_prediction = model.predict(images_test)
test_labels = []
for i in range(len(test_prediction)):
    test_labels.append(np.argmax(test_prediction[i]))
for i in range(len(sample_submission)):
    sample_submission.iloc[i,1]=test_labels[i]
np.argmax(model.predict(images_test[2].reshape(1,28,28,1)))
sample_submission.to_csv("submission.csv",index=False)
