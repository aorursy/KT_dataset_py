# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df.head()
X = df.drop("label", axis=1)
y = df["label"].values
X = X/255.0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)
from sklearn.metrics import f1_score, accuracy_score

def model_report(y,y_pred):
    print("Model Report:")
    print("Accuracy: {}".format(accuracy_score(y,y_pred)))
    print("f1: {}".format(f1_score(y,y_pred,average='weighted')))

from tensorflow import keras

X_train_10 = X_train.values.reshape(-1,28,28,1)
y_train_10 = keras.utils.to_categorical(y_train)
X_test_10 = X_test.values.reshape(-1,28,28,1)
y_test_10 = keras.utils.to_categorical(y_test)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.4))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(256,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= keras.optimizers.Adam(),metrics=['accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip=False,  
        vertical_flip=False
)

datagen.fit(X_train_10)
adaptive_learning_rate = keras.callbacks.ReduceLROnPlateau(
    monitor='val_acc',
    patience=3,
    factor=0.3,
    min_lr=0.00001
)

history = model.fit_generator(datagen.flow(X_train_10,y_train_10,batch_size=100),validation_data=(X_test_10,y_test_10),
                              epochs=120,
                             steps_per_epoch=X_train.shape[0]//100,
                             callbacks = [adaptive_learning_rate])

model.evaluate(X_test_10,y_test_10)
X_test.shape
fTest = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
fTest.head()
fTest = fTest/255.0
fTest_10 = fTest.values.reshape(-1,28,28,1)
y_fTest = model.predict_classes(fTest_10)
Submission = y_fTest
ndf = pd.DataFrame(range(1, len(Submission) + 1))
Submission
Submission.reshape(-1,1)
ndf
ndf2 = ndf
Submission2 = Submission
Submission2 = pd.DataFrame(Submission2.reshape(-1,1))
ndf2["Label"] = Submission2
ndf2.columns=["ImageId","Label"]
ndf2.to_csv("sib.csv",index=False)

