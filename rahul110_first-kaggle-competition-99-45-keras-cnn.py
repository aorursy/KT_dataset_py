# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
import time
start_time = time.time()

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
sns.countplot(df["label"])  # equally distributed (nearly)
y = df["label"].values
print(y.shape)
y = to_categorical(y).astype("uint8")
print(y.shape)
cols = df.columns.tolist()
cols.remove("label")
X = df[cols].values / 255.
print(X.shape)
X = X.reshape((X.shape[0], 28, 28, 1))
X.shape
# visualize images
for label, x in zip(y[:1], X[:1]):
    plt.figure()
    plt.suptitle(str(label))
    plt.imshow(x.reshape(28, 28), cmap='gray')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
del df, X, y
def create_model():
    model = Sequential()
    model.add(Conv2D(32, 5, activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(32, 5, activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, 3, activation="relu", padding='same'))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, 3, activation="relu", padding='same'))
    model.add(Conv2D(128, 3, activation="relu"))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax"))
    
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
model = create_model()
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=2, 
                                            factor=0.4, 
                                            min_lr=3e-6)
early_stops = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, verbose=2, mode='auto')
data_aug = ImageDataGenerator(rotation_range=20, width_shift_range=4, height_shift_range=4, zoom_range=0.1)
# Change the epochs to ~60 to 80 for better results
history = model.fit_generator(data_aug.flow(X_train, y_train, batch_size=128), steps_per_epoch=len(X_train)//128,
                              validation_data=(X_test, y_test), epochs=100, verbose=1, callbacks=[learning_rate_reduction])
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# score = model.test_on_batch(X_test, y_test)
# score
pd.read_csv("../input/sample_submission.csv").head()
pd.read_csv("../input/test.csv").head()
# create output
def make_submission(model, filename="submission.csv"):
    df = pd.read_csv("../input/test.csv")
    X = df.values / 255
    X = X.reshape(X.shape[0], 28, 28, 1)
    preds = model.predict_classes(X)
    subm = pd.DataFrame(data=list(zip(range(1, len(preds) + 1), preds)), columns=["ImageId", "Label"])
    subm.to_csv(filename, index=False)
#     return subm
make_submission(model, "submission.csv")
print(f"Finished in {int(time.time() - start_time)} seconds...")