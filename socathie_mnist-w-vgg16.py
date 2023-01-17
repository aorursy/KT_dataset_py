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
import matplotlib.pyplot as plt

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Softmax, Activation, Lambda, Concatenate, Dense, Flatten

from keras.utils import to_categorical

from keras.regularizers import l2

from keras import Model, Input

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model

import keras.backend as K

from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train
test
X = [train.iloc[i,1:].values for i in range(len(train))]

X = [x.reshape(28,28) for x in X]

X = [np.pad(x, 2) for x in X]

X = np.array(X)

X = X.reshape(X.shape[0],X.shape[1], X.shape[2],1)

X = np.repeat(X, 3, axis=-1)

X.shape
plt.imshow(X[0,:,:,0])

plt.show()
X_test = [test.iloc[i,:].values for i in range(len(test))]

X_test = [x.reshape(28,28) for x in X_test]

X_test = [np.pad(x, 2) for x in X_test]

X_test = np.array(X_test)

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2],1)

X_test = np.repeat(X_test, 3, axis=-1)

X_test.shape
plt.imshow(X_test[0,:,:,0])

plt.show()
n_classes = 10

y = [train.iloc[i,0] for i in range(len(train))]

y = np.array(y)

print(np.unique(y, return_counts=True))

y = to_categorical(y, num_classes=n_classes)

y.shape
val_rate = 0.1
gen = ImageDataGenerator(rotation_range=30, width_shift_range=2, height_shift_range=2, 

                         shear_range=30, validation_split=val_rate)

train_gen = gen.flow(X, y, subset="training", batch_size=32)

val_gen = gen.flow(X, y, subset="validation", batch_size=128)
vgg  = VGG16(include_top=False, weights=None, input_shape=(32,32,3), pooling="max")

vgg.summary()
x = vgg.layers[-1].output

x = Dense(10, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg.layers[0].output,outputs=x)

model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
# training parameters

epochs = 100 # maximum number of epochs

train_steps = int(42000*(1-val_rate))//32

val_steps = int(42000*val_rate)//128
early_stopping = EarlyStopping(patience=10, verbose=1)

model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)



history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=epochs,

                              validation_data=val_gen, validation_steps=val_steps, callbacks=[model_checkpoint])



model.save("model.h5")
best_model = load_model("best_model.h5")
print("Evaluating performance on all training data...")

results = best_model.evaluate(X, y, batch_size = 128)

print("train loss, train acc:", results)
print("Predicting on all available data...")

y_pred_one_hot = best_model.predict(X_test, verbose=1, batch_size=128)
y_pred = np.argmax(y_pred_one_hot, axis=-1)

print(y_pred.shape)
sub = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

sub
for i in range(len(y_pred)):

    sub.iloc[i].Label = y_pred[i]
sub.to_csv("submission.csv", index=False)
sub