import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

from keras.utils.np_utils import to_categorical

np.random.seed(1)

from keras.models import Sequential

from keras.layers import Dense, Activation



from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, InputLayer
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
y_train=train.label

X_train=train.drop(['label'],axis=1)
X_train = X_train / 255.0

test = test / 255.0
# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, 10)

model=Sequential()

model.add(Dense(512,activation='relu',input_shape=(784,)))

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])



history = model.fit(X_train, y_train,

                    batch_size=1,

                    epochs=10,

                    verbose=1)
print("Generating test predictions...")

preds = model.predict_classes(test, verbose=0)
loss = history.history['loss']

accuracy = history.history['accuracy']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training loss')

plt.plot(epochs, accuracy, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "mlp_keras_results.csv")

