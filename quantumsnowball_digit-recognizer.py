import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
y_train = train.label

X_train = train.drop(labels='label', axis=1)

X_test = test
X_train, X_test = X_train.values.reshape(42000,28,28,1), X_test.values.reshape(28000,28,28,1)
y_train = to_categorical(y_train)
X_train.shape, y_train.shape, X_test.shape
X_train = X_train/255

X_test = X_test/255
plt.imshow(X_train[0][:,:,0], cmap='copper')

plt.colorbar()
plt.imshow(y_train[:5], cmap='copper')

plt.colorbar()
input_shape = X_train.shape[1:]

num_classes = y_train.shape[1]
model = Sequential([

    Conv2D(128, kernel_size=(3, 3), activation='relu', input_shape=input_shape),

    Conv2D(256, kernel_size=(3, 3), activation='relu'),

    Conv2D(512, kernel_size=(3, 3), activation='relu'),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(256, activation='relu'),

    Dense(128, activation='relu'),

    Dense(num_classes, activation='softmax'),

])

model.compile(loss=tf.keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])
# if os.path.isfile('model.h5'):

#     model = tf.keras.models.load_model('model.h5')

#     print('Load model successfully')
model.summary()
model.fit(X_train, y_train,

          batch_size=128,

          epochs=20,

          validation_split = 0.2)
#model.save('model.h5')
submission = pd.DataFrame({'ImageId':range(1,len(X_test)+1), 'Label':model.predict(X_test).argmax(axis=1)})

submission.to_csv('submission.csv', index=False)