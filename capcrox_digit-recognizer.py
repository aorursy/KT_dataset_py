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
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_val = x_train,x_test
y_val = y_train,y_test
x_train,x_test = x_train/255.0,x_test/255.0
x_train.shape
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
r = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=50 )
import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label = 'loss')
plt.plot(r.history['val_loss'],label = 'val_loss')
plt.legend()
plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'],label = 'val_acc')
plt.legend()
print(model.evaluate(x_test,y_test))
predictions  = model.predict_classes(x_test)
y_true = np.argmax(y_test)


correct = np.nonzero(predictions==y_true)[0]
incorrect = np.nonzero(predictions!=y_true)[0]

print(incorrect.shape)
final_predictions = model.predict_classes(x_test)

submit = pd.DataFrame(final_predictions,columns=["Label"])
submit["ImageId"] = pd.Series(range(1,(len(final_predictions)+1)))
submission = submit[["ImageId","Label"]]
submission.shape

submission.to_csv("submission.csv",index=False)