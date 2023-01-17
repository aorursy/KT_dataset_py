# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
df_subm = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
print(df_train.shape)
print(df_test.shape)
train_labels = df_train["label"] 
df_train.drop(["label"], axis = 1, inplace = True)
df_train = df_train.values.reshape(-1,28,28,1)
df_test = df_test.values.reshape(-1,28,28,1)
print(df_train.shape)
print(df_test.shape)
# Change the value of i to check different labels
i = 1001
print("Image label is: ", train_labels[i])
plt.imshow(df_train[i][:,:,0])
# Normalizing Pixels
df_train = df_train/255.0
df_train = df_train/255.0
# Check distribution of differnet labels across Training Data
sns.countplot(train_labels)
X_train, X_test, y_train, y_test = train_test_split(df_train, train_labels, test_size = 0.05, random_state = 1)
# model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (28,28,1))
#                             ,tf.keras.layers.MaxPooling2D(2,2)
#                             ,tf.keras.layers.Conv2D(64, (3,3), activation = "relu")
#                             ,tf.keras.layers.MaxPooling2D(2,2)
#                             ,tf.keras.layers.Conv2D(64,(3,3), activation = "relu")
#                             ,tf.keras.layers.MaxPooling2D(2,2)
#                             ,tf.keras.layers.Flatten()
#                             ,tf.keras.layers.Dense(150 , activation = "relu")
#                             ,tf.keras.layers.Dense(10, activation = "softmax")])
# model.compile(optimizer = RMSprop(lr = 0.001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (28,28,1))
                            ,tf.keras.layers.MaxPooling2D(2,2)
                            ,tf.keras.layers.Conv2D(64, (3,3), activation = "relu")
                            ,tf.keras.layers.MaxPooling2D(2,2)
                            ,tf.keras.layers.Conv2D(64, (3,3), activation = "relu")
                            ,tf.keras.layers.MaxPooling2D(2,2)
                            ,tf.keras.layers.Flatten()
                            ,tf.keras.layers.Dense(150, activation = "relu")
                            ,tf.keras.layers.Dense(10, activation = "softmax")])
model.compile(optimizer = RMSprop(lr = 0.001), loss = "sparse_categorical_crossentropy",  metrics = ["accuracy"])
model.summary()
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 20)
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.legend()
plt.plot()
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.plot(history.history["loss"], label = "Training loss")
plt.legend()
plt.plot()
y_preds = np.argmax(model.predict(df_test), axis = 1)
# Checking the predictions on Test Data, Change the value of i to see different predictions  
i = 100
print("Image label is: ", y_preds[i])
plt.imshow(df_test[i][:,:,0])
df_subm.tail()
submission = pd.DataFrame({"ImageId" : range(1, 28001), "Label" : y_preds})
submission.to_csv("submission.csv", index = False)