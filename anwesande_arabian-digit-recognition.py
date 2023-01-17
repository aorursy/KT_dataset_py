# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras import regularizers
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
y_train=pd.read_csv("/kaggle/input/ahcd1/csvTrainLabel 13440x1.csv")
y_train.head()
X_train=pd.read_csv("/kaggle/input/ahcd1/csvTrainImages 13440x1024.csv")
X_test=pd.read_csv("/kaggle/input/ahcd1/csvTestImages 3360x1024.csv")
y_test=pd.read_csv("/kaggle/input/ahcd1/csvTestLabel 3360x1.csv")
y_train.info()
Y_train=y_train.values
Y_train
X_train.head(10)
X_train.shape
x_train=X_train.values
x_test=X_test.values
Y_test=y_test.values
x_test.shape
x_train_reshaped=x_train.reshape([-1,32,32,1])
x_test_reshaped=x_test.reshape([-1,32,32,1])
import matplotlib.pyplot as plt
rows = 5 # defining no. of rows in figure
cols = 6 # defining no. of colums in figure

f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 

for i in range(rows*cols): 
    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration
    plt.imshow(x_train_reshaped[i].reshape([32,32]),cmap="Blues") 
    plt.axis("off")
    plt.title(str(Y_train[i]), y=-0.15,color="green")
plt.savefig("digits.png")
x_mean=x_train_reshaped.mean()
x_std=x_train_reshaped.std()
x_test_norm=(x_test_reshaped-x_mean)/x_std
x_train_norm=(x_train_reshaped-x_mean)/x_std
x_train_norm
model=  tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape=[32,32,1],activation="relu"),#activity_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)for regularization
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(29,activation="softmax")
])
model.compile(metrics=["accuracy"], loss="sparse_categorical_crossentropy",optimizer="adam")
model.fit(x_train_norm,Y_train,epochs=15)
score = model.evaluate(x_test_norm, Y_test)
print('Test accuarcy: %0.2f%%' % (score[1] * 100))