# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
label = train_data["label"]
train_data = train_data.drop("label", axis = 1)
label.value_counts()
train_data.isnull().any().describe()
train_data.shape, test_data.shape
train_data = train_data / 255.0
test_data = test_data / 255.0
train_data = train_data.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)
train_data.shape, test_data.shape
label = tf.keras.utils.to_categorical(label, num_classes= 10)
label
var = plt.imshow(train_data[3][:,:,0])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size = 0.1,
                                                   random_state = 2)
def build_model():
    model = tf.keras.models.Sequential()
    
    
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = "Same",
                                    activation = "relu", input_shape = (28,28,1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = "Same",
                                    activation = "relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "Same",
                                    activation = "relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), padding = "Same",
                                    activation = "relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 256, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units = 10, activation = "softmax"))
    
    return model

model = build_model()
model.summary()
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = 0.001), loss= "categorical_crossentropy",
             metrics= ["accuracy"])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size= 64, epochs= 15)
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()
plt.plot(history.history['accuracy'], color='b')
plt.plot(history.history['val_accuracy'], color='r')
plt.show()
pred = model.predict(test_data)
pred.shape
pred
pred = np.argmax(pred, axis= 1)
pred.shape
pred
pred_df = pd.DataFrame(data= pred)
pred_df.index += 1
pred_df = pred_df.reset_index()
pred_df.columns = ["ImageId", "Label"]
pred_df.head()
pred_df.to_csv("cnn_out.csv", index= False)
