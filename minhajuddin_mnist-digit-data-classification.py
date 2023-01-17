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
train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
train_label = train_data['label']
train_data = train_data.drop('label',axis=1)
train_label , train_data = train_label.values, train_data.values
train_data = train_data/255
test_data = test_data/255
train_data = train_data.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)
from keras.utils import to_categorical
train_label = to_categorical(train_label, num_classes = 10)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3, random_state=42)
from keras.models import Sequential
from keras.layers import Conv2D, Dense,Dropout,MaxPool2D, Flatten
model = Sequential()

#First Layer
model.add(Conv2D(64, 3, activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Second Layer
model.add(Conv2D(32, 3, activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

#Final Output Layer
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Training Model 
history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, 
                    validation_data = (x_test, y_test))
#Visualizing Model Accuracy
import matplotlib.pyplot as plt

plt.title('Model Accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])
plt.show()
def prep_test_data(raw):
    x = raw[:,0:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, 28, 28, 1)
    out_x = out_x / 255
    return out_x

val_file = "/kaggle/input/digit-recognizer/test.csv"
val_data = np.loadtxt(val_file, skiprows=1, delimiter=',')
x_test = prep_test_data(val_data)
predictions = model.predict_classes(x_test)

indexes = [i for i in range(1,len(val_data)+1)]
output = pd.DataFrame({'ImageId': indexes,'Label': predictions})
output.to_csv('submission.csv', index=False)
output
