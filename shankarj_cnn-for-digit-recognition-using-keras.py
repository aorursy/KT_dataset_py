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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/digit-recognizer/train.csv')
print(data.describe())
data.head()
X,Y = data.drop('label',axis = 1),data['label']
X = X/255.0
X = np.array(X).reshape(X.shape[0],28,28)
Y = keras.utils.to_categorical(Y,num_classes = 10)
w=10
h=10
fig=plt.figure(figsize=(15, 15))
columns = 5
rows = 4
for i in range(1, columns*rows +1):
    img = np.random.randint(10, size=(h,w))
    fig.add_subplot(rows, columns, i)
    loc = random.randint(0,len(X)-1)
    plt.imshow(X[loc],cmap = 'gray')
plt.show()
X = X.reshape(-1,28,28,1)
Xtrain, xval, Ytrain, yval = train_test_split(X,Y,test_size = 0.22)
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',
                              activation = 'relu',input_shape = (28,28,1)))
model.add(keras.layers.Conv2D(filters = 64,kernel_size = (5,5),
                              activation = 'relu'))

model.add(keras.layers.MaxPooling2D((2,2),strides = (1,1),padding = 'same'))

model.add(keras.layers.Conv2D(filters = 32,kernel_size = (3,3),
                             activation = 'relu',padding = 'same'))
model.add(keras.layers.Conv2D(filters = 32,kernel_size = (3,3),
                             activation = 'relu'))

model.add(keras.layers.MaxPooling2D((2,2),padding = 'same'))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256,activation = 'relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10,activation = 'softmax'))
keras.utils.plot_model(model,to_file = 'model.png',show_shapes = True)
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
history = model.fit(Xtrain, Ytrain, epochs = 50, batch_size = 32, validation_data = (xval, yval))
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(train_acc,label = "Training")
plt.plot(val_acc,label = 'Validation/Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss,label = 'Training')
plt.plot(val_loss,label = 'Validation/Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
test_data = test_data/255.0
test_data = np.array(test_data).reshape(test_data.shape[0],28,28)
test_data = test_data.reshape(-1,28,28,1)
prediction = model.predict(test_data)
decoded_preds = np.argmax(prediction,axis = 1)
submissions = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submissions['Label'] = np.array(decoded_preds)
submissions.to_csv('submissions.csv')
