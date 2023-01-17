import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.nn import relu, softmax

%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#remove label column
x_train = train.drop(labels=['label'],axis=1)
#remove everything except label column
y_train = train['label']

#note: x_train and y_train are still Pandas Dataframes
x_train /= 255
test /= 255
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
plt.imshow(x_train[88][:,:,0], cmap='gray')
y_train = to_categorical(y_train, num_classes = 10)
seed=5602
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation=relu, padding='Same'))
model.add(Conv2D(32, (5, 5), activation=relu, padding='Same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation=relu, padding='Same'))
model.add(Conv2D(64, (3, 3), activation=relu, padding='Same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation=relu))
model.add(Dropout(0.5))
model.add(Dense(10, activation=softmax))

model.compile(optimizer = 'Adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train,y_train,epochs=30)
val_loss, val_acc = model.evaluate(x_val, y_val)
print(val_loss, val_acc)
from random import randint

n = randint(0, len(test))

prediction = model.predict(test)
prediction_NUM = np.argmax(prediction[n])

print(prediction_NUM)

plt.imshow(test[n][:,:,0], cmap='gray')
plt.show()
