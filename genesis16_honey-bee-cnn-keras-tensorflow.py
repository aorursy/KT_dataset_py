import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage import io, transform
print(os.listdir("../input/pollendataset/PollenDataset/"))
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPool2D, Flatten, Dense
from keras.optimizers import SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
data = pd.read_csv("../input/pollendataset/PollenDataset/pollen_data.csv")
data.head(10)
def dataset_gen(data, size = (300,180)):
    
    img_data = []
    labels = []
    for img_name, pollen_carrying in zip(data['filename'], data['pollen_carrying']):
        img = io.imread(os.path.join("../input/pollendataset/PollenDataset/images", img_name))
        img = transform.resize(img, size, mode = 'constant')
        img_data.append(img)
        labels.append(pollen_carrying)
        
    return np.array(img_data), np.array(labels)
x, y = dataset_gen(data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 7)

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = 2)

x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(x_train, y_train, test_size = 0.15, random_state = 7)

for data in [ x_train,y_train,x_test,y_test,x_train_val,x_test_val,y_train_val,y_test_val]:
    print(data.shape)
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (300,180,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,
                 (3,3),
                activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.60))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.50))
model.add(Dense(2, activation = 'softmax'))
model.summary()

model.compile(optimizer = 'Adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = x_train_val,
         y = y_train_val,
         batch_size = 16,
         epochs = 50,
         validation_data = (x_test_val, y_test_val))
pred = model.evaluate(x_test,
                      y_test,
                    batch_size = 32)
print(pred)
