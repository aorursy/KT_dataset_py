import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
y_train
y_test
# we need one hot encode to get the classes
y_train.shape
y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)
x_train = x_train/255
x_test = x_test/255
scaled_single = x_train[0]
scaled_single.max()
plt.imshow(scaled_single)
x_train.shape
x_test.shape
x_train = x_train.reshape(60000, 28, 28, 1)
x_train.shape
x_test = x_test.reshape(10000,28,28,1)
x_test.shape
model = Sequential()

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# https://keras.io/metrics/
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) # we can add in additional metrics https://keras.io/metrics/
model.summary()
early_stop = EarlyStopping(monitor='val_loss',patience=2)
model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])
model.metrics_names
losses = pd.DataFrame(model.history.history)
losses.head()
losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()
print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(x_test)
y_cat_test.shape
y_cat_test[0]
predictions[0]
y_test
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)
import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)

my_number = x_test[0]
plt.imshow(my_number.reshape(28,28))
# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_number.reshape(1,28,28,1))