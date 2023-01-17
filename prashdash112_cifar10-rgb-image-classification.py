import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
# Unpacking of the data in training & testing set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizing the data
x_train = x_train/255
x_test = x_test/255

# One hot encoding
y_cat_train = to_categorical(y_train,10)
y_cat_test = to_categorical(y_test,10)
model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss',patience=3)
model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses.head()
print('losses columns:',losses.columns)
print('metrics:',model.metrics_names)
losses[['accuracy','val_accuracy']].plot()
losses[['loss','val_loss']].plot()
print(model.metrics_names)
print(model.evaluate(x_test,y_cat_test,verbose=0))
predictions = model.predict_classes(x_test)
print('Classification report:')
print('\n')
print(classification_report(y_test,predictions))
print('\n')
print('Confusion matrix:')
print('\n')
print(confusion_matrix(y_test,predictions))
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
my_image = x_test[16]
print(y_test[16])
plt.imshow(my_image)
# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_image.reshape(1,32,32,3))
# model is correctly predicting the value for the image of object dog 