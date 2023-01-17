import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import matplotlib.pyplot as plt
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images[0].shape
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    tmp = train_images[i].reshape(img_rows, img_cols)
    plt.imshow(tmp)
    train_images[i].reshape(img_rows, img_cols, 1)
plt.show()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])
model.fit(train_images, train_labels,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1)
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
from keras.datasets import cifar10
import numpy as np
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images.shape
batch_size = 128
num_classes = 10
epochs = 35
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
#train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 3)
#test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 3)

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

train_images = train_images / 255.0
test_images = test_images / 255.0
train_images[0].shape
#train_images[0]
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    #tmp = train_images[i].reshape(img_rows, img_cols)
    #print(cur)
    #print(np.argmax(train_labels[i]))
    plt.imshow(train_images[i])
    #train_images[i].reshape(img_rows, img_cols, 1)
plt.show()
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adadelta(),
             metrics=['accuracy'])
model.fit(train_images, train_labels,
         batch_size=batch_size,
         epochs=epochs,
         verbose=1)
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
from sklearn.metrics import confusion_matrix
pred=model.predict_classes(test_images)
r_test_labels=np.argmax(test_labels, axis=1)
r_test_labels
con_mat=confusion_matrix(r_test_labels,pred)
print(con_mat)
#con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
import pandas as pd  
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
classes=[0,1,2,3,4,5,6,7,8,9]
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)
import seaborn as sns
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

