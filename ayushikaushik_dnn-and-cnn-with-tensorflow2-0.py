import numpy as np

import pandas as pd

#tensorflow version 2.0

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



#reading data

mnist_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

mnist_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
#standardization

mnist_train.iloc[:,1:] /= 255

mnist_test.iloc[:,1:] /= 255



#splitting features and target column

x_train = mnist_train.iloc[:,1:]

y_train = mnist_train.iloc[:,0]

x_test= mnist_test.iloc[:,1:]

y_test=mnist_test.iloc[:,0]



#further splitting train set into validation and training set

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
print("Let's have a look at the images in our dataset.")

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))

for i in range(36):

    plt.subplot(6, 6, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(np.array(x_test.iloc[i]).reshape(28,28))

    label_index = int(y_test[i])

    plt.title(class_names[label_index])

plt.show()
sns.countplot(y_train)

plt.title('Classes distribution in train set');
sns.countplot(y_validate)

plt.title('Classes distribution in validation set');
sns.countplot(y_test)

plt.title('Classes distribution in test set');
input_size = 784

output_size = 10

hidden_layer_size = 300

model = tf.keras.Sequential([

    tf.keras.layers.Dense(input_size, activation='relu'),  # input layer

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3rd hidden layer

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(output_size, activation='softmax') # output layer

])



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)



model.fit(x_train,y_train, epochs=60, validation_data=(x_validate,y_validate),validation_steps=1, verbose =2,callbacks=early_stop)
test_loss, test_accuracy = model.evaluate(x_test,y_test)

print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
pred = model.predict_classes(x_test)

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(10)]

print(classification_report(y_test,pred, target_names=target_names))
sns.heatmap(confusion_matrix(y_test,pred),cmap='seismic');
image_rows = 28

image_cols = 28

image_shape = (image_rows,image_cols,1)

x_train = tf.reshape(x_train,[x_train.shape[0],*image_shape])

x_test = tf.reshape(x_test,[x_test.shape[0],*image_shape])

x_validate = tf.reshape(x_validate,[x_validate.shape[0],*image_shape])
cnn_model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),

    tf.keras.layers.MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(), # flatten out the layers

    tf.keras.layers.Dense(196,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dense(10,activation = 'softmax')

])



cnn_model.compile(loss ='sparse_categorical_crossentropy',

                  optimizer='adam',metrics =['accuracy'])



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)



history = cnn_model.fit(

    x_train,

    y_train,

    batch_size=4096,

    epochs=75,

    verbose=1,

    validation_data=(x_validate,y_validate),

    callbacks=early_stop

)
plt.figure(figsize=(10, 10))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Training - Accuracy')
cnn_pred = cnn_model.predict_classes(x_test)

target_names = ["Class {}".format(i) for i in range(10)]

print(classification_report(y_test,cnn_pred, target_names=target_names))
score = cnn_model.evaluate(x_test,y_test,verbose=0)

print('Test Loss : {:.4f}'.format(score[0]))

print('Test Accuracy : {:.4f}'.format(score[1]))