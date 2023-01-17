import numpy as np

import pandas as pd

#tensorflow version 2.0

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import train_test_split



#reading data

mnist_train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

mnist_test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')



print("Datasets successfully loaded!")

print(f"Training dataset has {mnist_train.shape[0]} rows and {mnist_train.shape[1]} columns.")

print(f"Training dataset has {mnist_test.shape[0]} rows and {mnist_test.shape[1]} columns.")
#standardization

mnist_train.iloc[:,1:] /= 255

mnist_test.iloc[:,1:] /= 255



#splitting features and target column

x_train = mnist_train.iloc[:,1:]

y_train = mnist_train.iloc[:,0]

x_test= mnist_test.iloc[:,1:]

y_test=mnist_test.iloc[:,0]



#further splitting train set into validation and training set

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.3)
plt.figure(figsize=(10, 10))

for i in range(36):

    plt.subplot(6, 6, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(np.array(x_test.iloc[i]).reshape(28,28))

    label_index = int(y_test[i])

    plt.title(label_index)

plt.show()
sns.countplot(y_train)

plt.title('Classes distribution in train set');
sns.countplot(y_validate)

plt.title('Classes distribution in validation set');
sns.countplot(y_test)

plt.title('Classes distribution in test set');
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

    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),

    tf.keras.layers.MaxPooling2D(pool_size=2) ,

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(), # flatten out the layers

    tf.keras.layers.Dense(100,activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(100,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(25,activation = 'softmax')

])



cnn_model.compile(loss ='sparse_categorical_crossentropy',

                  optimizer='adam',metrics =['accuracy'])



early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



history = cnn_model.fit(

    x_train,

    y_train,

    batch_size=500,

    epochs=80,

    verbose=1,

    validation_data=(x_validate,y_validate),

    callbacks=early_stop

)
score = cnn_model.evaluate(x_test,y_test,verbose=0)

print('Test Loss : {:.4f}'.format(score[0]))

print('Test Accuracy : {:.4f}'.format(100*score[1]))
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

plt.title('Training - Accuracy');
cnn_pred = cnn_model.predict_classes(x_test)

target_names = ["Class {}".format(i) for i in range(24)]

print(classification_report(y_test,cnn_pred, target_names=target_names))
plt.figure(figsize=(10,10))

sns.heatmap(confusion_matrix(y_test,cnn_pred),cmap='seismic');