import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 





%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')





import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import InputLayer

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.utils import plot_model





from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, accuracy_score



!pip install mlxtend

import mlxtend as ml

from mlxtend.plotting import plot_confusion_matrix
# fashion MNIST dataset 

fashion_mnist = keras.datasets.fashion_mnist
# train and test 

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# class 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# shap of train  

X_train.shape
# shape labels 

y_train.shape
# images

plt.figure()

plt.imshow(X_train[0])

plt.colorbar()

plt.grid(False)

plt.show()
# plotting images 

for i in range(1,4):

  plt.figure(figsize=(12,6))

  plt.imshow(X_train[i])

  plt.colorbar()

  plt.grid(False)

  plt.show
# preprocessing

X_train = X_train / 255.0

X_test = X_test / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i],  cmap=plt.cm.binary) #cmap=GnBu

    plt.xlabel(class_names[y_train[i]])

plt.show()
model = Sequential([

                  InputLayer(input_shape=(28,28), name='input_layer'), # input layer

                  Flatten(name='Flatten'), # Flatten 1D array 

                  Dense(units=128, activation='relu', name='hidden_layer_1'),  # hidden layer 1

                  Dense(units=54, activation='relu', name='hidden_layer_2'),  # hidden layer 2 

                  Dense(units=10, activation='softmax', name='output_layer') # output layer (predictions)

                    ])
# summary 

print(model.summary())
model.compile(optimizer=SGD(learning_rate=0.001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
np.random.seed(seed=42)



history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.20)
# evaluation 

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
# first metrics the Neural network 

print('Binary CrossEntropy: {}'.format(test_loss))

print('Accuracy: {}'.format(test_acc))
# prediction 

predictions = model.predict(X_test)

predictions
# Predict class

np.argmax(predictions[0])
# metrics for each class 



y_pred = model.predict_classes(X_test)

y_proba = model.predict_proba(X_test)[:,1]

print('Acurácia: {}'.format(accuracy_score(y_test, y_pred)))

print('\n')

print(classification_report(y_test, y_pred))
history.history
# Accuracy 



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Accuracy', fontsize=14)

plt.xlabel('Epoch', fontsize=14)

plt.ylabel('Accuracy',fontsize=14)

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()





print('\n')

print('\n')





# Cross Entropy  

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('LogLoss', fontsize=14)

plt.xlabel('Epoch', fontsize=14)

plt.ylabel('Loss',fontsize=14)

plt.legend(['Train', 'Teste'], loc='upper left')

plt.show()
cm = confusion_matrix(y_test, y_pred)

plot_confusion_matrix(conf_mat=cm, figsize=(8,8)) # test argument called:  "class_names"

plt.title('Confusion Matrix', fontsize=14)

plt.tight_layout()
plot_model(model, "multi_layer_perceptron.png", show_shapes=True)