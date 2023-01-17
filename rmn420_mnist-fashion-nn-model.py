%matplotlib inline

import numpy as np

from tensorflow import keras

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.optimizers import SGD, adam

from keras.utils import np_utils

import matplotlib.pyplot as plt



np.random.seed(0)
fashion_mnist = keras.datasets.fashion_mnist



(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("X_train shape: ",X_train.shape)

print("y_train shape: ",y_train.shape)

print("X_test shape: ",X_test.shape)

print("y_test shape: ",y_test.shape)
X_train = X_train / 255.0

X_test = X_test / 255.0
plt.imshow(X_train[0])

plt.colorbar()

plt.grid(False)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[y_train[i]])

plt.show()
# 1: Build the model



#model = keras.Sequential(Dense(128, activation='relu'))

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation = 'relu'),

    keras.layers.Dense(10,activation= 'softmax')

])

# 2: Compile the model  

model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# 3: Fit the model

model.fit(X_train, y_train, epochs = 5)

# 4: evaluate

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Accuracy:", test_acc)
preds = model.predict(X_test)

preds[0]
np.argmax(preds[0])
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_test[i], cmap=plt.cm.binary)

    pred_label = np.argmax(preds[i])

    true_label = y_test[i]

    if(pred_label==true_label):

        color ='green'

    else:

        color = 'red'

    plt.xlabel("{} ({})".format(class_names[pred_label],class_names[true_label]),color=color)

plt.show()