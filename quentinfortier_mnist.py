import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import keras
X_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv") # chargement des données

X_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

X_train.head()
Y_train = X_train.label # séparation des données et des labels

del X_train['label']
for i in range(9): # affichage de quelques images

    plt.subplot(190 + (i+1))

    im = X_train.iloc[i].to_numpy().reshape(28, 28)

    plt.imshow(im, cmap=plt.get_cmap('gray'))

    plt.title(Y_train[i]);
Y_train.hist() # répartition des labels
X_train = X_train/255 - 0.5 # normalisation, utile pour les algorithmes de ML

X_test = X_test/255 - 0.5
model = keras.models.Sequential([

        keras.layers.Dense(128, activation='relu'),

        keras.layers.Dense(10, activation='softmax')

])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, keras.utils.to_categorical(Y_train), epochs=5)
Y_pred = model.predict(X_train)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

err = np.where(Y_pred_classes != Y_train)[0] # erreurs par le modèle

np.random.shuffle(err)
for i in range(9): # affichage d'images mal classifiées

    plt.subplot(190 + (i+1))

    im = X_train.iloc[err[i]].to_numpy().reshape(28, 28)

    plt.imshow(im, cmap=plt.get_cmap('gray'))

    plt.title(Y_pred_classes[err[i]]);
plt.hist(Y_pred_classes)
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_train, Y_pred_classes)
model_cnn = keras.models.Sequential([

  keras.layers.Reshape((28, 28, 1), input_shape=(784,)), # redimensionner pour appliquer un filtre de convolution

  keras.layers.Conv2D(8, 3, input_shape=(28, 28, 1)),

  keras.layers.MaxPooling2D(),

  keras.layers.Flatten(),

  keras.layers.Dense(10, activation='softmax'),

])

model_cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_cnn.fit(X_train, keras.utils.to_categorical(Y_train), epochs=5)
confusion_matrix(Y_train, Y_pred_classes)
results = model_cnn.predict(X_test)



results = np.argmax(results,axis = 1)



submission = pd.DataFrame({"ImageId" : range(1, len(results)+1), "Label" : results})



submission.to_csv("cnn_mnist.csv",index=False)
