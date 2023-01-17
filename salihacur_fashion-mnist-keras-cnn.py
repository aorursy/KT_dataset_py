# import libraries

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from keras.models import Sequential

from sklearn.metrics import confusion_matrix

from keras.optimizers import Adam,RMSprop

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
#filter warnings

warnings.filterwarnings("ignore")

# import data 

from keras.datasets  import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Examine size of the data 

print("Train Set Shape: ")

print(x_train.shape)

print("Test Set Shape: ")

print(x_test.shape)
# Unique Classes in Train and Test Datas

unique_classes,u_counts = np.unique(np.concatenate([y_train,y_test]),return_counts=True)

print(unique_classes)

print(u_counts)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

class_names_tr = ['Tişört / Üst', 'Pantolon', 'Kazak', 'Elbise', 'Ceket','Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Çizme']
num_classes = len(unique_classes)
# Plot the first 10 pictures in the dataset

for i in range(9):

    # define subplot

    plt.subplot(330+ 1 + i)

    # plot raw pixel data

    plt.imshow(x_train[i])

# show the figure

plt.show()
# Create Model with KERAS library

model=Sequential()

model.add(Conv2D(64, kernel_size=3, activation='sigmoid', padding='same', input_shape=(28,28,1)))

model.add(Conv2D(64, kernel_size=3, activation='relu',padding='same'))

model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(128, kernel_size=3, activation='sigmoid',padding='same'))

model.add(Conv2D(128, kernel_size=3, activation='relu',padding='same'))

model.add(MaxPooling2D(padding='same'))

model.add(Dense(256,activation='sigmoid'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))



# Model Summary

model.summary()
opt = Adam(lr = 0.0001)

# Compile Model

model.compile(

    optimizer=opt,

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy'])
# Resize image shapes

x_train = x_train.reshape(-1,28, 28, 1) 

x_test = x_test.reshape(-1,28, 28, 1) 
# Fit Model

history =  model.fit(x_train,y_train, 

                     epochs=50,

                     validation_data=(x_test, y_test))
# model save

model.save_weights("example.h5")
# Visualize Loss and Accuracy Rates

plt.plot(history.history["loss"],label="train_loss")

plt.plot(history.history["val_loss"],label="val_loss")

plt.legend()

plt.show()



plt.figure()

plt.plot(history.history["accuracy"],label="train_acc")

plt.plot(history.history["val_accuracy"],label="val_acc")

plt.legend()

plt.show()
# Predict images

y_pred = model.predict(x_test)

y_pred = np.argmax(np.round(y_pred),axis=1)
# Correctly predicted images

correct = np.where(y_pred==y_test)[0]

print("Found {} correct labels".format(len(correct)))

for i, correct in enumerate(correct[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(y_pred[correct], y_test[correct]))

    plt.tight_layout()
# incorrectly predicted images

incorrect = np.where(y_pred!=y_test)[0]

print("Found {} incorrect labels".format(len(incorrect)))

for i, incorrect in enumerate(incorrect[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(y_pred[incorrect], y_test[incorrect]))

    plt.tight_layout()
# confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# plot the confusion matrix

f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(cm, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
# Model Results

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

print('Confusion Matrix :')

print(cm) 

print('Accuracy Score :',accuracy_score(y_test, y_pred))

print('Report : ')

print(classification_report(y_test, y_pred))