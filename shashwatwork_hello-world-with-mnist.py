import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

import keras

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.vis_utils import plot_model

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping





sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
g = sns.catplot(x="label", kind="count", palette='bright', data=train)

g.fig.set_size_inches(16, 5)



g.ax.set_title('MNIST by Class', fontsize=20)

g.set_xlabels(' MNIST Class', fontsize=14)

g.set_ylabels('Number of Data Points', fontsize=14)
X = train.drop(['label'], 1).values

X = X / 255.0



y = train['label'].values



test_x = test.values

test_x = test_x / 255.0
X = X.reshape(-1,28,28,1)

test_x = test_x.reshape(-1,28,28,1)
y = to_categorical(y)



print(f"Shape of Label Data {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train__ = X_train.reshape(X_train.shape[0], 28, 28)



fig, axis = plt.subplots(1, 4, figsize=(20, 10))

for i, ax in enumerate(axis.flat):

    ax.imshow(X_train__[i], cmap='binary')

    digit = y_train[i].argmax()

    ax.set(title = f"Real Number is {digit}");
x_train = np.asarray(X_train)

x_test = np.asarray(X_test)



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std
input_shape = (28, 28, 1)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

plot_model(model, show_shapes=True, show_layer_names=True)
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(lr=0.001),

              metrics=['accuracy'])
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
batch_size = 32 

epochs = 50



history = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          callbacks = [checkpoint, early],

          verbose=1,

          validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
plt.figure(figsize=(14, 5))



plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'], label='Training Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label='Training Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)

print("Model loss: {0:.4f}, Model accuracy: {1:.4f}".format(final_loss, final_acc))
y_pred_enc = model.predict(test_x)

y_pred = [np.argmax(i) for i in y_pred_enc]
fig, ax = plt.subplots(figsize=(18, 12))

for ind, row in enumerate(test_x[:15]):

    plt.subplot(3, 5, ind+1)

    plt.title(y_pred[ind])

    img = row.reshape(28, 28)

    fig.suptitle('Predictions', fontsize=20)

    plt.axis('off')

    plt.imshow(img, cmap='Dark2')
mnist_test = "/kaggle/input/digit-recognizer/test.csv"

mnist_test = np.loadtxt(mnist_test, skiprows=1, delimiter=',')

num_images = mnist_test.shape[0]

out_x = mnist_test.reshape(num_images, 28, 28, 1)

out_x = out_x / 255

results = model.predict(out_x)

results = np.argmax(results,axis = 1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),"Label": results})

submissions.to_csv("submission.csv", index=False, header=True)

print('Submission csv Generated!')
