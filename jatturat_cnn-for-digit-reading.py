# Data Manipulation

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

import itertools



# Visualization

import matplotlib.pyplot as plt

%matplotlib inline



# Evaluation

from sklearn.metrics import confusion_matrix



# Neural Network

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")
train_set.isnull().sum().sum()
test_set.isnull().sum().sum()
X_train = train_set.drop(['label'], axis=1)

y_train = train_set['label']

X_test = test_set
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
X_train = X_train.astype("float32")/255.

X_test = X_test.astype("float32")/255.
fig, ax = plt.subplots(1, 5, figsize=(12,6))

for i in range(5):

    ax[i].imshow(X_train[i].reshape(28,28), cmap='gray')

    ax[i].set_title(y_train[i], fontsize=20)
y_train = to_categorical(y_train)

print(y_train[0])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=0)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer = Adam(lr=1e-4), loss='categorical_crossentropy', metrics=["accuracy"])
learning_rate = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
datagen = ImageDataGenerator(rotation_range = 10,

                             width_shift_range = 0.1,

                             height_shift_range = 0.1,

                             zoom_range = 0.1)
hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),

                           steps_per_epoch=600,

                           epochs=20,

                           verbose=2,  

                           validation_data=(X_val[:400,:], y_val[:400,:]), 

                           callbacks=[learning_rate])
final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
plt.plot(hist.history['loss'], color='b')

plt.plot(hist.history['val_loss'], color='r')

plt.show()

plt.plot(hist.history['acc'], color='b')

plt.plot(hist.history['val_acc'], color='r')

plt.show()
y_hat = model.predict(X_val)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)



classes = range(10)



plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)

plt.title('Confusion matrix')

plt.colorbar()

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes, rotation=45)

plt.yticks(tick_marks, classes)



thresh = cm.max() / 2.

for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

    plt.text(j, i, cm[i, j],

             horizontalalignment="center",

             color="white" if cm[i, j] > thresh else "black")



plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')
y_hat_submit = model.predict(X_test, batch_size=64)

y_pred_submit = np.argmax(y_hat_submit,axis=1)
with open('submission.csv', 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(y_pred_submit)) :

        f.write("".join([str(i+1),',',str(y_pred_submit[i]),'\n']))