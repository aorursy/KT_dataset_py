import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import random



import warnings

warnings.simplefilter('ignore')



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



sns.set(style='white', context='notebook', palette='deep')
Y_train = train['label']

X_train = train.drop('label',axis=1)

del train
plt.figure(figsize=(14,8))

graf = sns.countplot(Y_train, palette="deep")

plt.show()

Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

X_train, X_val, Y_train, Y_val =train_test_split(X_train,Y_train,test_size=0.2,random_state=42)
fig=plt.figure(figsize=(16, 16))

for i in range(1,10):

    rand_num = random.randint(0,25000)

    fig.add_subplot(3, 3, i)

    img_number = plt.imshow(X_train[rand_num][:,:,0])
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()
datagen = ImageDataGenerator(zoom_range = 0.1,

                            height_shift_range = 0.1,

                            width_shift_range = 0.1,

                            rotation_range = 10)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 20

batch_size = 16
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),

                           steps_per_epoch=X_train.shape[0] // batch_size,

                           epochs=epochs, 

                           verbose=2,

                           validation_data=(X_val,Y_val), 

                           callbacks=[learning_rate_reduction])
Y_hat = model.predict(X_val)

Y_pred = np.argmax(Y_hat, axis=1)

Y_true = np.argmax(Y_val, axis=1)

cm = confusion_matrix(Y_true, Y_pred)

classes = range(10)

plt.imshow(cm, interpolation='nearest', cmap = plt.cm.gray_r)

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
fig, ax = plt.subplots(2,1)

ax[0].plot(hist.history['loss'], color='b', label="Training loss")

ax[0].plot(hist.history['val_loss'], color='r', label="Validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(hist.history['acc'], color='b', label="Training accuracy")

ax[1].plot(hist.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
predicted_classes = model.predict_classes(test)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("submissions.csv", index=False, header=True)
model.save('my_model.h5')

json_string = model.to_json()