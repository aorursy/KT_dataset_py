import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import keras

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Lambda

from keras.utils import to_categorical 

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



import matplotlib.pyplot as plt

import itertools

df = pd.read_csv('../input/digit-recognizer/train.csv', encoding='utf8', delimiter=',')

df_test = pd.read_csv('../input/digit-recognizer/test.csv', encoding='utf8', delimiter=',')
y = df[['label']]

X = df.drop(['label'], axis=1)



# for model check

X_true = X.to_numpy().reshape(X.shape[0], 28, 28, 1) / 255.0

y_true = y



# split data to train and validation

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)



# reshape all sets to picture size (28, 28, 1-color) and normalize



x_train = x_train.to_numpy().reshape(x_train.shape[0], 28, 28, 1) / 255.0

y_train = to_categorical(y_train)



x_val = x_val.to_numpy().reshape(x_val.shape[0], 28, 28, 1) / 255.0

y_val = to_categorical(y_val)



x_test = df_test.to_numpy().reshape(df_test.shape[0], 28, 28, 1) / 255.0

datagen = keras.preprocessing.image.ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False, 

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False, 

        zca_whitening=False, 

        rotation_range=10,

        width_shift_range=0.07,

        height_shift_range=0.07,

        shear_range=0.02,

        zoom_range = 0.10,

        horizontal_flip=False)



datagen.fit(x_train)
def input_data(x): 

    return x



model = Sequential()

model.add(Lambda(input_data,input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(700, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(700, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', 

              loss='categorical_crossentropy',  

              metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',

                                           patience = 3,

                                           verbose = 1,

                                           factor = 0.5,

                                           min_lr = 0.00001)



callbacks_list = [learning_rate_reduction]

model.fit_generator(datagen.flow(x_train, y_train, batch_size=512),

                    steps_per_epoch=len(x_train) // 32, epochs=50,

                    callbacks=callbacks_list,

                    validation_data=(x_val, y_val))
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()





# check prediction for whole set

y_predicted = np.transpose(np.argmax(model.predict(X_true), axis=1))

confusion_mtx = confusion_matrix(y_true, y_predicted)

plot_confusion_matrix(confusion_mtx, classes = range(10))

np.argmax(model.predict(x_test), axis=1)