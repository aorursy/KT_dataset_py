import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from seaborn import heatmap

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import  Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense

from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
df_train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

df_test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
df_test.head()
train_label = df_train['label']

test_label = df_test['label']



df_test = df_test.drop(columns = ['label'])

df_train = df_train.drop(columns = ['label'])



X_train, X_val, y_train, y_val = train_test_split(df_train, train_label, test_size = .3, stratify = train_label)



X_train = (X_train/255).values.reshape(len(X_train), 28,28,1)

X_val = (X_val/255).values.reshape(len(X_val), 28,28, 1)

X_test = (df_test/255).values.reshape(len(df_test), 28,28, 1)
print(X_train.shape)

print(X_val.shape)

print(X_test.shape)
plt.figure(figsize = (8,6))

train_label.value_counts().plot(kind = 'bar', color = 'green')

test_label.value_counts().plot(kind = 'bar', color = 'red', alpha = .5)
#plt.imshow(df_test.iloc[0, ""].values.reshape((28,28)))

#plt.imshow(df_test.iloc[1, :].values.reshape((28,28)))

#plt.imshow(df_test.iloc[2, :].values.reshape((28,28)))



f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5):

    ax[i].imshow(df_test.iloc[i, :].values.reshape(28, 28))

plt.show()
def Model():

    

    model = Sequential()



    #CCN1

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',activation = 'relu', input_shape = (28,28, 1)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = (28,28, 1)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same',activation = 'relu', input_shape = (28,28, 1)))

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(.3))



    #CCN2

    model.add(Conv2D(filters = 128, padding = 'same', kernel_size = (2,2), activation = 'relu'))

    model.add(Conv2D(filters = 128, padding = 'same',kernel_size = (2,2), activation = 'relu'))

    model.add(Conv2D(filters = 128, padding = 'same',kernel_size = (2,2), activation = 'relu'))

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(.25))



    #CCN3

    model.add(Conv2D(filters = 256, padding = 'same',kernel_size = (2,2), activation = 'relu'))

    model.add(Conv2D(filters = 256, padding = 'same',kernel_size = (2,2), activation = 'relu'))

    model.add(Conv2D(filters = 256, padding = 'same', kernel_size = (2,2), activation = 'relu'))

    

    model.add(MaxPool2D(pool_size = (2,2)))

    model.add(Dropout(.25))

    

      #CCN3

    #model.add(Conv2D(filters = 512, kernel_size = (2,2), activation = 'relu'))

    #model.add(MaxPool2D(pool_size = (2,2)))

    #model.add(Dropout(.25))





    #Flatten

    model.add(Flatten())



    # Dense1

    model.add(Dense(1024,  activation = 'relu'))

    model.add(Dense(1024,  activation = 'relu'))

    model.add(Dropout(.2))



    # Dense2

    model.add(Dense(512,  activation = 'relu'))

    model.add(Dense(512,  activation = 'relu'))

    model.add(Dropout(.3))



    # Dense3

    model.add(Dense(128,  activation = 'relu'))

    model.add(Dropout(.3))



    # Dense4

    model.add(Dense(64,  activation = 'relu'))

    model.add(Dropout(.3))



    # Outpot

    model.add(Dense(10,  activation = 'softmax'))

    model.compile(optimizer = Adam(lr = .001), loss = sparse_categorical_crossentropy, metrics = ['accuracy'])

    

    return model
model = Model()

model.summary()
batch_size = 128



early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose = 0)

model_save =  ModelCheckpoint('model_weights.hdf5' , monitor = 'val_loss', save_best_only = True, mode = 'min')

reduce_lr =  ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=0.00001)



history = model.fit(X_train, y_train, callbacks=[early_stopping, model_save, reduce_lr], 

          batch_size =batch_size, epochs = 50, validation_data = (X_val, y_val), verbose = 1  )

plt.figure(figsize = (8,6))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Train', 'Test'])

plt.show()
score_train = model.evaluate(X_train, y_train)

score_val = model.evaluate(X_val, y_val)
print("Loss Train : {}  and  Accuracy Train : {}".format(score_train[0], score_train[1]))

print("Loss validate : {}  and  Accuracy Validate : {}".format(score_val[0], score_val[1]))
y_predict = model.predict_classes(X_test)
y_predict
CM = confusion_matrix(y_predict, test_label)

print(CM)
plt.figure(figsize = (12,8))

heatmap(CM, annot = True, cmap=plt.cm.Blues, fmt="d", linecolor ='black',  linewidths=1)
print("Prediction Accuracy:" )

print(accuracy_score(y_predict, test_label))