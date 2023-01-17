from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import optimizers

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

labels = train.ix[:,0].values.astype('int32')

X_train = (train.ix[:,1:].values).astype('float32')

X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels to binary class matrix

y_train = np_utils.to_categorical(labels) 
#Scaling the values



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

X_test_sca = scaler.fit_transform(X_test)

X_train_sca = scaler.fit_transform(X_train)

X_test_sca = X_test_sca.reshape(X_test_sca.shape[0],28, 28,1)

X_train_sca = X_train_sca.reshape(X_train_sca.shape[0],28,28,1)



nb_classes = y_train.shape[1]

for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train_sca[i], cmap=plt.get_cmap('gray'))

    plt.title(labels[i]);

    
input_shape = (28, 28, 1)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))



sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# we'll use categorical xent for the loss, and RMSprop as the optimizer

#model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta())
print("Training...")

model.fit(X_train_sca, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=2)


print("Generating test predictions...")

preds = model.predict_classes(X_test, verbose=0)



def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(preds, "keras-mlp.csv")

print("Test predictions Finished")
