import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt

import seaborn as sns

import math



# Input data files are available in the "../input/" directory.

data = pd.read_csv("../input/mushrooms.csv")
data.head()
Y = pd.get_dummies(data.iloc[:,0],  drop_first=False)

X = pd.DataFrame()

for each in data.iloc[:,1:].columns:

    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)

    X = pd.concat([X, dummies], axis=1)

    

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score

from keras import backend as K

from keras.layers import BatchNormalization

seed = 123456 



def create_model():

    model = Sequential()

    model.add(Dense(20, input_dim=X.shape[1], activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))



    model.compile(loss='binary_crossentropy' , optimizer='sgd', metrics=['accuracy'])

    

    return model
model = create_model()

history = model.fit(X.values, Y.values, validation_split=0.20, epochs=300, batch_size=100, verbose=0)



# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % 

      (100*history.history['acc'][-1], 100*history.history['val_acc'][-1]))