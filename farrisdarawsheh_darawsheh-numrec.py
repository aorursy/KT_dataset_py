# Imports & Froms and others 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation

from keras.optimizers import adam

from keras.callbacks import ReduceLROnPlateau

from keras import backend as K

from keras.utils.generic_utils import get_custom_objects

%matplotlib inline

np.random.seed(5)

#Loading the Number Recognition files provided by Kaggle. 



train =pd.read_csv("../input/train.csv")

test =pd.read_csv("../input/test.csv")

train.head()
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 5

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Showing an example of our training 

g = plt.imshow(X_train[5][:,:,0])
def swish(x):

    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})
#CNN Arcitecture with afromentioned "Swish Activation from Google"



model = Sequential()

get_custom_objects().update({'swish': Activation(swish )})

model.add(Conv2D(filters = 42, kernel_size = (5,5),padding = 'Same', 

                 activation ='swish', input_shape = (28,28,1)))

model.add(Conv2D(filters = 42, kernel_size = (5,5),padding = 'Same', 

                 activation ='swish'))

model.add(MaxPool2D(pool_size=(5,5)))

model.add(Dropout(0.50))

model.add(Conv2D(filters = 84, kernel_size = (3,3),padding = 'Same', 

                 activation ='swish'))

model.add(Conv2D(filters = 84, kernel_size = (3,3),padding = 'Same', 

                 activation ='swish'))

model.add(MaxPool2D(pool_size=(5,5), strides=(5,5)))

model.add(Dropout(0.50))

model.add(Flatten())

model.add(Dense(256, activation = "swish"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "sigmoid"))
optimizer= adam(lr=0.001,epsilon=1e-08,decay=0.0)

model.compile(optimizer =optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs = 30 

batch_size = 50

history_da = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

          validation_data = (X_val, Y_val), verbose = 2)



fig, ax = plt.subplots(2,1)

ax[0].plot(history_da.history['loss'], color='b', label="Training loss")

ax[0].plot(history_da.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history_da.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history_da.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Results

results = model.predict(test)

# Picking the ones with highest probability 

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("sub_Darawsheh.csv",index=False)