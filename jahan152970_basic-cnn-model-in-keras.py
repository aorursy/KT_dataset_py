import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Model
from keras.layers import Conv2D,Dense,Flatten,Dropout, MaxPooling2D,BatchNormalization,LeakyReLU
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import random as rn

import os



# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

#g = sns.countplot(Y_train)

Y_train.value_counts()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Some examples
g = plt.imshow(X_train[10][:,:,0])
def CNNModel(model_num=None):#input dim: 28x28x1
    model = Sequential()
    model.add(Conv2D(32,(5,5),activation="relu",padding='SAME',input_shape=(28,28,1)))#output dim: 28*28*32
    model.add(Conv2D(32,(5,5),activation="relu"))#output dim: 24x24x32
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))#output dim: 14x14x32
    model.add(Conv2D(64,(5,5),activation="relu"))#10*10*64
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))#5*5*64
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model
digitRecon = CNNModel()
earlyStopping = EarlyStopping(monitor='acc',
                              patience=4)

digitRecon.compile(optimizer='adam',loss="categorical_crossentropy", metrics=["accuracy"])
digitRecon.fit(X_train,Y_train, epochs = 35,batch_size=64,callbacks=[earlyStopping])
# predict results
results = digitRecon.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
#results.head(2)
val=rn.randint(0,test.shape[0]-1)
plt.imshow(test[val][:,:,0])
plt.title("Predicted value:"+str(results[val]))
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
