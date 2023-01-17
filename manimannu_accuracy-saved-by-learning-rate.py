import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import keras # Neural nets API

import numpy as np # Linear algebra

import pandas as pd # Data manipulation.
# Load data into train and test pandas dataframe

train_df=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_df=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# view top 5 rows. 

train_df.head()
test_df.head() # view top 5 rows of test data.
# shape of both train and test dataset.

train_df.shape ,test_df.shape
# drop target (label) into new one

target=train_df["label"]

train_df.drop("label",axis=1,inplace=True)
train_df.head()
train_df=train_df/255 # normalize will work better with cnn

test_df=test_df/255 # from [0:255] to [0:1]
X_train=train_df.values.reshape(-1,28,28,1) # reshaping to keras convention (sample,height,width,color)

test=test_df.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical

y_train=to_categorical(target,num_classes=10) # one hot encoding
y_train[0] # view first label after OHE.
import matplotlib.pyplot as plt



plt.figure(figsize=(10,5))



for i in range(30):

    plt.subplot(3,10,i+1)

    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis("off")

plt.subplots_adjust(wspace=0,hspace=0)

plt.show()
# train test split data one for training one for vaildation.

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.10,random_state=42)
plt.imshow(X_train[0].reshape((28,28))) # plot
y_train[0] # result for above plot.
batch_size=128

num_classes=10

epochs=20

inputshape=(28,28,1)
from keras.models import Sequential # import sequential convention so we can add layer after other.

import keras

from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,BatchNormalization

model=Sequential()



# add first convolutional layer.

model.add(Conv2D(32,kernel_size=(5,5),activation="relu",input_shape=inputshape))

# add second convolutional layer

model.add(Conv2D(64,(3,3),activation="relu"))

          

# add maxpooling layer

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128,kernel_size=(5,5),activation="relu"))

# add second convolutional layer

model.add(Conv2D(128,(3,3),activation="relu"))



# add one drop layer

model.add(Dropout(0.25))



# add flatten layer

model.add(Flatten())



# add dense layer

model.add(Dense(256,activation="relu"))

model.add(Dense(128,activation="relu"))

          

# add another dropout layer

model.add(Dropout(0.5))



# add dense layer

model.add(Dense(num_classes, activation='softmax'))
# complile the model and view its architecture

model.compile(loss="categorical_crossentropy",  optimizer="Adam", metrics=['accuracy'])

model.summary()
# callbacks

from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping

reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.3, min_lr = 0.00001)

checkpoint = ModelCheckpoint('save_weights.h5', monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')

early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 10, verbose = 1, restore_best_weights = True)



callbacks = [reduce_learning_rate, checkpoint, early_stopping]
# train model

model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test),callbacks=callbacks)

accuracy=model.evaluate(X_test,y_test)
pred = model.predict_classes(test)

res = pd.DataFrame({"ImageId":list(range(1,28001)),"Label":pred})

res.to_csv("output.csv", index = False)