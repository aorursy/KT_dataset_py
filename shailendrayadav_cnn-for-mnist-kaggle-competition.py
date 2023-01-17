import keras

import tensorflow as tf

import numpy as np

import pandas as pd

#data 

test_df=pd.read_csv("../input/test.csv")

test_df.head()
train_df=pd.read_csv("../input/train.csv")

train_df.head()
# using training data and dividing it in features and labels

#features

#reshaping the same into pixel size

X=train_df.iloc[:,1:].values

X=X.reshape(len(X),28,28,1) # 28*28 pixels and 1 is for no color,else 3 for RGB primary colors

X.shape

#labels

y=train_df.iloc[:,:1].values #labels have values ranging 0-9

y.shape
#using to_categorical for labels to remove raltion ships amongst them

y=keras.utils.to_categorical(y,num_classes=10)

y.shape
#converting the data into np arrays

X=np.array(X)

y=np.array(y)
X.shape,y.shape
#normalize the feature to range between 0-255

X=X/255
#Convinusional neural networkusing sequenctial model

from keras.models import Sequential

model=Sequential()

#adding conenutional layers

from keras.layers import Convolution2D,MaxPooling2D, Dense,Flatten,Dropout #CN and max pooling

#CN layer has random size,dimension of conv

model.add(Convolution2D(32,(3,3),activation="relu",input_shape=(28,28,1)))

#max pooling to get max value  from each convenution.

model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.40))



model.add(Convolution2D(32,(3,3),activation="relu"))

#max pooling to get max value  from each convenution.

model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Dropout(0.40))

#fully connected network

model.add(Flatten()) #converts data to vector

#model.add(Dense(100))# adding 100 dense nodes

model.add(Dropout(0.50)) # remove or drop  bad weight 20%

model.add(Dense(10,activation="softmax")) #output layer for probability of each output to be 0-9

model.summary()
#compiling the model. loss used ascategorical_crossentropy as we are doing a classification problem.

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#training

history=model.fit(X,y,epochs=100,batch_size=128,validation_split=0.2)
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(history.history["loss"])

plt.plot(history.history["val_loss"])

plt.legend()
test_df.head()
X_test=test_df.values

X_test=X_test.reshape(len(X_test),28,28,1) # 28*28 pixels and 1 is for no color,else 3 for RGB primary colors

X_test.shape
#Normalize

#since the data is in range between 0 -255 lets normalize it as a better model practice

X_test=tf.keras.utils.normalize(X_test,axis=1)

X_test.shape  # itxs now range between 0-1
model.predict(X_test)
results = model.predict(X_test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("mnist_data_compertition2.csv",index=False)

#save the mdel

model.save("mnist_CNN_model")