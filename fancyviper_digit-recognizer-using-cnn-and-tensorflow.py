import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data() #automatically train data into 70,30 no need of sklearn split
plt.imshow(x_train[0],cmap=plt.cm.binary)

plt.show()
x_train[0]
#Scalling,-normalization,standarization

#present normalization  features scales to (0-1)
x_train=tf.keras.utils.normalize(x_train,axis=1)

x_test=tf.keras.utils.normalize(x_test,axis=1)
x_train[0]
model=tf.keras.models.Sequential()# a feed forward model

model.add(tf.keras.layers.Flatten())#takes 28x28 and make it to 1x784 matrix

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))# a simple fully connected layer

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))#acitvation func-if it crosses threshold value activates ,relu trains(0-9),sigmoid train(0 or 1)



model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # takes multiple values trained in dense-layer

model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])#calculate the error to minimize the loss

model.fit(x_train,y_train,epochs=10)
val_loss,val_acc=model.evaluate(x_test,y_test)
val_loss
val_acc
model.save("firstModel.model")
new_model = tf.keras.models.load_model("firstModel.model")
prediction = new_model.predict(x_test)
prediction[0]
def show_numbers(i):

  plt.imshow(x_test[i])

  print("The Number shown in the image is:   ",np.argmax(prediction[i]), end='\n')
show_numbers(30)
plt.imshow(x_test[0],cmap=plt.cm.binary)

plt.show()
np.argmax(prediction[0])