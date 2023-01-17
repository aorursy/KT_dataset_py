import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train.shape
y_train.shape
x_test.shape
y_test.shape
x_train[0]
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

x_train[0]
x_test[0]
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten()) # adding flatten curve
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #relu used for adding Non Linearity

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #adding dense layer
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # output as softmax activation method
model.compile(optimizer="adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ['accuracy']
              )
model.fit(x_train,y_train,epochs=10)
validation_loss,validation_accuracy = model.evaluate(x_test,y_test)
print(validation_loss,validation_accuracy)
model.save("firstModel.model")
new_model = tf.keras.models.load_model("firstModel.model")
prediction = new_model.predict(x_test)
prediction[0]
def show_numbers(i):
  plt.imshow(x_test[i])
  print("The Number shown in the image is:   ",np.argmax(prediction[i]), end='\n')
show_numbers(30)
