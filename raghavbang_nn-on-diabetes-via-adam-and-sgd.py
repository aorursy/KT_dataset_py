import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
df=pd.read_csv("../input/diabetes/diabetes.csv")
df.head()

x=df.drop("Outcome",axis=1)

from sklearn.preprocessing import scale
x=scale(x)
y=df["Outcome"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4,random_state=0)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(12, activation="relu", input_shape=(8,)),  
  tf.keras.layers.Dense(8, activation="relu"),
  tf.keras.layers.Dense(1,activation="sigmoid")
])
from tensorflow.keras.optimizers import Adam
model.compile(optimizer = Adam(lr=0.0022), 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])
history = model.fit(xtrain, 
                    ytrain,
                   epochs =500,batch_size=40,validation_data=(xtest,ytest))
acc=model.evaluate(xtest,ytest)
print(acc)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss =  history.history['loss']
val_acc=history.history["val_accuracy"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and validation  loss')
plt.legend()

plt.show()
model1 = tf.keras.Sequential([
  tf.keras.layers.Dense(12, activation="relu", input_shape=(8,)),  
  tf.keras.layers.Dense(8, activation="relu"),
  tf.keras.layers.Dense(1,activation="sigmoid")
])

model1.compile(optimizer =tf.keras.optimizers.SGD(
    learning_rate=0.1), 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])
history = model1.fit(xtrain, 
                    ytrain,
                   epochs =500,batch_size=40,validation_data=(xtest,ytest))
import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss =  history.history['loss']
val_acc=history.history["val_accuracy"]
val_loss=history.history["val_loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation  accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='validation Loss')
plt.title('Training and validation  loss')
plt.legend()

plt.show()
acc=model1.evaluate(xtest,ytest)
print(acc)
