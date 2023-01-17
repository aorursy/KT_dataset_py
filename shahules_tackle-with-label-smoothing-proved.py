import tensorflow as tf



tf.keras.losses.binary_crossentropy(

    y_true, y_pred,

    from_logits=False,

    label_smoothing=0

)



train[np.where(y == 0)] = 0.1

train[np.where(y == 1)] = 0.9

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

plt.style.use('ggplot')

epochs=10
data= load_breast_cancer()
print("The dataset contains {} samples with {} features".format(data['data'].shape[0],data['data'].shape[1]))

train_X,test_X,train_y,test_y=train_test_split(data['data'],data['target'],random_state=77)
train_y[:40]=1
def model():

    inp = tf.keras.Input(shape=(30))

    

    x= tf.keras.layers.Dense(64,activation='relu')(inp)

    x=tf.keras.layers.Dense(32,activation='relu')(x)

    x=tf.keras.layers.Dense(1,'sigmoid')(x)

    

    model=tf.keras.Model(inp,x)

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

    

    
model=model()

model.summary()


history=model.fit(train_X,train_y,epochs=epochs)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.plot(np.arange(1,epochs+1),history.history['loss'],color='red',alpha=1)

plt.gca().set_xlabel("Epochs")

plt.gca().set_ylabel("loss")

plt.gca().set_title("Loss without Label smoothing")



plt.subplot(1,2,2)

plt.plot(np.arange(1,epochs+1),history.history['accuracy'],color='red',alpha=1)

plt.gca().set_xlabel("Epochs")

plt.gca().set_ylabel("accuracy")

plt.gca().set_title("accuracy without Label smoothing")





plt.show()
y_pre = model.predict(test_X)

print(accuracy_score(test_y,np.round(y_pre)))
def label_smoothing(y_true,y_pred):

    

     return tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=0.1)
def model():

    inp = tf.keras.Input(shape=(30))

    

    x= tf.keras.layers.Dense(64,activation='relu')(inp)

    x=tf.keras.layers.Dense(32,activation='relu')(x)

    x=tf.keras.layers.Dense(1,'sigmoid')(x)

    

    model=tf.keras.Model(inp,x)

    

    model.compile(optimizer='Adam',loss=label_smoothing,metrics=['accuracy'])

    return model

    
model=model()

history=model.fit(train_X,train_y,epochs=epochs)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.plot(np.arange(1,epochs+1),history.history['loss'],color='red',alpha=1)

plt.gca().set_xlabel("Epochs")

plt.gca().set_ylabel("loss")

plt.gca().set_title("Loss with Label smoothing")



plt.subplot(1,2,2)

plt.plot(np.arange(1,epochs+1),history.history['accuracy'],color='red',alpha=1)

plt.gca().set_xlabel("Epochs")

plt.gca().set_ylabel("accuracy")

plt.gca().set_title("accuracy with Label smoothing")





plt.show()
y_pre = model.predict(test_X)

print(accuracy_score(test_y,np.round(y_pre)))