import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
%matplotlib inline

print('TensorFlow Version:', tf.__version__)
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
y_train = train['label']
train.drop('label',axis=1,inplace = True)
X_train = np.array(np.reshape(train, (train.shape[0], 784))/255)
y_train = tf.keras.utils.to_categorical(y_train)
X_test = np.array(np.reshape(test, (test.shape[0], 784))/255)
print("X_train shape :",X_train.shape)
print("y_train shape :",y_train.shape)
print("X_test shape :",X_test.shape)

# inputs = Input(shape=(784,)),  
model = keras.Sequential([
    tf.keras.layers.Dense(200, activation = 'tanh' , input_shape =(784,)),
    tf.keras.layers.Dense(200, activation = 'tanh'),
    tf.keras.layers.Dense(200, activation = 'tanh'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
model.compile(optimizer='adam', 
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    X_train,y_train, 
    epochs=50, 
    steps_per_epoch=50,
)
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))


output = pd.DataFrame({'ImageId' : np.arange(1,len(pred)+1) , 'Label': pred })
output.to_csv('my_submission.csv',index =False)
