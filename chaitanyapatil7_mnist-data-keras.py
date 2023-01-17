import pandas as pd
import tensorflow as tf

mnist_train = pd.read_csv('../input/train.csv')
mnist_test = pd.read_csv('../input/test.csv')

mnist_test.head()
Xtrain = mnist_train.drop('label',axis=1) #X
ytrain = mnist_train['label']             #y
import matplotlib.pyplot as plt
%matplotlib inline
Xtrain=Xtrain/255
Xtest = mnist_test
Xtest = Xtest/255
Xtrain = Xtrain.values.astype('float32')
Xtest = Xtest.values.astype('float32')
plt.imshow(Xtest[4].reshape(28,28),cmap=plt.cm.binary)











Xtrain.shape[0]
Xtrain = Xtrain.reshape(Xtrain.shape[0],28,28)
Xtrain.shape
Xtest = Xtest.reshape(Xtest.shape[0],28,28)
Xtest.shape

Xtrain[0]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
ytrain = ytrain.values.astype('uint8')
ytrain
model.fit(x=Xtrain,y=ytrain,batch_size=128,epochs=8)
plt.imshow(Xtest[0])
Xtest
plt.imshow(Xtest[27])
predictions = model.predict([Xtest])
import numpy as np

np.argmax(predictions[27])
len(predictions)
Label=[]
for array in predictions:
    Label.append(np.argmax(array))
Label
pred=pd.DataFrame({'Label':Label})
idx=pd.DataFrame({'ImageId':range(1,len(pred)+1)})

OUTPUT_RESULT="submission.csv"
submission=pd.concat([idx,pred],axis=1)
submission.to_csv(OUTPUT_RESULT,index=False)
submission
