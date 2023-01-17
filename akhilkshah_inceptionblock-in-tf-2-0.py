import numpy as np
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Flatten,Dropout,MaxPool2D,Dense
train = pd.read_csv(os.path.join('/kaggle/input/fashionmnist/fashion-mnist_train.csv'))

train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]

train_x = train_x.to_numpy()
train_y = train_y.to_numpy()

train_y = np.eye(10)[train_y]

train_x = np.reshape(train_x,(train_x.shape[0],28,28,1))
train_y = np.reshape(train_y,(train_y.shape[0],10))

train_x = train_x/255
train_x = train_x.astype('float64')

print(train_x.shape)
print(train_y.shape)
test = pd.read_csv(os.path.join('/kaggle/input/fashionmnist/fashion-mnist_test.csv'))

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]

test_x = test_x.to_numpy()
test_y = test_y.to_numpy()

test_y = np.eye(10)[test_y]

test_x = np.reshape(test_x,(test_x.shape[0],28,28,1))
test_y = np.reshape(test_y,(test_y.shape[0],10))

test_x = test_x/255
test_x = test_x.astype('float64')

print(test_x.shape)
print(test_y.shape)
class InceptionNet(tf.keras.Model):
    
    def __init__(self):
        super(InceptionNet,self).__init__()
        
        self.conv1 = Conv2D(192,1,activation='relu',padding='same')
        
        # Inception Layers
        self.c_incep1_1 = Conv2D(64,1,activation='relu',padding='same')
        
        self.c_incep2_1 = Conv2D(96,1,activation='relu',padding='same')
        self.c_incep2_2 = Conv2D(128,3,activation='relu',padding='same')
        
        self.c_incep3_1 = Conv2D(16,1,activation='relu',padding='same')
        self.c_incep3_2 = Conv2D(32,5,activation='relu',padding='same')
        
        self.c_incep4_1 = MaxPool2D(pool_size=(3,3),strides=(1,1),padding='same')
        self.c_incep4_2 = Conv2D(32,1,activation='relu',padding='same')
        
        # Dense Layers
        self.flatten = Flatten()
        self.dense1 = Dense(20,activation='relu')
        self.dense2 = Dense(10,activation='softmax')
        
    def call(self,input_vector):
        
        a1 = self.conv1(input_vector)
        
        a2 = self.InceptionBlock(a1)
        
        a2 = self.flatten(a2)
        a3 = self.dense1(a2)
        a4 = self.dense2(a3)
        
        return a4
    
    def InceptionBlock(self,vector):
        
        block1 = self.c_incep1_1(vector)
        
        block2 = self.c_incep2_1(vector)
        block2 = self.c_incep2_2(block2)
        
        block3 = self.c_incep3_1(vector)
        block3 = self.c_incep3_2(block3)
        
        block4 = self.c_incep4_1(vector)
        block4 = self.c_incep4_2(block4)
                
        output = tf.concat([block1,block2,block3,block4], axis=3)
        
        return output
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
cce = tf.keras.losses.CategoricalCrossentropy()

def custom_loss(target,predicted):
    cost = cce(target,predicted)
    return cost
# Gradient Tapping

model = InceptionNet()

@tf.function
def grad(batch_x,batch_y):
    
    with tf.GradientTape() as t:
        pred = model(batch_x)
        cost = custom_loss(batch_y,pred)
        g = t.gradient(cost,model.trainable_variables)
        opt.apply_gradients(zip(g,model.trainable_variables))
        
        return cost
        
batch_size = 512
temp = int(train_x.shape[0]/batch_size)
for epoch in range(25):
    
    prev = 0
    nxt = batch_size
    c = 0
    
    for _ in range(temp):
        
        batch_x = train_x[prev:nxt]
        batch_y = train_y[prev:nxt]
        c += grad(batch_x,batch_y)
        
        prev = nxt
        nxt += batch_size
        
    print("Loss at epoch ", epoch, " - ", (c/temp).numpy())
#Acuracy
def get_accuracy(pred,target):
    prediction = tf.argmax(pred,1)
    temp = tf.argmax(target,1)
    equality = tf.equal(prediction,temp)
    
    return equality.numpy()
# Train Accuracy 
batch_size = 512
temp = int(train_x.shape[0]/batch_size)

prev = 0
nxt = batch_size

answer = []
for epoch in range(temp):
    
    answer.extend(get_accuracy(model(train_x[prev:nxt]),train_y[prev:nxt]))
    prev = nxt
    nxt += batch_size
    
accuracy = tf.reduce_mean(tf.cast(answer,tf.float32))
print("Accuracy - ", accuracy.numpy()*100)
# Test Accuracy 
batch_size = 512
temp = int(test_x.shape[0]/batch_size)

prev = 0
nxt = batch_size

answer = []
for epoch in range(temp):
    
    answer.extend(get_accuracy(model(test_x[prev:nxt]),test_y[prev:nxt]))
    prev = nxt
    nxt += batch_size
    
accuracy = tf.reduce_mean(tf.cast(answer,tf.float32))
print("Accuracy - ", accuracy.numpy()*100)
# Keras
model = InceptionNet()
model.compile(loss=custom_loss,optimizer=opt,metrics=['accuracy'])
model.fit(x=train_x,y=train_y,batch_size=512,epochs=10)
model.evaluate(x=test_x,y=test_y)