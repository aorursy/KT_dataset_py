import numpy as np 
import tensorflow as tf
import os
print(os.listdir("../input"))
tf.enable_eager_execution()
class Linear(tf.keras.Model):
    def __init__(self):
        #print("__init___")
        super().__init__()
        self.dense = tf.keras.layers.Dense(1,activation=None,kernel_initializer=tf.zeros_initializer(),bias_initializer=tf.zeros_initializer())
    def call(self,input):
        #print(input)
        return self.dense(input)
X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
for e in range(100):
    with tf.GradientTape() as tape:
        y_pre = model(X)
        loss = tf.reduce_mean(tf.square(y-y_pre))
        grads = tape.gradient(loss,model.variables)
        optimizer.apply_gradients(zip(grads,model.variables))
print(model.variables)
