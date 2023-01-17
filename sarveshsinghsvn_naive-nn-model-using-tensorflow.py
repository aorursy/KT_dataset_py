import os
print(os.listdir("../input"))
import tensorflow as tf
import pandas as pd
import numpy as np
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#test the data sizes and 
print(train.shape)
Y = train["label"]
X = train.drop(labels = ["label"],axis = 1) 

print(Y.shape)
print(X.shape)
print(test.shape)
#depth = 10
#Y_encoded = tf.one_hot(Y,depth)
#print(Y_encoded)
nb_classes = 10
Y_encoded = np.eye(nb_classes)[Y]
print(Y_encoded)
print(Y_encoded.shape)
#split data into train and validation
X = X.values
test = test.values
Y_train = Y_encoded[0:38000,]
Y_val = Y_encoded[38000:42000,]
X_train = X[0:38000,:]
X_val = X[38000:42000,:]

print(Y_train.shape)
print(X_train.shape)
class Model:
    def __init__(self,X, Y):
        self.X = tf.placeholder(dtype=tf.float32, shape=X.shape, name="FEED_X")
        self.Y = tf.placeholder(dtype=tf.float32, shape=Y.shape, name="FEED_Y")
        self._prediction = None
        self._optimize = None
        self._error = None
        
    #Neurons in first and last layers are fixed, one hidden layer NN
    def modelDimInit(self,neurons_each_layer,batch_size):
        self.neurons_each_layer = neurons_each_layer
        self.batch_size = batch_size
        
    #shape of weights W = []
    def prediction(self):
        if not self._prediction:
            layer1_size = tf.cast(self.X.shape[1],tf.int64)
            self.b1 = tf.zeros(self.neurons_each_layer,dtype=tf.float32, name = "BIAS1")
            self.w1 = tf.Variable((tf.truncated_normal([layer1_size, self.neurons_each_layer])), dtype=tf.float32, name = "W1")
            layer2_neurons = tf.add(tf.matmul(self.X, self.w1), self.b1)

            print(layer2_neurons.shape)
            
            layer2_activation = tf.sigmoid(layer2_neurons, name = "ACTIVATION_LAYER_1")
            
            layer3_size = tf.cast(self.Y.shape[1], tf.int64)
            print(layer3_size)
            self.b2 = tf.zeros(layer3_size,dtype=tf.float32, name = "BIAS_2")
            self.w2 = tf.Variable((tf.truncated_normal([self.neurons_each_layer, layer3_size])),dtype=tf.float32, name = "W2")
            
            layer3_neurons = tf.add(tf.matmul(layer2_activation,self.w2),self.b2, name = "ACTIVATION_LAYER_2")
            print(layer3_neurons.shape)
            self._prediction = tf.nn.softmax(layer3_neurons, name = "OUTPUT_LAYER")
        return self._prediction
    
    def optimize(self):
        if not self._optimize:
            self.cross_entropy = tf.reduce_sum(tf.losses.log_loss(self.Y, self._prediction))
            self.optimizer = tf.train.GradientDescentOptimizer(0.3)
            self._optimize = self.optimizer.minimize(self.cross_entropy)
        return self._optimize
    
    def error(self):
        if not self._error:
            argmax_pred = tf.argmax(self._prediction, 1)
            argmax_true = tf.argmax(self.Y, 1)
            print (argmax_pred)
            print(argmax_true)
            self.compared = tf.cast(tf.equal(argmax_pred,argmax_true),dtype=tf.int32)
            correct_prediction = tf.reduce_sum(self.compared)
            self._error = tf.subtract(tf.cast(1.0,dtype=tf.float64), tf.divide(correct_prediction, self.batch_size))
        return self._error
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    batch_size = 1000
    print(X_train[0:batch_size,:].shape)
    model_inst = Model(X_train[0:batch_size,:], Y_train[0:batch_size,:])
    model_inst.modelDimInit(340,batch_size)
    
    pred = model_inst.prediction()
    opt = model_inst.optimize()
    error = model_inst.error()
    
    sess.run(model_inst.w1.initializer)
    sess.run(model_inst.w2.initializer)
    
    for i in range(100):
        train_error = 0
        val_error = 0
        for j in range(38):
            sess.run(pred, feed_dict={model_inst.X:X_train[j*batch_size:(j+1)*batch_size,:],model_inst.Y:Y_train[j*batch_size:(j+1)*batch_size,:]})
            train_error += sess.run(error, feed_dict={model_inst.X:X_train[j*batch_size:(j+1)*batch_size,:],model_inst.Y:Y_train[j*batch_size:(j+1)*batch_size,:]})
            sess.run(opt, feed_dict={model_inst.X:X_train[j*batch_size:(j+1)*batch_size,:],model_inst.Y:Y_train[j*batch_size:(j+1)*batch_size,:]})
        for k in range(4):
            val_error += sess.run(error, feed_dict={model_inst.X:X_val[k*batch_size:(k+1)*batch_size,:],model_inst.Y:Y_val[k*batch_size:(k+1)*batch_size,:]})
        print("train error = ",train_error/38 ," val error = ", val_error/4)
    
    Y_test = np.zeros((28000,10))
    for l in range(28):
        Y_test_batch = sess.run(pred,feed_dict={model_inst.X:test[l*batch_size:(l+1)*batch_size,:]})
        Y_test[l*batch_size:(l+1)*batch_size,:] = Y_test_batch
        

    print(Y_test)
    predictions = np.argmax(Y_test,axis=1)
    submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
    submissions.to_csv("DR.csv", index=False, header=True)   
        
