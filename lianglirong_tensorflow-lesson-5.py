import numpy as np 
import tensorflow as tf
import os
import pickle
print(os.listdir("../input"))
tf.enable_eager_execution()
class DataLoader():
    def __init__(self):
        #print("__init__")
        with open("../input/mnist.pkl/mnist.pkl",mode="rb") as f:
            tr_d, va_d, te_d = pickle.load(f,encoding="bytes")
            train_x = tr_d[0]
            train_y = tr_d[1]
            self.train_x = np.array([np.reshape(X,(784)) for X in train_x])
            self.train_y = np.array([self.oneHot(y) for y in train_y])

            
            validation_x = va_d[0]
            validation_y = va_d[1]
            self.validation_x = np.array([np.reshape(X,(784)) for X in validation_x])
            self.validation_y = np.array([self.oneHot(y) for y in validation_y])
            
            test_x = te_d[0]
            test_y = te_d[1]
            self.test_x = np.array([np.reshape(X,(784)) for X in test_x])
            self.test_y = np.array([self.oneHot(y) for y in test_y])

    def oneHot(self,y):
        e = np.zeros(10)
        e[y] = 1
        return e
    
    def get_batch(self, batch_size):
        index = np.random.randint(0,len(self.train_x),batch_size)
        #print(index)
        #print(type(self.train_x))
        return self.train_x[index,:],self.train_y[index]
    
    def getTestData(self):
        return self.test_x,self.test_y
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=[5,5],padding="same",activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2)
        
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)
        
    def call(self,inputs):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
    def predict(self, inputs):
        logits = self(inputs)
        return tf.argmax(logits,axis=-1)
num_batches = 100
batch_size = 50
learning_rate = 0.001
model = CNN()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
checkpoint = tf.train.Checkpoint(myModel=model)

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        #print(y_logit_pred.shape)
        #print(y.shape)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=y_logit_pred)
        if(batch_index%10 == 0):
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
            checkpoint.save("model/model.ckpt")
        grads = tape.gradient(loss, model.variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
restore_model = CNN()
checkpoint = tf.train.Checkpoint(myModel=restore_model)
checkpoint.restore(tf.train.latest_checkpoint("model"))
test_x,test_y = data_loader.getTestData()
num_test_samples = test_x.shape[0]
test_y = tf.argmax(test_y,axis=-1)

predict_y = restore_model.predict(test_x).numpy()
print(predict_y.shape)
print(test_y.shape)


print("test accuracy: %f" % (sum(predict_y == test_y.numpy()) /num_test_samples ))


