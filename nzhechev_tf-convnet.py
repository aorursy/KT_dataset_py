# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import os



input_folder = "../input"



train_data  = pd.read_csv(os.path.join(input_folder, "train.csv"))

                          

train_images = train_data.iloc[:,1:].values

train_labels = train_data.iloc[:,:1].values    



print(train_images.shape, train_labels.shape)
import matplotlib.pyplot as plt

%matplotlib inline



#handy function to plot image with its label

def plot_image(images, labels, index):

    plt.imshow(images [index].reshape(28, 28), cmap="Greys", interpolation="None")

    plt.title(labels [index])
plot_image(train_images, train_labels, 0)
import tensorflow as tf



class TF_CNN:

    def __init__(self, conv_layers, layers, reg=1e-3, learning_rate = 0.1):            

        self.x = tf.placeholder(tf.float32, shape = [None, 28 * 28])

        self.y = tf.placeholder(tf.float32, shape = [None, 10])



        depth = 1

        conv_X = tf.reshape(self.x, [-1,28,28,1])

        for conv_def in conv_layers:

            size = conv_def ["size"]

            filters = conv_def ["filters"]

                        

            conv_W = self.weights([size, size, depth, filters])

            conv_b = self.biases([filters])

            

            output = tf.nn.relu(self.conv2d(conv_X, conv_W) + conv_b)

            output = self.max_pool_2x2(output)

            

            conv_X = output

            depth = filters

        

        size = int(28 / (2**len(conv_layers)))        

        input_size = size * size * filters

        X = tf.reshape(conv_X, [-1, input_size])

        for count in layers:

            W = self.weights([input_size, count])

            b = self.biases([count])                             

            

            output = tf.matmul(X, W) + b

            

            X = tf.nn.relu(output)         

            

            input_size = count

        



        self.scores = output

        

        self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.y) )

        

        self.sess = tf.Session()                 

            

        optimizer = tf.train.AdamOptimizer(learning_rate)

        self.training_step = optimizer.minimize(self.loss)

        

        self.sess.run( tf.global_variables_initializer() )

    

    def weights(self, shape):        

        return tf.Variable( tf.truncated_normal(shape, stddev=0.1) )

    

    def biases(self, shape):        

        return tf.Variable( tf.constant(0.1, shape=shape) )  

    

    def conv2d(self, x, W):

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    

    def max_pool_2x2(self, x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  

    def train(self, inputs, targets):

        _, loss = self.sess.run([self.training_step, self.loss], {self.x: inputs, self.y: targets})

        

        return loss

        

    def query(self, inputs):

        return self.sess.run(self.scores, {self.x: inputs})    

        

    def get_accuracy(self, inputs, targets):

        correct_prediction = tf.equal(tf.argmax(self.scores,1), tf.argmax(self.y,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        

        return self.sess.run(accuracy, {self.x: inputs, self.y: targets})
#convert to one-hot vectors

def one_hot(labels, num_classes):

    y = np.zeros((labels.shape [0], num_classes))

    y [np.arange(labels.shape [0]), labels.flatten()] = 1

    

    return y



def prepare_images(images):

    inputs = images.astype(float)

    inputs -= 127.5

    inputs /= 127.5

    

    return inputs



def train_validation_split(inputs, targets, ratio = 0.8):

    data_size = inputs.shape[0]

    p = np.random.permutation(data_size)

    

    train_size = int(data_size * ratio)    

    

    ti = p [:train_size]

    tv = p [train_size:]

    

    return inputs [ti], targets [ti], inputs [tv], targets [tv]



def random_batch(inputs, targets, size = 100):

    data_size = inputs.shape[0]

    p = np.random.permutation(data_size)

    

    i = p [:size]

    

    return inputs [i], targets [i]

    
inputs = prepare_images(train_images)

targets = one_hot(train_labels, 10)



print("Inputs: ", inputs.shape)

print("Targets: ", targets.shape)



train_inputs, train_targets, validation_inputs, validation_targets = train_validation_split(inputs, targets)



print("Train inputs:", train_inputs.shape, ", targets: ", train_targets.shape)

print("Validation inputs:", validation_inputs.shape, ", targets: ", validation_targets.shape)
def batch_iterator(inputs, targets, batch_size=1):

    size = inputs.shape [0]

        

    start = 0

    while start < size:

        end = min(start + batch_size, size)

                

        yield inputs [start:end], targets [start:end]

        

        start = end   
import time



def train_cycle(nn, inputs, targets, epochs_count = 1, batch_size=64, dump_ratio = 0.1, validation_size=1000):    

    dump_period = (epochs_count * inputs.shape[0] / batch_size) * dump_ratio    



    start_time = time.time()

    steps = 0

    for i in range(epochs_count):

        for X, y in batch_iterator(inputs, targets, batch_size):

            loss = nn.train(X, y)

            

            if steps % dump_period == 0:

                print("Loss after step %d is %.3f" % (steps, loss))

                

            steps += 1                       

            

        accuracy = nn.get_accuracy( *random_batch(validation_inputs, validation_targets, validation_size) )

        print("Accuracy after epoch %d: %.4f" % (i + 1, accuracy))

       

        

    elapsed_time = time.time() - start_time

    print("%d training steps took %.1f seconds (%.3f seconds/epochs)" % (steps, elapsed_time, \

                                                                        elapsed_time / steps))      

        

    return accuracy
conv_layers = [{"size":5, "filters":16}, {"size":5, "filters":32}]

layers = [100, 10]

nn = TF_CNN(conv_layers, layers, reg = 1e-5, learning_rate = 0.001)



train_cycle(nn, train_inputs, train_targets, epochs_count = 15)
index = np.random.randint(0, inputs.shape [0])

p = np.argmax(nn.query(inputs [index:index + 1]))



print("Predicted:", p)

plot_image(train_images, train_labels, index)
test_data  = pd.read_csv(os.path.join(input_folder, "test.csv"))



test_images = test_data.values.astype('float32')

#train_images = data.iloc[:,1:].values



print(test_data.shape)
plt.imshow(test_images [254].reshape(28, 28), cmap="Greys", interpolation="None")
def chunks_iterator(data, size):    

    for i in range(0, data.shape [0], size):

        yield data [i:i + size]     

labels = []

for chunk in chunks_iterator(test_data, 100):

    prob = nn.query(chunk)

    labels.extend( np.argmax(prob, axis=1) )



submission = pd.DataFrame({"ImageId": list(range(1,len(labels)+1)), "Label": labels})

submission.to_csv("output.csv", index=False, header=True)



print("Submission saved!")