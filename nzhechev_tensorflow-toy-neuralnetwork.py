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



train_data  = pd.read_csv("../input/train.csv")

train_images = train_data.iloc[:,1:].values

train_labels = train_data.iloc[:,:1].values    



print(train_images.shape, train_labels.shape)
import matplotlib.pyplot as plt

%matplotlib inline



#handy function to plot image with its label

def plot_image(images, labels, index):

    plt.imshow(images [index].reshape(28, 28), cmap="Greys", interpolation="None")

    plt.title(labels [index])
plot_image(train_images, train_labels, 1)
import tensorflow as tf



class TFNeuralNetwork:

    def __init__(self, layers):

        self.layers_count = len(layers)

        

        self.x = tf.placeholder(tf.float32, shape = [None, layers[0]])

        self.y = tf.placeholder(tf.float32, shape = [None, layers[self.layers_count - 1]])

                 

        self.weights = [None] * (self.layers_count - 1)

        self.biases = [None] * (self.layers_count - 1)

        

        outputs = [None] * (self.layers_count - 1)

        for i in range(self.layers_count - 1):

            in_dim = layers [i]

            out_dim = layers [i + 1]

            

            W = tf.Variable(tf.random_normal([in_dim, out_dim], stddev = np.sqrt(2.0/in_dim)))

            b = tf.Variable(tf.zeros([out_dim]))

            

            prev_output = self.x if i == 0 else outputs [i - 1]

        

            output = tf.matmul(prev_output, W) + b

            outputs [i] = output if i == self.layers_count - 2 else tf.nn.relu(output)

            

        self.scores = outputs [-1]

        self.loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.y) )

        

        self.sess = tf.Session()

        

        self.sess.run( tf.global_variables_initializer() )

        

        self.learning_rate = tf.placeholder(tf.float32)

        

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.training_step = optimizer.minimize(self.loss)

        

    def train(self, inputs, targets, learning_rate = 0.5):

        self.sess.run(self.training_step, {self.x: inputs, self.y: targets, self.learning_rate : learning_rate})

        

    def query(self, inputs):

        return self.sess.run(self.scores, {self.x: inputs})    

    

    def get_loss(self, inputs, targets):

        return self.sess.run(self.loss, {self.x: inputs, self.y: targets})

    

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
inputs = prepare_images(train_images)

targets = one_hot(train_labels, 10)



print("Inputs: ", inputs.shape)

print("Targets: ", targets.shape)
def random_batch(inputs, targets, size=128):

     p = np.random.permutation(inputs.shape[0]) [:size]

     

     return inputs [p], targets [p]

    
nn = TFNeuralNetwork([784, 100, 10])
import time



training_steps = 10000

dump_period = training_steps * 0.1



start_time = time.time()

for i in range(training_steps):

    batch_inputs, batch_targets = random_batch(inputs, targets)

    nn.train(batch_inputs, batch_targets, 0.1)    

    

    if i % dump_period == 0:

        loss = nn.get_loss(inputs, targets)

        

        print("Loss after step %d is %.3f" % (i + 1, loss))

        

elapsed_time = time.time() - start_time

print("%d training steps took %.f seconds (%.3f seconds/step)" % (training_steps, elapsed_time, \

                                                                  elapsed_time / training_steps))



print("Accuracy: %.3f" % nn.get_accuracy(inputs, targets))
p = np.argmax(nn.query(inputs [0:1]))



print("Predicted:", p)

plot_image(train_images, train_labels, 0)
test_data  = pd.read_csv("../input/test.csv")



test_images = test_data.values.astype('float32')

#train_images = data.iloc[:,1:].values



print(test_data.shape)



plt.imshow(test_images [1].reshape(28, 28))
prob = nn.query(test_data)

labels = np.argmax(prob, axis=1)



submission = pd.DataFrame({"ImageId": list(range(1,len(labels)+1)), "Label": labels})

submission.to_csv("output.csv", index=False, header=True)



print("Submission saved!")