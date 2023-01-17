%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual,FloatSlider
uniform = lambda x: (np.abs(x) <= 1) and 1/2 or 0
triangle = lambda x: (np.abs(x) <= 1) and  (1 - np.abs(x)) or 0
gaussian = lambda x: (1.0/np.sqrt(2*np.pi))* np.exp(-.5*x**2) 
plt.rcParams['figure.figsize'] = [20, 5]
plt.subplot(1, 3, 1)
plt.title('Uniform')
plt.plot([uniform(i) for i in np.arange(-2, 2, 0.1)])

plt.subplot(1, 3, 2)
plt.title('Triangular')
plt.plot([triangle(i) for i in np.arange(-2, 2, 0.1)])

plt.subplot(1, 3, 3)
plt.title('Gaussian')
plt.plot([gaussian(i) for i in np.arange(-2, 2, 0.1)])
plt.show()
plt.rcParams['figure.figsize'] = [20, 5]
plt.hist(np.array([np.random.normal(0, 1, 200) + np.random.rand(200) * 4,np.random.normal(5, 2, 200)+ np.random.rand(200),np.random.normal(10, 1, 200)+ np.random.rand(200)]).flatten(), bins=100);
@interact(h=FloatSlider(min=.1,max=10,step=.01,value=30))
def distributions(h):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.plot([triangle(ln/h)  for ln in np.arange(-10, 10, 0.1)])
    plt.plot([gaussian(ln/h)  for ln in np.arange(-10, 10, 0.1)])
    plt.plot([uniform(ln/h)  for ln in np.arange(-10, 10, 0.1)])  

inp = np.array([np.random.normal(0, 1, 200) + np.random.rand(200) * 4,np.random.normal(5, 2, 200)+ np.random.rand(200),np.random.normal(10, 1, 200)+ np.random.rand(200)]).flatten()
@interact_manual(h=FloatSlider(min=.1,max=3,step=.01,value=30))
def kdf(h):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.plot([(1.0/(len(inp)*h))*np.sum([triangle((ln - d)/h) for d in inp]) for ln in np.arange(0, 20, 0.1)],label="triangle")
    plt.plot([(1.0/(len(inp)*h))*np.sum([gaussian((ln - d)/h) for d in inp]) for ln in np.arange(0, 20, 0.1)],label="gaussian")
    plt.plot([(1.0/(len(inp)*h))*np.sum([uniform((ln - d)/h) for d in inp]) for ln in np.arange(0, 20, 0.1)],label="uniform")
    plt.legend()
inp1 = np.array([np.random.normal(0, 1, 200)]).flatten()
@interact_manual(h=FloatSlider(min=.1,max=3,step=.01,value=30))
def kdf(h):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.plot([(1.0/(len(inp)*h))*np.sum([triangle((ln - d)/h) for d in inp1]) for ln in np.arange(-10, 10, 0.1)],label="triangle")
    plt.plot([(1.0/(len(inp)*h))*np.sum([gaussian((ln - d)/h) for d in inp1]) for ln in np.arange(-10, 10, 0.1)],label="gaussian")
    plt.plot([(1.0/(len(inp)*h))*np.sum([uniform((ln - d)/h) for d in inp1]) for ln in np.arange(-10, 10, 0.1)],label="uniform")
    plt.legend()
import tensorflow as tf
from sklearn.model_selection import train_test_split


train = pd.read_csv('../input/train.csv')
labels = train.loc[:, train.columns == 'label'].values.flatten()
images = train.loc[:, train.columns != 'label'].values
    
x_train, x_test, y_train, y_test = train_test_split(images,labels, test_size=0.33, random_state=42)
# create a boolean matrix of the correct answers
y_train_labels = [[(value == i) * 1 for i in range(0,10)] for value in y_train]
y_test_labels = [[(value == i) * 1 for i in range(0,10)] for value in y_test]
# uniform_tf = lambda x: (tf.math.abs(x) <= 1) and 1/2 or 0
# triangle_tf = lambda x: (np.abs(x) <= 1) and  (1 - np.abs(x)) or 0
gaussian_tf = lambda x: (1.0/tf.sqrt(2*np.pi))* tf.exp(-.5*x**2) 
def _pattern(input,name,weight_shape,h,droput):
    with tf.variable_scope(name) as scope:
        w1 = tf.get_variable('weight',weight_shape,initializer=tf.initializers.truncated_normal(0,1))
        b1 = tf.get_variable('bias',[weight_shape[0], 1],initializer=tf.constant_initializer(0))
        layer = tf.nn.dropout(tf.add(tf.matmul(w1, tf.transpose(inputs)),b1),droput)
        bandwidth = tf.constant(1.0/(h * weight_shape[0]))
        return tf.multiply(tf.reduce_sum(tf.map_fn(lambda x: (gaussian_tf(x)/h),layer),axis=0),bandwidth)
# shape of image test samples 
# 60000 sample with a 28X28 size image flatten 784
x_train.shape

tf.reset_default_graph() 
# N number of traning example with a 28*28 size image
inputs = tf.placeholder(tf.float32, shape=(None,x_train.shape[1]), name='inputs')
# 0-9
labels = tf.placeholder(tf.float32, shape=(None, 10), name='labels')
# droupout probability
keep_prob = tf.placeholder(tf.float32,shape=(), name='keep_prob')

zero = _pattern(inputs,'zero',[20,x_train.shape[1]],.4,keep_prob)
one = _pattern(inputs,'one',[20,x_train.shape[1]],.4,keep_prob)
two = _pattern(inputs,'two',[20,x_train.shape[1]],.4,keep_prob)
three = _pattern(inputs,'three',[20,x_train.shape[1]],.4,keep_prob)
four = _pattern(inputs,'four',[20,x_train.shape[1]],.4,keep_prob)
five = _pattern(inputs,'five',[20,x_train.shape[1]],.4,keep_prob)
six = _pattern(inputs,'six',[20,x_train.shape[1]],.4,keep_prob)
seven = _pattern(inputs,'seven',[20,x_train.shape[1]],.4,keep_prob)
eight = _pattern(inputs,'eight',[20,x_train.shape[1]],.4,keep_prob)
nine = _pattern(inputs,'nine',[20,x_train.shape[1]],.4,keep_prob)
result = tf.stack([zero, one,two,three,four,five,six,seven,eight,nine],axis=1)

# Loss function and optimizer
lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Prediction
pred_label = tf.argmax(result,1)
correct_prediction = tf.equal(pred_label, tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Configure GPU not to use all memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Start a new tensorflow session and initialize variables
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

# This is the main training loop: we train for 50 epochs with a learning rate of 0.05 and another 
# 50 epochs with a smaller learning rate of 0.01
performance_drouput = []
for learning_rate in [0.05, 0.01]:
    for epoch in range(200):
        avg_cost = 0.0

        # For each epoch, we go through all the samples we have.
        for i in range(0,x_train.shape[0]):
            # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
            _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate, 
                                                          inputs: [x_train[i]],
                                                          labels: [y_train_labels[i]],
                                                         keep_prob:.5})
            
            avg_cost += c
        avg_cost /= x_train.shape[0]    
        performance_drouput += [accuracy.eval(feed_dict={inputs: x_test, labels: y_test,keep_prob:1})]
        
        # Print the cost in this epcho to the console.
        if epoch % 10 == 0:
            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))
acc_train = accuracy.eval(feed_dict={inputs: x_train, labels: y_train_labels,keep_prob:1})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_test = accuracy.eval(feed_dict={inputs: x_test, labels: y_test_labels,keep_prob:1})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))
sess.close()
# Start a new tensorflow session and initialize variables
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
# This is the main training loop: we train for 50 epochs with a learning rate of 0.05 and another 
# 50 epochs with a smaller learning rate of 0.01
performance_no_droupout = []
for learning_rate in [0.05, 0.01]:
    for epoch in range(200):
        avg_cost = 0.0

        # For each epoch, we go through all the samples we have.
        for i in range(0,x_train.shape[0]):
            # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
            _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate, 
                                                          inputs: [x_train[i]],
                                                          labels: [y_train[i]],
                                                         keep_prob:1})
            avg_cost += c
        avg_cost /= x_train.shape[0]    
        performance_no_droupout += [accuracy.eval(feed_dict={inputs: x_test, labels: y_test,keep_prob:1})]
        
        # Print the cost in this epcho to the console.
        if epoch % 10 == 0:
            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))
acc_train = accuracy.eval(feed_dict={inputs: x_train, labels: y_train,keep_prob:1})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_test = accuracy.eval(feed_dict={inputs: x_test, labels: y_test,keep_prob:1})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))
sess.close()
plt.plot(performance_drouput,label='with drouput')
plt.plot(performance_no_droupout,label='without dropout')
plt.legend()