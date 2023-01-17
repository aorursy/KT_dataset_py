%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ipywidgets import  interact, interactive, fixed, interact_manual,FloatSlider
uniform = lambda x: (np.abs(x) <= 1) and 1/2 or 0
triangle = lambda x: (np.abs(x) <= 1) and  (1 - np.abs(x)) or 0
gaussian = lambda x: (1.0/np.sqrt(2*np.pi))* np.exp(-.5*x**2) 
# some sample kernel that can be run with a parzen window
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
for i,h in enumerate([.4,.9,1,4]):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(221 + i)
    plt.plot([triangle(ln/h)  for ln in np.arange(-10, 10, 0.1)],label="triangle")
    plt.plot([gaussian(ln/h)  for ln in np.arange(-10, 10, 0.1)],label="gaussian")
    plt.plot([uniform(ln/h)  for ln in np.arange(-10, 10, 0.1)],label="uniform")  
    plt.legend()
    plt.title('parzen windows normal: h: %f' % (h))
plt.show()
# A larger value of h samples over a larger region
plt.rcParams['figure.figsize'] = [40, 5]
inp = np.array([np.random.normal(0, 1, 200) + np.random.rand(200) * 4,np.random.normal(5, 2, 200)+ np.random.rand(200),np.random.normal(10, 1, 200)+ np.random.rand(200)]).flatten()
plt.hist(inp, bins=100);
plt.show()
for i,h in enumerate([.05,.4,1,4]):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(221 + i)
    plt.plot([(1.0/(len(inp)*h))*np.sum([triangle((ln - d)/h) for d in inp]) for ln in np.arange(0, 20, 0.1)],label="triangle")
    plt.plot([(1.0/(len(inp)*h))*np.sum([gaussian((ln - d)/h) for d in inp]) for ln in np.arange(0, 20, 0.1)],label="gaussian")
    plt.plot([(1.0/(len(inp)*h))*np.sum([uniform((ln - d)/h) for d in inp]) for ln in np.arange(0, 20, 0.1)],label="uniform")
    plt.legend()
    plt.title('parzen windows: h: %f' % (h))
    
plt.tight_layout()
plt.show()
# applying the window over a random distributio. 
# A large value of h will obscure a lot of major features of the structure while a small h is highly suceptible to noise.
inp1 = np.array([np.random.normal(0, 1, 200)]).flatten()
for i,h in enumerate([.05,.4,1,4]):
    plt.rcParams['figure.figsize'] = [20, 10]
    plt.subplot(221 + i)
    plt.plot([(1.0/(len(inp)*h))*np.sum([triangle((ln - d)/h) for d in inp1]) for ln in np.arange(-10, 10, 0.1)],label="triangle")
    plt.plot([(1.0/(len(inp)*h))*np.sum([gaussian((ln - d)/h) for d in inp1]) for ln in np.arange(-10, 10, 0.1)],label="gaussian")
    plt.plot([(1.0/(len(inp)*h))*np.sum([uniform((ln - d)/h) for d in inp1]) for ln in np.arange(-10, 10, 0.1)],label="uniform")
    plt.legend()
    plt.title('parzen windows normal: h: %f' % (h))
    
plt.show()
# running on a simlar but single normal distribution. 
# Read the CSV input file and show first 5 rows
df_train = pd.read_csv('../input/train.csv')
df_train.head(5)
# We can't do anything with the Name, Ticket number, and Cabin, so we drop them.
df_train = df_train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
# To make 'Sex' numeric, we replace 'female' by 0 and 'male' by 1
df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(int) 
# We replace 'Embarked' by three dummy variables 'Embarked_S', 'Embarked_C', and 'Embarked Q',
# which are 1 if the person embarked there, and 0 otherwise.
df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)
df_train = df_train.drop('Embarked', axis=1)
# We normalize the age and the fare by subtracting their mean and dividing by the standard deviation
age_mean = df_train['Age'].mean()
age_std = df_train['Age'].std()
df_train['Age'] = (df_train['Age'] - age_mean) / age_std

fare_mean = df_train['Fare'].mean()
fare_std = df_train['Fare'].std()
df_train['Fare'] = (df_train['Fare'] - fare_mean) / fare_std
# In many cases, the 'Age' is missing - which can cause problems. Let's look how bad it is:
print("Number of missing 'Age' values: {:d}".format(df_train['Age'].isnull().sum()))

# A simple method to handle these missing values is to replace them by the mean age.
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
# With that, we're almost ready for training
df_train.head()
# Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and test set
x_train = df_train.drop('Survived', axis=1).values
y_train = [[(value == i) * 1 for i in range(0,2)] for value in df_train['Survived'].values]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# uniform_tf = lambda x: (tf.math.abs(x) <= 1) and 1/2 or 0
# triangle_tf = lambda x: (np.abs(x) <= 1) and  (1 - np.abs(x)) or 0
gaussian_tf = lambda x: (1.0/tf.sqrt(2*np.pi))* tf.exp(-.5*x**2) 
def _pattern(input,name,feature_count,h):
    with tf.variable_scope(name) as scope:
        bias = tf.get_variable('bias',[feature_count, 1],initializer=tf.constant_initializer(0))
        bandwidth = tf.constant(1.0/(h * feature_count))
        return tf.multiply(tf.reduce_sum(tf.map_fn(lambda x: (gaussian_tf(x)/h),input + tf.transpose(bias)),axis=1),bandwidth)

tf.reset_default_graph() 
# N number of traning example with a 28*28 size image
inputs = tf.placeholder(tf.float32, shape=(None,x_train.shape[1]), name='inputs')
# 0-2 survived or perished
labels = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

survive = _pattern(inputs,'survived',x_train.shape[1],.2)
perished = _pattern(inputs,'perished',x_train.shape[1],.2)
result = tf.stack([survive,perished],axis=1)

# Loss function and optimizer
lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=labels))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

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
performance = []
for learning_rate in [0.05, 0.01]:
    for epoch in range(200):
        avg_cost = 0.0

        # For each epoch, we go through all the samples we have.
        for i in range(0,x_train.shape[0]):
            # Finally, this is where the magic happens: run our optimizer, feed the current example into X and the current target into Y
            _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate, 
                                                          inputs: [x_train[i]],
                                                          labels: [y_train[i]]})
            avg_cost += c
        avg_cost /= x_train.shape[0]    
        performance += [accuracy.eval(feed_dict={inputs: x_test, labels: y_test})]
        
        # Print the cost in this epcho to the console.
        if epoch % 10 == 0:
            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, avg_cost))
acc_train = accuracy.eval(feed_dict={inputs: x_train, labels: y_train})
print("Train accuracy: {:3.2f}%".format(acc_train*100.0))

acc_test = accuracy.eval(feed_dict={inputs: x_test, labels: y_test})
print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))
plt.plot(performance,label='performance')
plt.legend()
print('max performance:',max(performance))
