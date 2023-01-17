import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import tensorflow as tf
import matplotlib.pyplot as plt
dataset = datasets.load_breast_cancer()
dataset.keys()
print(dataset['DESCR'])
dataset['data']
#dataset['feature_names']
#dataset['target']
#dataset['target_names']
import pandas as pd
df = pd.DataFrame(np.c_[dataset['target'],dataset['data']], columns=np.r_[['target'],dataset['feature_names']])
df.head(10)
X_all = np.c_[np.ones([dataset['data'].shape[0],1]), dataset['data']]
X_all = scale(X_all) # Scale the data - this is important!
y_all = dataset['target'].reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
tf.reset_default_graph()
tf.set_random_seed(42)
X = tf.placeholder(dtype=tf.float32, shape=(None,X_train.shape[1]), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, y_train.shape[1]), name='y')
theta = tf.get_variable("theta", [X_train.shape[1],1], dtype=tf.float32)
logits = tf.matmul(X,theta, name='logits')
y_proba = tf.sigmoid(logits, name='y_proba')
error = tf.losses.log_loss(y, y_proba)
learning_rate = 0.01

# Using gradient descent
#gradients = tf.gradients(error, [theta])[0]
#training_op = tf.assign(theta, theta - learning_rate * gradients)

# Alternatively, we could use an out-of-the box gradient descent optimizer:
#optimizer = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer')

# Or even better, a more advanced optimizer, such as Adam
optimizer = tf.train.AdamOptimizer(learning_rate,name='optimizer')

#The training operation is then:
training_op = optimizer.minimize(error)
epochs = 10000
train_errs = np.ones([epochs])
test_errs = np.ones([epochs])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    train_errs[0] = error.eval(feed_dict={X: X_train, y: y_train})
    test_errs[0] = error.eval(feed_dict={X: X_test, y: y_test}) 
    print("Epoch",0,"Train Error =",np.round(train_errs[0],6),"Test Error =",np.round(test_errs[0],6))
    for epoch in range(epochs):
        train_errs[epoch] = error.eval(feed_dict={X: X_train, y: y_train})
        test_errs[epoch] = error.eval(feed_dict={X: X_test, y: y_test})                         
        if (epoch+1) % 1000 == 0:
            print("Epoch",epoch+1,"Train Error =",np.round(train_errs[epoch],6),"Test Error =",np.round(test_errs[epoch],6))
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
fig, ax = plt.subplots(1)
ax.plot(range(epochs), train_errs, label='Training error')
ax.plot(range(epochs), test_errs, label='Test error')
ax.set(xlabel='epoch', ylabel='error')
ax.legend();
# Overwrite into code above and re-run all.
#reg_lambda = 0.01 #tf.Variable(0.1, name="reg_lambda")
#error = tf.losses.log_loss(y, y_proba) + reg_lambda / 2 * tf.reduce_sum(tf.multiply(theta,theta))