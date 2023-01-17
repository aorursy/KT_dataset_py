import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
dataset = datasets.load_boston()
m = dataset['target'].shape[0]
n = dataset['data'].shape[1]

# Scale the data - this is important!
scaler = StandardScaler()
X_scaled = np.c_[np.ones([m,1]), scaler.fit_transform(dataset['data'])]
y_vals = dataset['target'].reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_vals, test_size=0.2, random_state=42)
tf.reset_default_graph()
tf.set_random_seed(42)
X = tf.placeholder(dtype=tf.float32, shape=(None,n+1), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None,1), name='y')
theta = tf.get_variable("theta", [n+1,1], dtype=tf.float32)
y_pred = tf.matmul(X,theta, name='y_pred')
error = tf.reduce_mean(tf.square(y_pred - y))
learning_rate = 0.05
#optimizer = tf.train.GradientDescentOptimizer(learning_rate,name='optimizer')
optimizer = tf.train.AdamOptimizer(learning_rate,name='optimizer')
training_op = optimizer.minimize(error)
epochs = 1000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    theta_start = tf.transpose(theta).eval()
    print("Epoch", 0, "Train Error =", error.eval(feed_dict={X: X_train, y: y_train}),"Test Error =", error.eval(feed_dict={X: X_test, y: y_test}))
    for epoch in range(epochs):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        if (epoch+1) % 100 == 0:
            print("Epoch", epoch+1, 
                  "Train Error =", error.eval(feed_dict={X: X_train, y: y_train}),
                  "Test Error =", error.eval(feed_dict={X: X_test, y: y_test}))
    theta_end = tf.transpose(theta).eval()
X_all = np.c_[np.ones([m,1]), dataset['data']]
y_all = dataset['target']
theta_norm = np.linalg.inv(X_all.T.dot(X_all)).dot(X_all.T).dot(y_all)
norm_error = np.sum(np.square(y_all - X_all.dot(theta_norm))) / y_all.shape[0]
print('Norm equation error:',norm_error)
# Overwrite into code above and re-run all.
#reg_lambda = 0.01 #tf.Variable(0.1, name="reg_lambda")
#error = tf.reduce_mean(tf.square(y_pred - y)) + reg_lambda / 2 * tf.reduce_sum(tf.square(theta))