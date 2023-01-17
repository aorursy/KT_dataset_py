import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
credit_card = pd.read_csv('../input/creditcard.csv')
X = credit_card.drop(columns='Class', axis=1)
y = credit_card.Class.values
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_x = tf.placeholder(tf.float32)
a = tf.constant(5., tf.float32, name='a', shape=(2,))
y = a * tf.pow(input_x,2)
x = np.array([1,2])
with tf.Session() as sess:
    result = sess.run(y, {input_x: x})
    print(result)
g = tf.gradients(y, [input_x])
grad_y_x = 0
with tf.Session() as sess:
    grad_y_x = sess.run(g,{input_x: np.array([1,2])})
    print(grad_y_x)

z = tf.log(y)
with tf.Session() as sess:
    result_z = sess.run(z,  {input_x: np.array([1,2])})
    print('z =', result_z)
    delta_z = tf.gradients(z, [y, input_x])
    grad_z_y, grad_z_x = sess.run(delta_z,  {input_x: np.array([1,2])})
    print('Gradient with respect to y', grad_z_y)
    print('Gradient with respect to x', grad_z_x)
    print('Manual chain rule', grad_z_y * grad_y_x)
#Generate data
np.random.seed(42)
eps = 0.2 * np.random.randn(1000)
x = np.random.randn(2,1000)
y = 2 * x[0,:] + 3 * x[1,:] + eps
# Setup the computational graph with loss function
input_x = tf.placeholder(tf.float32, shape=(2,None))
y_true = tf.placeholder(tf.float32, shape=(None,))
w = tf.Variable(initial_value=np.ones((1,2)), dtype=tf.float32)
y_hat = tf.matmul(w, input_x)
loss = tf.reduce_mean(tf.square(y_hat - y_true))
grad_loss_w = tf.gradients(loss, [w])
init = tf.global_variables_initializer()
losses = np.zeros(10)
with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    # Gradient descent
    for i in range(0,10):
        # Calculate gradient
        dloss_dw = sess.run(grad_loss_w, {input_x:x,
                                          y_true:y})
        # Apply gradient to weights with learning rate
        sess.run(w.assign(w - 0.1 * dloss_dw[0]))
        # Output the loss
        losses[i] =  sess.run(loss, {input_x:x,
                                     y_true:y})
        print(i+1, 'th Step, current loss: ', losses[i])
    print('Found minimum', sess.run(w))
plt.plot(range(10), losses)
plt.title('Loss')
plt.xlabel('Iteration')
_ = plt.ylabel('RMSE')
tf.train.RMSPropOptimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
losses = np.zeros(10)
with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    # Gradient descent
    for i in range(0,10):
        _, losses[i] =  sess.run([train, loss], {input_x:x,
                                     y_true:y})
    print('Found minimum', sess.run(w))
plt.plot(range(10), losses)
plt.title('Loss')
plt.xlabel('Iteration')
_ = plt.ylabel('RMSE')
np.random.seed(42)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
n_epochs = 10
batch_size = 25
losses = np.zeros(n_epochs)
with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    # Gradient descent
    indices = np.arange(x.shape[1])
    for epoch in range(0,n_epochs):
        np.random.shuffle(indices)
        for i in range(int(np.ceil(x.shape[1]/batch_size))):
            idx = indices[i*batch_size:(i+1)*batch_size]
            x_i = x[:,idx]
            x_i = x_i.reshape(2,batch_size)
            y_i = y[idx]
            sess.run(train, {input_x: x_i, 
                             y_true:y_i})
        
        if epoch%1==0: 
            loss_i = sess.run(loss, {input_x: x, 
                             y_true:y})
            print(epoch, 'th Epoch Loss: ', loss_i)
        loss_i = sess.run(loss, {input_x: x, 
                             y_true:y})
        losses[epoch]=loss_i
    print('Found minimum', sess.run(w))
plt.plot(range(n_epochs), losses)
plt.title('Loss')
plt.xlabel('Iteration')
_ = plt.ylabel('RMSE')
# Setup the computational graph with loss function
input_x = tf.placeholder(tf.float32, shape=(None, 30))
y_true = tf.placeholder(tf.float32, shape=(None,1))
w = tf.Variable(initial_value=tf.random_normal((30,1), 0, 0.1, seed=42), dtype=tf.float32)
logit = tf.matmul(input_x, w)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logit))
y_prob = tf.sigmoid(logit)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
n_epochs = 100
losses = np.zeros(n_epochs)
aucs = np.zeros(n_epochs)
with tf.Session() as sess:
    # Initialize the variables
    sess.run(init)
    # Gradient descent
    for i in range(0,n_epochs):
        _, iloss, y_hat =  sess.run([train, loss, y_prob], {input_x: X_train,
                                                           y_true: y_train.reshape(y_train.shape[0],1)})
        losses[i] = iloss
        aucs[i] = roc_auc_score(y_train, y_hat)
        if i%10==0:
            print('%i th Epoch Train AUC: %.4f Loss: %.4f' % (i, aucs[i], losses[i]))
    
    # Calculate test auc
    y_test_hat =  sess.run(y_prob, {input_x: X_test,
                                             y_true: y_test.reshape(y_test.shape[0],1)})
    weights = sess.run(w)
plt.figure(figsize=(11,6))
plt.subplot(2,1,1)
plt.plot(range(n_epochs), losses)
plt.title('Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.subplot(2,1,2)
plt.plot(range(n_epochs), aucs)
plt.title('AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC Score')
plt.tight_layout()
roc_auc_score(y_test, y_test_hat) * 100
def get_color(c):
    if -0.01 < c and c < 0.0075:
        return 'orange'
    elif c>=0.0075:
        return 'green'
    else:
        return 'red'

plt.figure(figsize=(10,8))
colors = [get_color(c) for c in np.sort(weights[:,0])]
plt.bar(np.arange(30), np.sort(weights[:,0]),  width=0.3, color=colors)
feature_names = X.columns[np.argsort(weights[:,0])]
plt.title('Feature influence')
plt.ylabel('weight')
_ = plt.xticks(np.arange(30), feature_names, rotation=60, ha='right')
