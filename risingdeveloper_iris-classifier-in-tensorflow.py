import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
# Config the matlotlib backend as plotting inline in IPython
%matplotlib inline

sess = tf.Session()

#Read our data 
data = pd.read_csv('../input/Iris.csv', index_col = 0)
data.head()
data.shape
# Lets visualize with seaborn
g = sns.pairplot(data, hue="Species", size=2.5)
#extract columns to use
cols = data.columns
features = cols[0:4] #select features excluding the target label
label = cols[4]

X = data[features]
y = pd.get_dummies(data[label])

#split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)
y
batch_size = 5
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
Y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)
#Variables
W = tf.Variable(tf.zeros(shape=[4, 3])) # Since we are having four features mapping one output
b = tf.Variable(tf.constant(1.))
#Define our model
Y_pred = tf.add(tf.matmul(X_data, W), b)

#Add the loss
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= Y_pred, labels= Y_target))

# Choose our optimization algorithm
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)

#Initialize the graph
init = tf.global_variables_initializer()
sess.run(init)


loss_plot = []
for i in range(1000):
    sess.run(train_step, feed_dict={X_data: X_train, Y_target: y_train})
    if (i+1) % 200 == 0:
        print('Step #' + str(i+1) + ' W = ' + str(sess.run(W)) + 'b = ' + str(sess.run(b)))
        loss_plot.append(sess.run(tf.reduce_mean(loss),  feed_dict={X_data: X_train, Y_target: y_train}))

plt.plot(b_plt, loss_plot)
plt.show()


