import tensorflow as tf

import numpy as np

import json

import glob
data = []

for f in glob.glob("../input/pubChem_p_*.json"):

    with open(f, "rt") as f:

        data.extend(json.load(f))

        

n = len(data)
atoms = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']

na = len(atoms)
x_ = tf.placeholder(dtype=tf.float32, shape=(None, na))

W_ = tf.Variable(tf.random_normal((na,1)))

p_ = tf.reshape(tf.matmul(x_, W_), (-1,))
y_ = tf.placeholder(dtype=tf.float32, shape=(None,))

loss_ = tf.reduce_mean((p_ - y_) ** 2)

train_ = tf.train.AdamOptimizer().minimize(loss_)
x = np.array([[sum([1 for e in d['atoms'] if e['type'] == atom]) for atom in atoms] for d in data])

y = np.array([d['En'] for d in data])
x_train = x[:n//2]

y_train = y[:n//2]



x_test = x[n//2:]

y_test = y[n//2:]
sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(20000):

    ids = np.random.choice(n//2, size=2048, replace=False)

    x_batch = x_train[ids]

    y_batch = y_train[ids]

    

    _, loss = sess.run([train_, loss_], feed_dict={x_: x_batch, y_: y_batch})

    

    if i % (1000-1) == 0:

        print("RMSE", loss ** 0.5)
sess.run(loss_, feed_dict={x_: x_test, y_: y_test}) ** 0.5