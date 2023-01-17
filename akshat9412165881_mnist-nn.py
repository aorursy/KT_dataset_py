import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib.layers import fully_connected

from tensorflow.contrib.layers import batch_norm

import pandas as pd
df_train = pd.read_csv("../input/train.csv").iloc[:]

y_train = df_train.label

X_train = df_train.drop(["label"],axis=1)

df_test = pd.read_csv("../input/test.csv").iloc[:]

class fetch_mnist:

    def __init__(self,X=X_train,y=y_train,Xt=df_test):

        self.X_train = X/255

        self.y_train = y

        self.X_test = Xt/255

    def fetch_train_batch(self,n,cnn=False):

        rand_indx = np.random.permutation(len(self.X_train))

        self.X_train = self.X_train.iloc[rand_indx]

        self.y_train = self.y_train.iloc[rand_indx]

        if cnn==False:

            return(self.X_train[:n],self.y_train[:n])

        if cnn==True:

            shape = np.r_[n,28,28,1]

            data=self.X_train[:n]

            X_out = np.zeros(shape)

            X_out[:,:,:,0] = [np.array(data.iloc[i]).reshape(28,28) for i in range(data.shape[0])]

            return(X_out,self.y_train[:n])

    def fetch_test_batch(self,n,cnn=False):

        if cnn==False:

            return(self.X_test[:n])

        if cnn==True:

            shape = np.r_[n,28,28,1]

            data=self.X_train[:n]

            X_out = np.zeros(shape)

            X_out[:,:,:,0] = [np.array(data.iloc[i]).reshape(28,28) for i in range(data.shape[0])]

            return(X_out,self.y_train[:n])

        return(self.X_test[:n],self.y_test[:n])
def update_weights(n_layers,parameters):

    a=[variable.name for variable in tf.trainable_variables()]

    a=[a[i][:len(a[i])-2] for i in range(len(a))]

    for i in range(n_layers*2):

        ind = a[i].find('/')

        with tf.variable_scope(a[i][:ind],reuse=True):

            var = tf.get_variable(a[i][ind+1:])

        forge = tf.assign(var,best_weights)

    return forge.eval(feed_dict={best_weights:parameters[i]})
#Definition of the constants

n_feature = 28*28

early_stop_step=6

n_output = 20

hd_nuer = [300,100,n_output]

lr=0.01

n_epoch=10

batch_size=8000

train_batch_size=1000

mnist = fetch_mnist()

accuracy_test_1=np.empty((1,1))
#Tensorflow graph construction

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32,shape=(None,n_feature),name="X")

Y = tf.placeholder(dtype=tf.int32,shape=(None),name="Y")

best_weights = tf.placeholder(dtype=tf.float32)

is_training=tf.placeholder(tf.bool,shape=(),name="is_training")

bn_params={'is_training':is_training,'decay':0.75,'updates_collections':None}

# hid_init = tf.initializers.random_normal(stddev=0.5)

hidden = fully_connected(X,hd_nuer[0],normalizer_fn=batch_norm,normalizer_params=bn_params)

for i in range(1,len(hd_nuer)-1):

    hidden = fully_connected(hidden,hd_nuer[i],normalizer_fn=batch_norm,normalizer_params=bn_params)

logits = fully_connected(hidden,hd_nuer[len(hd_nuer)-1],activation_fn=None,normalizer_fn=batch_norm,normalizer_params=bn_params)

xentry=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)

loss = tf.reduce_sum(xentry,axis=0)

optimizer = tf.train.AdamOptimizer(learning_rate=lr)

training_op = optimizer.minimize(loss)

correct=tf.nn.in_top_k(logits,Y,1)

al_in = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fully_connected")

params_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fully_connected/weights")
#Tensorflow graph execution

with tf.Session() as sess:

    sess.run(init)

    # # logits_val = X.eval(feed_dict={X:x[1:3,:],Y:y[1:3]})

    # X_batch, y_batch = mnist.train.next_batch(batch_size)

    # print(X_batch)

    best_val=0

    step=0

    ep = 0

    for ep in range(n_epoch):

        for iteration in range(70):

            X_batch, y_batch = mnist.fetch_train_batch(batch_size)

            sess.run(training_op, feed_dict={is_training:True,X: X_batch, Y: y_batch})

            acc_train = al_in.eval(feed_dict={is_training:False,X: X_batch,Y: y_batch})

            print(ep, "Train accuracy:", acc_train)

            parama = sess.run(params_1)

            if step>early_stop_step:break

            if (acc_train >= best_val):

                best_val=acc_train

                parameters = sess.run(params)

                step=0

            else:

                step+=1

        if step>early_stop_step:break

        ep+=1

    update_weights(len(hd_nuer),parameters)

    save_path=saver.save(sess,"../working/mnist_class.ckpt")

    sess.close()

#Testing the damn! code

with tf.Session() as sess:

    saver.restore(sess,"../working/mnist_class.ckpt")

    X_test= mnist.fetch_test_batch(28000)

    acc_test = logits.eval(feed_dict={is_training:False,X:X_test})

    out_put=np.argmax(acc_test,axis=1)

    sess.close()
data = pd.DataFrame({'ImageId':range(1,len(X_test)+1),'Label':out_put})

data.to_csv("../working/output.csv",index=False)

print(data.head())