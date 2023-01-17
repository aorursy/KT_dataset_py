import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf



n_inputs = 784 

n_hidden1 = 500

n_hidden2 = 400

n_hidden3 = 300

n_hidden4 = 200

n_hidden5 = 100

n_outputs = 10



X=tf.placeholder(tf.float32,shape=(None,n_inputs),name="X")

Y=tf.placeholder(tf.int64,shape=(None),name="Y")

def neural_network(X,n_neurons,name,activation=None):

    with tf.name_scope("name"):

        n_inputs=int(X.get_shape()[1])

        stddev=2/np.sqrt(n_inputs)

        init=tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)

        W=tf.Variable(init,name="kernel")

        b=tf.Variable(tf.zeros([n_neurons]),name="bias")

        Z=tf.matmul(X,W)+b

        if activation is not None:

            return activation(Z)

        else:

            return Z

print("Initialized...")
training=tf.placeholder_with_default(False,shape=(),name="training")

dropout_rate=0.8

X_drop=tf.layers.dropout(X,dropout_rate,training=training)

he_init=tf.contrib.layers.variance_scaling_initializer()

with tf.name_scope("dnn"):

    hidden1=tf.layers.dense(X,n_hidden1,name="hidden1",activation=tf.nn.relu,kernel_initializer=he_init)

    hidden2=tf.layers.dense(hidden1,n_hidden2,name="hidden2",activation=tf.nn.relu,kernel_initializer=he_init)

    hidden3=tf.layers.dense(hidden2,n_hidden3,name="hidden3",activation=tf.nn.relu,kernel_initializer=he_init)

    hidden4=tf.layers.dense(hidden3,n_hidden4,name="hidden4",activation=tf.nn.relu,kernel_initializer=he_init)

    hidden5=tf.layers.dense(hidden4,n_hidden5,name="hidden5",activation=tf.nn.relu,kernel_initializer=he_init)

    

    logits=tf.layers.dense(hidden5,n_outputs,name="Outputs",kernel_initializer=he_init)

    

print("dnn")
with tf.name_scope("loss"):

    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits)

    loss=tf.reduce_mean(xentropy,name="loss")

print("loss")
learning_rate=0.0001

with tf.name_scope("train"):

    optimizer=tf.train.AdamOptimizer(learning_rate)

    training_op=optimizer.minimize(loss)

print("train")
with tf.name_scope("eval"):

    correct=tf.nn.in_top_k(logits,Y,1)

    accuracy=tf.reduce_mean((tf.cast(correct,tf.float32)))

print("eval")
init=tf.global_variables_initializer()

saver=tf.train.Saver()



print("init")
dataset=pd.read_csv("../input/train.csv")

X_data=dataset.iloc[:,1:785].values

Y_data=dataset.iloc[:,0:1].values

from sklearn.cross_validation import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X_data,Y_data,test_size=0.2)

print("input")


y_train=y_train.reshape(33600,)

y_valid=y_valid.reshape(8400,)

n_epochs=300

batch_size=20

lengthofdata=len(X_train)

print("Initialized")
list1=[]

with tf.Session() as sess:

    init.run()

    for epochs in range(n_epochs):

        i=0

        while i<lengthofdata:

            start=i

            end=i+batch_size

            X_batch=np.array(X_train[start:end])

            Y_batch=np.array(y_train[start:end])

            i=end

            sess.run(training_op,feed_dict={X:X_batch,Y:Y_batch})

        acc_train=accuracy.eval(feed_dict={X:X_batch,Y:Y_batch})

        X_valid=np.array(X_valid)

        y_valid=np.array(y_valid)

        acc_test=accuracy.eval(feed_dict={X:X_valid,Y:y_valid})

        

        print(epochs, "Train Accuracy: ",acc_train,"  test Accuracy: ",acc_test)

        list1.append(acc_test)

    save_path=saver.save(sess,"./my_model_final.ckpt")

    
%matplotlib inline

plt.plot(list1)

plt.xlabel("No Of Epochs")

plt.ylabel("Validation Accuracy")
#Source https://github.com/ageron/handson-ml/blob/master/14_recurrent_neural_networks.ipynb

from IPython.display import clear_output, Image, display, HTML



def strip_consts(graph_def, max_const_size=32):

    """Strip large constant values from graph_def."""

    strip_def = tf.GraphDef()

    for n0 in graph_def.node:

        n = strip_def.node.add() 

        n.MergeFrom(n0)

        if n.op == 'Const':

            tensor = n.attr['value'].tensor

            size = len(tensor.tensor_content)

            if size > max_const_size:

                tensor.tensor_content = "b<stripped %d bytes>"%size

    return strip_def



def show_graph(graph_def, max_const_size=32):

    """Visualize TensorFlow graph."""

    if hasattr(graph_def, 'as_graph_def'):

        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)

    code = """

        <script>

          function load() {{

            document.getElementById("{id}").pbtxt = {data};

          }}

        </script>

        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>

        <div style="height:600px">

          <tf-graph-basic id="{id}"></tf-graph-basic>

        </div>

    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))



    iframe = """

        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>

    """.format(code.replace('"', '&quot;'))

    display(HTML(iframe))
show_graph(tf.get_default_graph())