import tensorflow as tf

import numpy as np

print("import successful!")
x = tf.constant([1,2,3,4])
x = tf.constant([1,2,3,4,5])

y = tf.constant([6,7,8,9,10])



result = tf.multiply(x,y)

result
sess = tf.Session()



print(sess.run(result))



sess.close()
# Regular Python

a = 'Hello World'

print(a)
# Tensorflow

a = tf.constant("Hello World")

print(a)
# A computation graph to calculate the area of a circle

pi = tf.constant(3.14, name="pi")

r = tf.placeholder(tf.float32, name="r")

    

a = pi * r * r
# Inspect the default graph TF has created for us

graph = tf.get_default_graph()

print(graph.get_operations())
# A function to display the graph within Jupyter notebook

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

                tensor.tensor_content = "<stripped %d bytes>"%size

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
show_graph(tf.get_default_graph().as_graph_def())
with tf.Session() as sess:

    print(a.eval(feed_dict={r: [5]}))
# y = x+z, Goal is to calculate z through TF GD optimization

x = tf.constant([[1., 2.]])

y = tf.constant([[12., 4.]])

Z = tf.Variable(tf.zeros([1, 2]))
# Define the error

yy = tf.add(x, Z)

deviation = tf.square(y - yy)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(deviation)
#Initialize the variables

init = tf.global_variables_initializer()

sess = tf.Session();

sess.run(init)



# The actual training (Finding teh value for Z in which the error is minimal)

for i in range(5000):

 sess.run(train_step)

 if i%100==0:

    print(sess.run(Z))
from sklearn.datasets import load_boston

from sklearn.preprocessing import scale

from matplotlib import pyplot as plt



# Get the data

total_features, total_prices = load_boston(True)

# Take a look at the data

total_features[:5,]



# Keep 300 samples for training

train_features = scale(total_features[:300])

train_prices = total_prices[:300]



# Keep 100 samples for validation

valid_features = scale(total_features[300:400])

valid_prices = total_prices[300:400]



# Keep remaining samples as test set

test_features = scale(total_features[400:])

test_prices = total_prices[400:]



# Init random weights

w = tf.Variable(tf.truncated_normal([13, 1], mean=0.0, stddev=1.0, dtype=tf.float64))

b = tf.Variable(tf.zeros(1, dtype = tf.float64))



def calc(x, y):

    # Returns predictions and error

    predictions = tf.add(b, tf.matmul(x, w))

    error = tf.reduce_mean(tf.square(y - predictions))

    return [ predictions, error ]



y, cost = calc(train_features, train_prices)

# Feel free to tweak these 2 values:

learning_rate = 0.025

epochs = 3000

plt.plot(points[0], points[1], 'r--')

plt.axis([0, epochs, 50, 600])

plt.show()



valid_cost = calc(valid_features, valid_prices)[1]



print('Validation error =', sess.run(valid_cost), '\n')



test_cost = calc(test_features, test_prices)[1]



print('Test error =', sess.run(test_cost), '\n')

test_features[:5,]
y = calc(test_features, test_prices)[0]

predictions = sess.run(y)

predictions
plt.plot(range(len(test_prices)), test_prices)

plt.plot(range(len(predictions.ravel())), predictions.ravel())



plt.show()



sess.close()