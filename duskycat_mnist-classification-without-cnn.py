import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)
batch_size = 150
learning_rate = 0.001
epochs = 26
input_neuron = 28*28 #Dimensions of the Input Data
hidden_layer_one = 768 # Number of neurons in the first Hidden layer
hidden_layer_two = 768 # Number of neurons in the Second Hidden layer
hidden_layer_three = 768 # Number of neurons in the Third Layer
output_neuron = 10 # Number of the classes we want to predict
x = tf.placeholder(tf.float32,[None,input_neuron])
y = tf.placeholder(tf.float32,[None,output_neuron])
def Neural_Network(data):
	# Hidden Layer and Output Layer Weights Initialization
	hidden_one_w = tf.Variable(tf.random_uniform([input_neuron,hidden_layer_one],seed=12))
	hidden_two_w = tf.Variable(tf.random_uniform([hidden_layer_one,hidden_layer_two],seed=12))
	hidden_three_w = tf.Variable(tf.random_uniform([hidden_layer_two,hidden_layer_three],seed=12))
	output_w = tf.Variable(tf.random_uniform([hidden_layer_three,output_neuron],seed=12))

	# Hidden Layer and Output Layer Biases Initialization
	hidden_one_bias = tf.Variable(tf.random_uniform([hidden_layer_one],seed=12))
	hidden_two_bias = tf.Variable(tf.random_uniform([hidden_layer_two],seed=12))
	hidden_three_bias = tf.Variable(tf.random_uniform([hidden_layer_three],seed=12))
	output_bias = tf.Variable(tf.random_uniform([output_neuron],seed=12))

	# Main Computation Graph for Matrix Multiplication and Addition
	h_1 = tf.add(tf.matmul(x,hidden_one_w),hidden_one_bias)
	h_2 = tf.add(tf.matmul(h_1,hidden_two_w),hidden_two_bias)
	h_3 = tf.add(tf.matmul(h_2,hidden_three_w),hidden_three_bias)
	out = tf.add(tf.matmul(h_3,output_w),output_bias)
	return out
def train_neural_network(x):
	prediction = Neural_Network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer,cost], feed_dict={x: epoch_x,y: epoch_y})
				epoch_loss += c
			print('Epoch',epoch,'completed out of',epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))
train_neural_network(x)
# This Model gained an Accuracy of ~89% Without the use of CNNs