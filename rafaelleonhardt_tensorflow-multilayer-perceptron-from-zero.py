import numpy as np
import tensorflow as tf
# creating the input data to our neural network
# Are four elemtents of x1 and x2 columns
data_input_x = np.array([[0.0, 0.0], 
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.0]])
data_input_x
# creating the classification that we know to out input data ('classe' column)
# XOR
data_y = np.array([[0], [1], [1], [0]])
data_y
# Define the variables used during de processing
# Two weights to only one neuron
# Weights are initialized with zero
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

weights = { 
            # 3x2 matrix for weights between input layer and hidden layer
            # initialize weights using randomic 
            'hidden_layer': tf.Variable(tf.random_normal([input_neurons,hidden_neurons]), name='weights_hidden_layer'), # 2x3 matrix
    
            # 2x3 matrix for weights betwwen hidden layer and ouput layer
            'output_layer': tf.Variable(tf.random_normal([hidden_neurons, output_neurons]), name='weights_output_layer') # 3x1 matrix
          }

# define the bias
bias = {
        'hidden_layer': tf.Variable(tf.random_normal([hidden_neurons]), name='bias_hidden_layer'),
        'output_layer': tf.Variable(tf.random_normal([output_neurons]), name='bias_output_layer')
       }

#
records = len(data_input_x)
x_placeholder = tf.placeholder(tf.float32, [records, input_neurons], name='xph')
y_placeholder = tf.placeholder(tf.float32, [records, output_neurons], name='yph')

# define our hidden layer calculation adding a bias
hidden_layer = tf.add(tf.matmul(x_placeholder, weights['hidden_layer']), bias['hidden_layer'])
hiddel_layer_activation = tf.sigmoid(hidden_layer)

# define our outputlayer calculation
output_layer = tf.add(tf.matmul(hiddel_layer_activation, weights['output_layer']), bias['output_layer'])

# define our activation function to transform the output layer values into knowed classes (0 or 1)
predictions = tf.sigmoid(output_layer)

# define score function to evaluate the accuracy
error = tf.losses.mean_squared_error(y_placeholder, predictions)

# define function used to adjust the weights during the training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(error)

# Create the initializer function TensorFlow Variables used during the processing
init = tf.global_variables_initializer()
with tf.Session() as s:
    s.run(init)
    for epoch in range(10000):
        mean_error = 0
        _, cost = s.run([optimizer, error], feed_dict={ x_placeholder: data_input_x, y_placeholder: data_y})
        if epoch % 200 == 0:
            mean_error += cost / records
            print('Epoch: ', epoch+1, ' - Mean Error: ', mean_error)
            
    best_weights, best_bias = s.run([weights, bias])
print('\n\nWeights to the best accuracy: \n', best_weights)
print('\n\nBias to the best accuracy: \n', best_bias)
hidden_layer_test = tf.add(tf.matmul(x_placeholder, best_weights['hidden_layer']), best_bias['hidden_layer'])
hiddel_layer_activation_test = tf.sigmoid(hidden_layer_test)
output_layer_test = tf.add(tf.matmul(hiddel_layer_activation_test, best_weights['output_layer']), best_bias['output_layer'])
predictions_test = tf.sigmoid(output_layer_test)
with tf.Session() as s:
    s.run(init)
    print('Classes: \n', s.run(predictions_test, feed_dict = { x_placeholder: data_input_x }))
    # values very close to 0, 1, 1 and 0 as our XOR knowed classes value.