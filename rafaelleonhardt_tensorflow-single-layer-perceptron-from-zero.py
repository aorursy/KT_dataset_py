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
data_y = np.array([[0.0], [0.0], [0.0], [1.0]])
data_y
def step_function(sum_value):
    return tf.cast(tf.to_float(tf.math.greater_equal(sum_value, 1)), tf.float64)
# Define the variables used during de processing
# Two weights to only one neuron
# Weights are initialized with zero
weights = tf.Variable(tf.zeros([2,1], dtype = tf.float64))

# define our outputlayer calculation
output_layer = tf.matmul(data_input_x, weights)

# define our activation function to transform the output layer values into knowed classes (0 or 1)
predictions = step_function(output_layer)

# define score function to evaluate the accuracy
error = tf.subtract(data_y, predictions)

# define delta function used to adjust the weights during the training
delta = tf.matmul(data_input_x, error, transpose_a = True)
learningRate = 0.1
train = tf.assign(weights, tf.add(weights, tf.multiply(delta, learningRate)))

# Create the initializer function TensorFlow Variables used during the processing
init = tf.global_variables_initializer()
with tf.Session() as s:
    s.run(init)
    print('Output layer result: \n', s.run(output_layer))
    print('Prediction result: \n', s.run(predictions))
    print('Error result: \n', s.run(error))
    print('\n')
    for epoch in range(15):
        train_error, _ = s.run([error, train])
        train_error_sum = tf.reduce_sum(train_error)
        print('Epoch: ', epoch+1, ' - Error: ', s.run(train_error_sum))
        if train_error_sum.eval() == 0.0:
            break; # learned and got 100% accuracy
    print('\nWeights to the best accuracy: \n', s.run(weights))