import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist/', one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

x_train.shape
y_test.shape
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
plt.imshow(x_train[102].reshape((28,28)), cmap='gray')
plt.title('Classe: ' + str(np.argmax(y_train[102])))
x_batch, y_batch = mnist.train.next_batch(128)
x_batch.shape
neurons_input = x_train.shape[1] # 784 pixels converted from a 28x28 image
print('Input Layer Neurons: ', neurons_input)

neurons_hidden1 = neurons_hidden2 = neurons_hidden3 = int((x_train.shape[1] + y_train.shape[1]) / 2) # (784+10)/2 = 397
print('Hidden1 Layer Neurons: ', neurons_hidden1)
print('Hidden2 Layer Neurons: ', neurons_hidden2)
print('Hidden3 Layer Neurons: ', neurons_hidden3)

neurons_output = y_train.shape[1] # 10 of target classifications
print('Output Layer Neurons: ', neurons_output)
weights = {
    'hidden1': tf.Variable(tf.random_normal([neurons_input, neurons_hidden1])),
    'hidden2': tf.Variable(tf.random_normal([neurons_hidden1, neurons_hidden2])),
    'hidden3': tf.Variable(tf.random_normal([neurons_hidden2, neurons_hidden3])),
    'output': tf.Variable(tf.random_normal([neurons_hidden3, neurons_output])),
}
bias = {
    'hidden1': tf.Variable(tf.random_normal([neurons_hidden1])),
    'hidden2': tf.Variable(tf.random_normal([neurons_hidden2])),
    'hidden3': tf.Variable(tf.random_normal([neurons_hidden3])),
    'output': tf.Variable(tf.random_normal([neurons_output]))
}
xph = tf.placeholder('float', [None, neurons_input])
yph = tf.placeholder('float', [None, neurons_output])
def run_process(x, weights, bias):
    hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']), bias['hidden1']))
    hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['hidden2']), bias['hidden2']))
    hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, weights['hidden3']), bias['hidden3']))
    output_layer = tf.add(tf.matmul(hidden_layer3, weights['output']), bias['output'])
    return output_layer
# train model functions
model = run_process(xph, weights, bias)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=yph))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(error)

# prediction result
predictions = tf.nn.softmax(model)
predictions_final = tf.equal(tf.argmax(predictions, 1), tf.argmax(yph,1))

# score function
score = tf.reduce_mean(tf.cast(predictions_final, tf.float32))
with tf.Session() as s:
    
    # require to initialize the TensorFlow Variables
    s.run(tf.global_variables_initializer())
    
    # running the trainning for 5000 epochs
    for epoch in range (5000):
        x_batch, y_batch = mnist.train.next_batch(128)
        _, cost = s.run([optimizer, error], feed_dict = { xph: x_batch, yph: y_batch })
        if epoch % 100 == 0:
            acc = s.run([score], feed_dict = {xph: x_batch, yph: y_batch})
            print('Epoch: '+ str(epoch+1) + ' - Error: ' + str(cost) + ' - Accuracy: ' + str(acc))
    
    print('Trained.')
    
    # evaluate the accuracy using our test data
    print(s.run(score, feed_dict = { xph: x_test, yph: y_test }))