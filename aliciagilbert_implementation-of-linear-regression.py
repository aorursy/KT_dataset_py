import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
obs = 1000

x1 = np.random.uniform(-10, 10, (obs,1))
x2 = np.random.uniform(-10, 10, (obs,1))

inputs = np.column_stack((x1, x2))
noise = np.random.uniform(-1, 1, (obs,1))

y = 2 * x1 - 3 * x2 + 4 + noise
#to use the 3D plot, you need to reshape the generated targets, y
targets = y.reshape(obs,)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x1, x2, targets)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Targets')

ax.view_init(azim=100)

plt.show()
initial = 0.1

w = np.random.uniform(-initial, initial, (2,1))
b = np.random.uniform(-initial, initial, 1)

#print the initial w and b to the console
print(w)
print(b)
learning_rate = 0.01
iterations = 100
for i in range(iterations):
    y_predicted = np.dot(inputs, w) + b
    deltas = y_predicted - y
    loss = (1/obs) * np.sum(deltas ** 2) / 2
    print(loss)
    
    #need to scale the deltas the same way as the loss
    deltas_scaled = deltas / obs
    
    #compute gradients
    dw = np.dot(inputs.T, deltas_scaled)
    db = np.sum(deltas_scaled)
    
    #update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
#our final weights and biases
print("Weights: {} \n Biases: {}".format(w,b))
print("The model derived using our optimization algorithm is: \ny = {} * x1 + {} * x2 + {}".format(w[0][0],w[1][0],b[0]))
plt.plot(y_predicted, y)
plt.xlabel('predicted')
plt.ylabel('targets')

plt.show()
import tensorflow as tf
np.savez('data', inputs=inputs, targets=y)
train_data = np.load('data.npz')

#declare variables to hold our input and output variable size
#recall we have 2 inputs (x1,x2) and 1 output, y
num_inputs = 2
num_outputs = 1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_outputs,
    kernel_initializer = tf.random_uniform_initializer(minval = -initial, maxval = initial),
    bias_initializer = tf.random_uniform_initializer(minval = -initial, maxval = initial))    
])
model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate), loss = 'mean_squared_error')
model.fit(train_data['inputs'], train_data['targets'], epochs = iterations, verbose = 2)
#get our weights and biases
w = model.layers[0].get_weights()[0]
b = model.layers[0].get_weights()[1]

print("Weights: {} \n Biases: {}".format(w,b))
print("The model derived using TensorFlow is: \ny = {} * x1 + {} * x2 + {}".format(w[0][0],w[1][0],b[0]))
#get prediction values at each epoch
predictions = model.predict(train_data['inputs'])
plt.plot(predictions, train_data['targets'])
plt.xlabel('predicted')
plt.ylabel('targets')

plt.show()