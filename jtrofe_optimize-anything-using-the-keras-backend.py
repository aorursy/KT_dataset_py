import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
x_vals = np.linspace(0, 10, 50)                       # Get 50 points in between 0 and 10
y_vals = x_vals * 5 + 7                               # Model the equation 5x + 7
y_vals += np.random.rand(y_vals.shape[0]) * 2-1       # Add a little variation
plt.scatter(x_vals, y_vals,color='black')             # Scatter plot of our random points
plt.show()                                            #
X = K.placeholder(shape=(None, 1))                    # Inputs
weight_array = np.array([[0.],[0.]])                  # Initial value for weights
W = K.variable(value=weight_array)                    # Weights
Y = X * W[0] + W[1]                                   # Output
y_true = K.constant(y_vals.reshape(len(y_vals),1))    # Actual value of the equation for each x_val
loss = K.mean(K.square(y_true - Y))                   # Means squared error
grads = K.gradients(loss, W)[0]                       # Gradients are loss with respect to weights
iterate = K.function([X], [grads, loss])              # Given [X], calculate gradients and loss
x_input = [np.array(x_vals).reshape(len(x_vals),1)]   # So it has the right shape
loss_history = []                                     # Initialize loss history
for _ in range(500):                                  # Do gradient descent for 500 iterations
    grad_vals, loss_val = iterate(x_input)            # Run "iterate" to get gradients and loss
    weight_array = weight_array - grad_vals * 0.01    # Update weight array with learning rate 0.01
    K.set_value(W, weight_array)                      # Update the Keras variable for weights
    loss_history.append(loss_val)                     # Add loss to our history for graphing
predict = K.function([X],[Y])                         # Create a function to calculate Y given X
y_pred = np.squeeze(predict(x_input))                 # Use our weights run "predict"
fig = plt.figure(figsize=(10, 5))                     # Create a space 10 x 5
ax = fig.add_subplot(121, title='Predictions')        # Add our predictions to the 1st of 2 subplots
ax.scatter(x_vals, y_vals, color='black')             # Scatter our training example points
ax.plot(x_vals, y_pred, color='y', linewidth=3)       # Plot the predictions we made using our weights
ax = fig.add_subplot(122, title='Cost History')       # Add a second subplot for the cost history
ax.plot(loss_history)                                 # Plot cost history
plt.show()                                            #
print(np.squeeze(weight_array))                       # Could also use K.eval(W) to see value of W
weight_array = np.array([[5.],[7.]])                  # Initial value for weights
K.set_value(W, weight_array)                          # Weights
x_vals = np.array([[0]])                              # Initialize input x to 0
y_true = K.constant(np.array([[22.5]]))               # Set y_true to desired output
loss = K.mean(K.square(y_true - Y))                   # Means squared error
grads = K.gradients(loss, X)[0]                       # Gradient is loss with respect to input
iterate = K.function([X], [grads, loss])              # Given [X], calculate gradients and loss
loss_history = []                                     # Initialize loss history
for _ in range(20):                                   # Do gradient descent for 20 iterations
    grad_vals, loss_val = iterate([x_vals])           # Run "iterate" to get gradients and loss
    x_vals = x_vals - grad_vals * 0.01                # Update input value
    loss_history.append(loss_val)                     # Add loss to our history for graphing
print('Best value for X: %.5f' % np.squeeze(x_vals))  # Print optimized input value
predict = K.function([X],[Y])                         # Create a function to calculate Y given X
print('Output: %.1f' % np.squeeze(predict([x_vals]))) # Get the output with our optimized input
plt.plot(loss_history)                                # Plot loss
plt.title('Cost History')                             #
plt.show()                                            #