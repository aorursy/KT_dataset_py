# Import necessary modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt

 
# Define data for the model
x = data = np.linspace(1,2,200)
y = x*4 + np.random.randn(*x.shape) * 0.3


# Create a Sequential model
model = Sequential()


# Add layers to the Sequential model
model.add(Dense(1, input_dim=1, activation='linear'))


# Compile the model
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])


# Declare initial weights and bias
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init)) 


# Train the model
model.fit(x,y, batch_size=1, epochs=30, shuffle=False)


# Set final weights and bias
weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))


# Predict the results
predict = model.predict(data)


# Visualize the results
plt.figure(figsize=(12,8))
plt.plot(data, predict, 'b', data , y, 'k.')
plt.show()