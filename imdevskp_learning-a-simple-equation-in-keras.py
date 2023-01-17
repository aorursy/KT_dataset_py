# importing nessary libaries



import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
# model learn the equation y=2x-1 from the data xs and corresponding ys



xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)
plt.figure(figsize=(5, 5))

plt.scatter(xs, ys)

plt.show()
model = keras.Sequential([

    keras.layers.Dense(units=1, input_shape=[1])

])



# Sequntial - allows you to create models layer-by-layer for most problems

# Dense - a layer of connected neurons

# units = 1, since there is only one neuron here

# input_shape = [1], since the shape of the array is a 1D array
# We use sgd optimizer which looks at mean squared error to improve the model



model.compile(optimizer='sgd', loss='mean_squared_error')
# epoch value indicates who many times the model wil go through the training 



history = model.fit(xs, ys, epochs=500, verbose=False)
model.summary()
history.history.keys()
plt.figure(figsize=(5, 5))



plt.plot(history.history['loss'], label='Training Loss')

plt.title('Loss')



plt.show()
# predict the value for 10 using the model



print(model.predict([10.0]))
# calculate the for 10 manually but putting it in the equation

print(10*2-1)