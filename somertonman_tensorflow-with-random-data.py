import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import warnings  



#nothing serious, just turning off some warnings and notifications not to overload the notebook

warnings.filterwarnings('ignore')
# number of observations (i.e. records,rows) of our sample data

observations = 1000



# random vectors for Xs and Zs

xs = np.random.uniform(low=-10,high=10, size=(observations,1))

zs = np.random.uniform(-10,10,size=(observations,1))



# stacking 2 random data columns together

generated_inputs = np.column_stack((xs,zs))



# targets = f(x,z) = 2x + 3z + 5 + noise



noise = np.random.uniform(-1,1,(observations,1))



generated_targets = 2*xs + 3*zs + 5 + noise



np.savez('random_tf', inputs=generated_inputs, targets=generated_targets)



#load npZ

training_data = np.load('random_tf.npz')

training_data
# we want to measure time taken to perform model training here in between 2 points

import time

start = time.time()



input_size=2

output_size=1



model = tf.keras.Sequential([

    tf.keras.layers.Dense(output_size)

])



model.compile(optimizer='sgd',loss='mean_squared_error')

model.fit(training_data['inputs'],training_data['targets'],epochs=50,verbose=0)



#extracting weights and biases from trained model

wbs= model.layers[0].get_weights()

weights=wbs[0]

biases=wbs[1]



end = time.time()

print('Weights: ', weights)

print('Biases: ', biases)

print('Time taken: ', round(end - start,2),'s')
predicted_targets = model.predict_on_batch(training_data['inputs'])



deltas = predicted_targets - generated_targets



fig=plt.figure(figsize=(15,10))

plt.hist(deltas,bins=30)

plt.title("Histogram of errors in predicted values vs. real target values")

plt.show()
fig=plt.figure(figsize=(15,10))

ax = fig.add_subplot(1,1,1)

ax.plot(deltas)

ax.set_ylim(-10,10)

ax.set_title("Deltas of the predicted values vs. real target values on scale [-10,10]")

plt.show()
fig = plt.figure(figsize=(15,10))

plt.plot (generated_targets,predicted_targets)

plt.xlabel('Real targets')

plt.ylabel('Predicted targets')

plt.show()