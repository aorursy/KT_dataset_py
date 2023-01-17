# Objectives:

# *. Test the pickleability of keras, versus tf.keras models
# *. Relevant thread : https://github.com/tensorflow/tensorflow/issues/34697

# Conclusions:

# *. Tensorflow 2.2.0 based Keras models are not pickleable
# *. Keras 2.4.3 based Keras models are also not pickleable
# *. We can however, for each process that we wish to execute, load the model separately and then parallelize the predictions easily.
    # *. Note also, that in this case, we did not import tensorflow, and / or keras inside the individual scoring function. 
        # *. In other words, we load tensorflow / keras once globally, and were able to safely re-use that without any issues. 

# Usual imports

import pickle
import keras
import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np

import time

# Check if we have more than 1 cores

print(mp.cpu_count())


# Print relevant versions

print(keras.__version__)


print(tf.__version__)

print(tf.keras.__version__)
# Now let's fit a model, using fake data

# Fake data with 4 observations, and 5 input features
X = np.arange(20).reshape(4,5)
print(X)

y = np.arange(4).reshape(4,1)
print(y)


# Define a simple model and fit it
# Note that here we can replace keras with tf.keras, and it still fails

model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=5, activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, y, epochs=1, verbose=1, workers=1, use_multiprocessing=True)


# Save the model to Disk

model.save('my_regression_model.h5', include_optimizer=True)

# Restore back from Disk

loaded_model = keras.models.load_model('my_regression_model.h5')

# Confirm that models are the same

print('Perform Layer-wise comparison to ensure they are the same.')

for l1, l2 in zip(model.layers, loaded_model.layers):
    print(l1.get_config() == l2.get_config())


# Create a series of other inputs which we would like to score in parallel

datasets_to_score = list()

for i in range(0,10):
    datasets_to_score.append(np.arange(20).reshape(4,5))
    

# Define the scoring function

def test_tf_scoring(dataset_to_score):
    # The model below can also be replaced with loaded_model, but with the same results
    prediction = model.predict(dataset_to_score)
    return prediction



Parallel(n_jobs=8)(delayed(test_tf_scoring)(dataset_to_score) for dataset_to_score in datasets_to_score) #this will spit out the error above


# The above fails for tf.keras, and keras ( at least that is consistent ! )


# Next, let's define a new method which will load the model, per function invocation

def test_tf_scoring_loadmodel(dataset_to_score):
    local_model = keras.models.load_model('my_regression_model.h5')
    prediction = local_model.predict(dataset_to_score)
    return prediction


Parallel(n_jobs=8)(delayed(test_tf_scoring_loadmodel)(dataset_to_score) for dataset_to_score in datasets_to_score) #this will spit out the error above


# Clone Keras Model

copy_keras_model = keras.models.clone_model(loaded_model)
copy_keras_model.set_weights(loaded_model.get_weights())

print('Perform Layer-wise comparison')

for l1, l2 in zip(copy_keras_model.layers, loaded_model.layers):
    print(l1.get_config() == l2.get_config())
    
print('Perform Overall model comparison')

print(copy_keras_model.get_config() == loaded_model.get_config())


# Compare output of models

print(loaded_model.predict(datasets_to_score[0]))

print(copy_keras_model.predict(datasets_to_score[0]))


# We can therefore see that the 'new' model is effectively the same as the old model from a prediction standpoint
# This cloned model is however, not compiled, and therefore cannot be used for further fitting or optimization, but we don't want to do that anyways


# Next, let's define a new method which will clone an existing ( in-memory ) model, and try to use it

def test_tf_scoring_clonemodel(dataset_to_score):
    cloned_local_model = keras.models.clone_model(loaded_model)
    cloned_local_model.set_weights(loaded_model.get_weights())
    prediction = cloned_local_model.predict(dataset_to_score)
    return prediction

Parallel(n_jobs=8)(delayed(test_tf_scoring_clonemodel)(dataset_to_score) for dataset_to_score in datasets_to_score) #this will spit out the error above


