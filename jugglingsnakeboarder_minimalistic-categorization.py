# Minimalistic categorization sample 

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data createtd with make_blobs from the sklearn.datasets 



from sklearn.datasets import make_blobs

from keras import models

from keras import layers

import matplotlib.pyplot as plt

# creating 1000 samples of 2-dimensional input data and target-vector with 0/1 categorical outputs

features, targetvector = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 0.5, shuffle = True, random_state = 1)

print("The first three rows of Matrix of features: \n",features[:3])

print("The targetvector: ",targetvector[:10])
# visualisation of the data, created with sklearn "make_blobs"

plt.scatter(features[:,0],features[:,1],c=targetvector)
# split up the features into training-data and test-data 

features_training_data = features[0:750] #the first 750 features

features_test_data = features[750:1000]#the last 250 features

print(features_training_data.shape)

print(features_test_data.shape)
# split up the target-vector into training- an test-data

targetvector_training_data = targetvector[0:750]

targetvector_test_data = targetvector[750:1000]

print(targetvector_training_data.shape)

print(targetvector_test_data.shape)
# Creating the model 16'relu' x 16'relu' x 1'sigmoid' units

NN = models.Sequential()

NN.add(layers.Dense(16, activation='relu', input_shape=(2,)))

NN.add(layers.Dense(16, activation='relu'))

NN.add(layers.Dense(1,activation='sigmoid'))

NN.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# split up validation data from training-data

features_val_data = features_training_data[:250]# the first 250 training-data

features_rest_data = features_training_data[250:]# the last 500 training-data



targetvector_val_data = targetvector_training_data[:250]# the first 250 target-data

targetvector_rest_data = targetvector_training_data[250:]# the last 500 target-data

protocol = NN.fit(features_rest_data,targetvector_rest_data, epochs=8,batch_size=100, validation_data=(features_val_data,targetvector_val_data))
result = NN.evaluate(features_test_data,targetvector_test_data)

print(result)
the_protocol = protocol.history

#the_protocol.keys()

error_values = the_protocol['loss']

validation_error_values = the_protocol['val_loss']

epochs = range(1,len(error_values)+1)

accuracy = the_protocol['accuracy']

validation_accuracy = the_protocol['val_accuracy']

plt.subplot(211)

plt.title("loss-function training/validation")

plt.xlabel("epochs")

plt.ylabel("value of loss-function")

plt.plot(epochs, error_values, 'bo', label = "loss training")

plt.plot(epochs, validation_error_values, 'b',label='loss validation')

plt.legend()



plt.subplot(212)

plt.title("korrect classification rate training/validation")

plt.plot(epochs, validation_accuracy, 'g--', label='validation accuracy')

plt.plot(epochs, accuracy, 'g', label='korrect classification rate')

plt.xlabel("epochs")

plt.ylabel("korrect classification rate")

plt.legend()

plt.show()


