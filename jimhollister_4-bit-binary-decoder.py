import time

import numpy as np     # linear algebra support

import keras as ks     # high-level machine learning API
# Load the training data

training_data = np.loadtxt('../input/binary_decoder_training_integers.csv', delimiter=',')



# Extract the inputs and outputs into separate matricies

training_input = training_data[:,0:4]

print("training_input:\n{}".format(training_input))

training_output = training_data[:,4:5]



# Convert the base-10 values to one-hot format to turn this into a categorization problem

training_output = ks.utils.to_categorical(training_output)

print("training_output:\n{}".format(training_output))
model = ks.models.Sequential()

model.add(ks.layers.Dense(20, input_dim=4, activation='relu'))

model.add(ks.layers.Dense(20, activation='relu'))

model.add(ks.layers.Dense(16, activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(training_input, training_output, epochs=200, batch_size=1, verbose=0)



score = model.evaluate(training_input, training_output, verbose=0)

print("Test loss = {}, test accuracy = {}".format(score[0], score[1]))
predicted_output = model.predict(training_input)

for i in range(len(training_input)):

    print("binary = {}, decoded = {}".format(training_input[i], np.round(predicted_output[i], 1)))