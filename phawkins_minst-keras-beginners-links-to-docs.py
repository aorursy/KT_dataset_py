import numpy as np

#from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, Reshape

from keras.optimizers import Nadam

import matplotlib.pyplot as plt

%matplotlib inline
train = np.loadtxt(fname='../input/train.csv', dtype=int, delimiter=',', skiprows=1)

test = np.loadtxt(fname='../input/test.csv', dtype=int, delimiter=',', skiprows=1)



print('Training set shape: ', str(train.shape))

print('Test set shape:     ', str(test.shape))
np.random.shuffle(train)

validation, train = np.split(ary=train, indices_or_sections=[5000])



print('Training set shape:     ', str(train.shape))

print('Validation set shape:   ', str(validation.shape))
trainLabels, trainImages = np.split(train, [1], axis=1)

validationLabels, validationImages = np.split(validation, [1], axis=1)



print('Training labels shape: ', str(trainLabels.shape))

print('Training images shape: ', str(trainImages.shape))

print('Validation labels shape: ', str(validationLabels.shape))

print('Validation images shape: ', str(validationImages.shape))
# Step 1

trainLabels = np.reshape(trainLabels, -1)

validationLabels = np.reshape(validationLabels, -1)



print('Training labels shape: ', str(trainLabels.shape))

print('Validation labels shape: ', str(validationLabels.shape))
# Step 2

trainLabels = np.eye(10, dtype=int)[trainLabels]

validationLabels = np.eye(10, dtype=int)[validationLabels]



print('Training labels shape: ', str(trainLabels.shape))

print('Validation labels shape: ', str(validationLabels.shape))
print('Some training labels: ', str(trainLabels[0:5]))
# hyperparameters

epochs = 10

batch_size = 256

num_classes = 10

alpha = 0.001  # AKA - learning rate



# model construction

model = Sequential()

model.add(Dense(64, 

                input_shape=(784,), 

                activation='relu', 

                kernel_initializer='RandomUniform',

                bias_initializer='RandomNormal'))

model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))



model.compile(optimizer=Nadam(alpha),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
training = model.fit(trainImages, trainLabels,

                     validation_data=(validationImages, validationLabels),

                     epochs=epochs,

                     batch_size=batch_size, 

                     verbose=1)
# show the loss and accuracy

loss = training.history['loss']

val_loss = training.history['val_loss']

acc = training.history['acc']

val_acc = training.history['val_acc']



# loss plot

tra = plt.plot(loss)

val = plt.plot(val_loss, 'r')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend(["Training", "Validation"])



plt.show()



# accuracy plot

plt.plot(acc)

plt.plot(val_acc, 'r')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Accuracy')

plt.legend(['Training', 'Validation'], loc=4)

plt.show()
import pandas as pd

from IPython.display import FileLink



predictions = model.predict_classes(test, verbose=1)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("sub.csv", index=False, header=True)



FileLink('sub.csv')