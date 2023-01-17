from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense



# Define the number of inputs and outputs

out_nodes = 2

in_nodes = 2



model = Sequential()    # This function allows you to create a sequantial model (a stack) to which you can add as many dense layers as you wish.



# Define a hidden layer with single perceptron.

dense_layer = Dense(out_nodes, activation='sigmoid', kernel_initializer="Ones", bias_initializer="Ones")  # An activation function in a neural network provides non-linearity to the data which is important for learning features from the input data, else the learning will stop at a particular stage and leads to a dying neuron problem.



model.add(dense_layer)
import numpy as np



# Generate some random data

train_data = np.random.random((1000, 100))

train_labels = np.random.randint(2, size=(1000, 1))

test_data = np.random.random((100, 100))

test_labels = np.random.randint(2, size=(100, 1))



units = 32



model = Sequential()



model.add(Dense(units, activation='relu', input_dim=100))       # Input dimension should be equal to the number of features

model.add(Dense(units, activation='relu'))



# The output should be a single outcome so one Dense layer is defined with a single unit.

model.add(Dense(1, activation='sigmoid'))
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad, Adamax, Nadam, RMSprop



adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)

adad = Adadelta(lr=1.0,rho=0.95,epsilon=None,decay=0.0)

adag = Adagrad(lr=0.01,epsilon=None,decay=0.0)

adamax = Adamax(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)

nadam = Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=None,schedule_decay=0.004)

rms = RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0.0)



loss = ['sparse_categorical_crossentropy','mean_squared_error','mean_absolute_error',

        'categorical_crossentropy','categorical_hinge']



metrics = ['accuracy','precision','recall']
# Compile the above created model

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])  # Optimises the learning by updating the weights with Stochastic Gradient Descent method.



# Train the model by fitting the train data to the model we compiled in the above line. This is stored in a variable because the output of 'fit' function is a history class which consists of 4 key, value pairs for accuracy, val_accuracy, loss, val_loss

history = model.fit(train_data, train_labels, epochs=30, batch_size=128)

_, train_acc = model.evaluate(train_data, train_labels, verbose=1)

_, test_acc = model.evaluate(test_data, test_labels, verbose=1)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
import tensorflow as tf

from sklearn.model_selection import train_test_split

mnist_fashion = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist_fashion.load_data()



# split training set into training set and validation set using train_test_split provided by scikit-learn 

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=77)



num_classes = 10   # The items in the dataset are to be classified into 1 of the 10 classes.



print(x_train.shape, x_val.shape, x_test.shape)
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



plt.figure(figsize=(6,6))

for i in range(16):

    plt.subplot(4,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[y_train[i]])

plt.show()
from tensorflow.keras.layers import Flatten



model = Sequential()

model.add(Flatten())  # This function flattens the input data



# Feel free to play around with different parameters here like number of units in each layer or switching the activation function or increasing/decreasing the number of layers.

model.add(Dense(512, activation='relu'))    



model.add(Dense(256, activation='relu')) 



model.add(Dense(128, activation='relu'))



model.add(Dense(64, activation='relu'))



model.add(Dense(10, activation='softmax'))   # The number of units in the last layer should always be the number of classes in which we have to classify our input data.
model.compile(optimizer=adam, loss=loss[0], metrics=metrics[0])
batch_size = 128

epochs = 50



history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val))  # You can set verbose to 1 to get the status of your model training, 2 to get one line per epoch, here I kept it 0 to keep the notebook precise. 



_, train_acc = model.evaluate(x_train, y_train, verbose=1)

_, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
def plot_history(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper left')

    plt.show()
plot_history(history)
from  tensorflow.keras import regularizers



# Build the model

model = Sequential()

model.add(Flatten())



# Add l2 regularizer (with 0.001 as regularization value) as kernel regularizer 

model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))



# Add l2 regularizer (with 0.001 as regularization value) as kernel regularizer 

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))



# Add l2 regularizer (with 0.001 as regularization value) as kernel regularizer 

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))



# Add l2 regularizer (with 0.001 as regularization value) as kernel regularizer 

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))



# Add a Dense layer with number of neurons equal to the number of classes, with softmax as activation function

model.add(Dense(10, activation='softmax'))



# Compile the model created above.

model.compile(optimizer=adam, loss=loss[0], metrics=metrics[0])





# Fit the model created above to training and validation sets.

history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val))



# Call the plot_history function to plot the obtained results

plot_history(history)



# Evaluate the results

_, train_acc = model.evaluate(x_train, y_train, verbose=1)

_, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
from tensorflow.keras.layers import Dropout



# Build the model

model = Sequential()

model.add(Flatten())



model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.3))   # Add a dropout layer with 0.3 probability

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.3))   # Add a dropout layer with 0.3 probability

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.3))   # Add a dropout layer with 0.3 probability

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(10, activation='softmax'))

 

model.compile(optimizer=adam, loss=loss[0], metrics=metrics[0])



history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val))



plot_history(history)



_, train_acc = model.evaluate(x_train, y_train, verbose=1)

_, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import os



os.mkdir('my_checkpoint_dir')



# Early stopping, for more refer documentation here: https://keras.io/callbacks/

es_callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model = Sequential()

model.add(Flatten())



model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.3))

model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dropout(0.3))

model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=adam, loss=loss[0], metrics=metrics[0])



history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_val, y_val), callbacks=[es_callback])  # The callbacks parameter of the fit() function is responsible to handle the Early Stopping.



plot_history(history)



_, train_acc = model.evaluate(x_train, y_train, verbose=1)

_, test_acc = model.evaluate(x_test, y_test, verbose=1)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))