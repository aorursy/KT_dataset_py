import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns

%matplotlib inline 



import theano

import theano.tensor as T

import lasagne
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 128

COLS = 128

CHANNELS = 3



train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

train_images = train_dogs[:1000] + train_cats[:1000]

#random.shuffle(train_images)

test_images =  test_images[:25]



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



X_train = prep_data(train_images)

X_test = prep_data(test_images)



print("Train shape: {}".format(X_train.shape))

print("Test shape: {}".format(X_test.shape))

n_train = len(X_train)
labels = []

for i in train_images:

    if 'dog' in i:

        labels.append(1)

    else:

        labels.append(0)

        

y_train = np.zeros((n_train,2))



for i,l in enumerate(labels):

    y_train[i,0] = l

    y_train[i,1] = 1-l
X_train = (X_train)/255

X_test = (X_test)/255



N = 1600



X_val = X_train[N:]

y_val = y_train[N:]

X_train = X_train[0:N]

y_train = y_train[0:N]



n_train = len(X_train)

n_test = len(X_test)

n_val = len(X_val)

print(n_train)

print(n_val)

print(n_test)
i0 = 22

plt.imshow(X_test[i0,0])

plt.show()
def build_model(input_var=None):

    network = lasagne.layers.InputLayer((None,CHANNELS, ROWS, COLS),input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network,32,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(network,32,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network,(2,2))

    network = lasagne.layers.Conv2DLayer(network,64,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(network,64,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network,(2,2))

    network = lasagne.layers.Conv2DLayer(network,128,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(network,128,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network,(2,2))

    network = lasagne.layers.Conv2DLayer(network,256,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.Conv2DLayer(network,256,(3,3),nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network,(2,2))

    network = lasagne.layers.FlattenLayer(network)

    network = lasagne.layers.DenseLayer(network,256,nonlinearity=None)

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(network,128,nonlinearity=None)

    network = lasagne.layers.DropoutLayer(network, p=0.5)

    network = lasagne.layers.DenseLayer(network,2,nonlinearity=lasagne.nonlinearities.softmax)

    

    return network
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)

    if shuffle:

        indices = np.arange(len(inputs))

        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

        if shuffle:

            excerpt = indices[start_idx:start_idx + batchsize]

        else:

            excerpt = slice(start_idx, start_idx + batchsize)

    yield inputs[excerpt], targets[excerpt]







def main(num_epochs=500):

   

    # Prepare Theano variables for inputs and targets

    input_var = T.tensor4('inputs')

    target_var = T.matrix('targets')



    # Create neural network model (depending on first command line parameter)

    print("Building model and compiling functions...")

    network = build_model(input_var)  



    # Create a loss expression for training, i.e., a scalar objective we want

    # to minimize (for our multi-class problem, it is the cross-entropy loss):

    prediction = lasagne.layers.get_output(network)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    loss = loss.mean()

    # We could add some weight decay as well here, see lasagne.regularization.



    # Create update expressions for training, i.e., how to modify the

    # parameters at each training step. Here, we'll use Stochastic Gradient

    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.

    params = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.nesterov_momentum(

            loss, params, learning_rate=0.01, momentum=0.9)



    # Create a loss expression for validation/testing. The crucial difference

    # here is that we do a deterministic forward pass through the network,

    # disabling dropout layers.

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)

    test_loss = test_loss.mean()

    # As a bonus, also create an expression for the classification accuracy:

#     test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),

#                       dtype=theano.config.floatX)

    



    # Compile a function performing a training step on a mini-batch (by giving

    # the updates dictionary) and returning the corresponding training loss:

    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

    

    val_fn = theano.function([input_var, target_var], test_loss,allow_input_downcast=True)

    

    pred = lasagne.layers.get_output(network, deterministic=True)

    get_pred = theano.function([input_var], pred[0], allow_input_downcast=True)



    # Finally, launch the training loop.

    print("Starting training...")

    # We iterate over epochs:

    

    loss_train = np.zeros(num_epochs)

    loss_val = np.zeros(num_epochs)

    

    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:

        train_err = 0

        train_batches = 0

        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):

            inputs, targets = batch

            train_err += train_fn(inputs, targets)

            train_batches += 1

            

        # And a full pass over the validation data:

        val_err = 0

        val_batches = 0

        for batch in iterate_minibatches(X_val, y_val, 100, shuffle=False):

            inputs, targets = batch

            err = val_fn(inputs, targets)

            val_err += err

            val_batches += 1

            

        loss_train[epoch] = train_err/train_batches

        loss_val[epoch] = val_err/val_batches

            

        if epoch%1==0:

            print("Epoch {} of {}".format(epoch + 1, num_epochs))

            print("training loss:\t\t{:.6f}".format(train_err / train_batches))

            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            

    N_test = len(X_test)

    yPred = np.zeros(N_test)

    for i in range(N_test):

        t = np.reshape(X_test[i],(1,3,ROWS,COLS))

        yPred[i] = get_pred(t)[0]

    

    plt.plot(loss_train,'b',label='train loss')

    plt.plot(loss_val,'r',label='validation loss')

    plt.legend()

    plt.grid()

    plt.xlabel('epoch')

    plt.show()

    

    return yPred