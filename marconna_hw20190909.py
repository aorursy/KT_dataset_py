# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install -U tensorflow==2.0.0-alpha0
import tensorflow as tf

#Check version of tensorflow

tf.__version__
import random

import math

def seed_everything(SEED): 

    np.random.seed(SEED) 

    tf.random.set_seed(SEED) 

    random.seed(SEED)

seed_everything(807)
dat_train = np.load('/kaggle/input/fordham-cs6000-hw/mnist_data/mnist.train.npy')

submission = pd.read_csv('/kaggle/input/fordham-cs6000-hw/mnist_data/sample_submission.csv')

train_label = np.load('/kaggle/input/fordham-cs6000-hw/mnist_data/mnist.trainlabel.npy')

dat_test = np.load('/kaggle/input/fordham-cs6000-hw/mnist_data/mnist.test.npy')
#define encoding function

def encode_labels(y, num_labels):

    onehot = np.zeros((num_labels, y.shape[0]))

    for i in range(y.shape[0]):

        onehot[y[i], i] = 1.0

    return onehot
N = dat_train.shape[0]

#get training data, validation data and test data set

from sklearn.model_selection import train_test_split

X = np.array([np.concatenate(dat_train[i,:,:]) for i in range(N)])

Y = encode_labels(train_label, 10).T

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2,random_state = 1)

X_train, X_valid, Y_train, Y_valid = X_train.T, X_valid.T, Y_train.T, Y_valid.T
#step 1. initial parameters

def initialize_parameters(layers_dims, N):

    n_1,n_2,n_3 = layers_dims

    W1 = np.array([[1/N]*n_1]*n_2)

    W2 = np.array([[1/N]*n_2]*n_3)

#    W3 = np.array([[1/N]*n_2]*n_3)

    b1 = np.array([[1/N]]*n_2)

    b2 = np.array([[1/N]]*n_3)

#    b3 = np.array([[1/N]]*n_3)

    parameters = {'W1':W1,

                  'b1':b1,

                  'W2':W2,

                  'b2':b2}

#                'W3':W3,

#                'b3':b3}

    return parameters
#define softmax function

def softmax(A):

    expA = np.exp(A)

    return expA / expA.sum(axis= 0 , keepdims=True)
#step 2.forward propagation

def forward_propagation(X,parameters):

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

#    W3 = parameters["W3"]

#    b3 = parameters["b3"]

    



    Z1 = np.dot(W1,X)+b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2,A1)+b2

    A2 = softmax(Z2)

#    Z3 = np.dot(W3,A2)+b3

#    A3 = softmax(Z3)

    

    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

  #           "Z3": Z3,

  #           "A3": A3}

    

    return A2, cache
#step 3. compute cost

def compute_cost(A2,Y):

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost

    #when caculate logprobs, we may face log(0) problem, so we add a small number to A2

    #A2[A2<=1e-100] = 1e-100

    #A2[(1-A2)<=1e-100] = 1-(1e-100)

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2), 1 - Y)

    #print(logprobs)

    cost = - np.sum(logprobs)/m 

    cost = np.squeeze(cost)     

    

    return cost
#step 4.backward propagation

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    

    # First, retrieve W1,W2 and W3 from the dictionary "parameters".

    W1 = parameters["W1"]

    W2 = parameters["W2"]

 #   W3 = parameters["W3"]



    # Retrieve also A1,A2 ans A3 from dictionary "cache".



    A1 = cache["A1"]

    A2 = cache["A2"]

   # A3 = cache["A3"]



    # Backward propagation: calculate dW1, db1, dW2, db2. 

    

  #  dZ3 = A3-Y

  #  dW3 = 1/m*np.dot(dZ3,A2.T)

  #  db3 = 1/m*np.sum(dZ3,axis = 1,keepdims = True)

  #  dZ2 = np.dot(W3.T,dZ3)*(1-np.power(A2,2))

    dZ2 = A2-Y

    dW2 = 1/m*np.dot(dZ2,A1.T)

    db2 = 1/m*np.sum(dZ2,axis = 1,keepdims = True)

    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))

    dW1 = 1/m*np.dot(dZ1,X.T)

    db1 = 1/m*np.sum(dZ1,axis = 1,keepdims = True)

    

    grads = {"dW1": dW1,

             "db1": db1,

             "dW2": dW2,

             "db2": db2}

   #          "dW3": dW3,

   #         "db3": db3}

    

    return grads
#step 5.  mini-batch gradient descent 

def random_mini_batches(X, Y, mini_batch_size = 32):

    m = X.shape[1]              

    mini_batches = []

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((10,m))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:,num_complete_minibatches*mini_batch_size:]

        mini_batch_Y = shuffled_Y[:,num_complete_minibatches*mini_batch_size:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
#step 6.update parameters

def update_parameters(parameters, grads, learning_rate):

    # Retrieve each parameter from the dictionary "parameters"

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

   # W3 = parameters["W3"]

   # b3 = parameters["b3"]

    

    # Retrieve each gradient from the dictionary "grads"

    dW1 = grads["dW1"]

    db1 = grads["db1"]

    dW2 = grads["dW2"]

    db2 = grads["db2"]

   # dW3 = grads["dW3"]

   # db3 = grads["db3"]

    

    # Update rule for each parameter

    W1 = W1-learning_rate*dW1

    b1 = b1-learning_rate*db1

    W2 = W2-learning_rate*dW2

    b2 = b2-learning_rate*db2

   # W3 = W3-learning_rate*dW3

   # b3 = b3-learning_rate*db3    



    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    #              "W3": W3,

    #              "b3": b3}

    

    return parameters
#step 7. Neural Network

def my_NN_model(X,Y,layers_dims,

                N = N,

                learning_rate = 0.01,

                mini_batch_size = 32,

               num_epochs = 50,

               print_cost = True):

    L = len(layers_dims)             # number of layers in the neural networks

    costs = []                       # to keep track of the cost

    

    # Initialize parameters

    parameters = initialize_parameters(layers_dims,N)

    minibatches = random_mini_batches(X, Y, mini_batch_size)

    for i in range(num_epochs):

        for minibatch in minibatches:

        #minibatch  = random.choice(minibatches)

        # Select a minibatch

            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation

            a2, caches = forward_propagation(minibatch_X, parameters)



            # Compute cost

            cost = compute_cost(a2, minibatch_Y)

           # print(cost)

            # Backward propagation

            grads = backward_propagation(parameters, caches,minibatch_X, minibatch_Y)



            # Update parameters

            parameters = update_parameters(parameters, grads, learning_rate)



            # Print the cost every 100 epoch

        if print_cost and i % 10 == 0:

            print ("Cost after epoch %i: %f" %(i, cost))

        if print_cost and i % 1 == 0:

            costs.append(cost)



    return costs,parameters
#get each layer unit numbers

n_1 = 784

n_2 = 256

n_3 = 10

layers_dims = [n_1,n_2,n_3]

costs,parameters = my_NN_model(X_train, Y_train,layers_dims,learning_rate = 0.01, num_epochs = 50)
#step 8.prediction

def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)

    predictions = np.argmax(A2, axis=0)

    ### END CODE HERE ###

    return predictions
#get predictions for validation dataset

pred = predict(parameters, X_valid)

org = np.argmax(Y_valid, axis=0)
##check validation accuracy for each class

def class_acc(pred, org):

    acc_dic = {}

    for c in range(10):

        class_index = [i for i in range(len(org)) if org[i]==c]

        positive_num = np.sum([1 if pred[i]==c else 0 for i in class_index])

        acc_dic[c] = positive_num/len(class_index)

    return acc_dic

print('Validation accuracy for each class:\n',class_acc(pred, org))
#get predictions for test data set

X_test = np.array([np.concatenate(dat_test[i,:,:]) for i in range(dat_test.shape[0])]).T

pred_test = predict(parameters, X_test)
#submit my predictions

submission = pd.read_csv('/kaggle/input/fordham-cs6000-hw/mnist_data/sample_submission.csv',index_col = 0)

submission['class'] = pred_test

submission.to_csv('Neural_Network_Submission.csv')
submission.head()
from tensorflow import keras

from tensorflow.keras.layers import Dense,Flatten,Activation

from tensorflow.keras import Sequential
input_dim = X_train.shape[0]



model = Sequential()

model.add(Dense(256, input_dim = input_dim , activation = 'tanh'))

model.add(Dense(10, activation = 'softmax'))

optimizer = keras.optimizers.SGD(learning_rate=0.01,momentum=0.0,name='SGD')

model.compile(loss = 'categorical_crossentropy' , optimizer = optimizer , metrics = ['accuracy'] )

model.summary()
#model.fit(X_train.T, Y_train.T, epochs = 50, batch_size = 32)

model.fit(X_train.T,

         Y_train.T,

         batch_size=32,

         epochs=50,

         validation_data=(X_valid.T, Y_valid.T))

scores = model.evaluate(X_valid.T, Y_valid.T)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#create predictions for tensorflow/keras method

pred_test_tensor = np.argmax(model.predict(X_test.T),axis =1)
submission_tensor = submission.copy()

submission_tensor['class'] = pred_test_tensor

submission_tensor.to_csv('Tensorflow_Submission.csv')
import time

def tensor_diff_optimizer_model(X_train,Y_train,X_valid,Y_valid,X_test,optimizer,optimizer_name=''):

    input_dim = X_train.shape[0]

    model = Sequential()

    model.add(Dense(256, input_dim = input_dim , activation = 'tanh'))

    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy' , optimizer = optimizer , metrics = ['accuracy'] )

    start_time = time.time()

    history =  model.fit(X_train.T,

                         Y_train.T,

                         batch_size=32,

                         epochs=50,

                         validation_data=(X_valid.T, Y_valid.T))

    training_time = time.time() - start_time

    pred_valid = np.argmax(model.predict(X_valid.T),axis =1)

    pred_test = np.argmax(model.predict(X_test.T),axis =1)

    org_valid = np.argmax(Y_valid.T,axis =1)

    index = ['validation_acc_'+str(i) for i in range(10)]+['validation_acc_overall','test_acc','Training Time','Training Loss', 'Validation Loss']

    output = pd.DataFrame(index= index)

    history.history

    info = list(class_acc(pred_valid, org_valid).values()) + [history.history['val_accuracy'][-1],np.nan, 

                                                  training_time,history.history['loss'][-1],history.history['val_loss'][-1]]

    output[optimizer_name] = info

    return output,pred_test
#SGD optimizer

optimizer1 = keras.optimizers.SGD(learning_rate=0.01,momentum=0.0,name='SGD')

optimizer1_name = 'SGD'

output1,pred_test1 = tensor_diff_optimizer_model(X_train,Y_train,X_valid,Y_valid,X_test,optimizer1,optimizer1_name)

submission_tensor['class'] = pred_test1

submission_tensor.to_csv('SGD_Submission.csv')
#Adam optimizer

optimizer2 = keras.optimizers.Adam(learning_rate=0.01,name='Adam')

optimizer2_name = 'Adam'

output2,pred_test2  = tensor_diff_optimizer_model(X_train,Y_train,X_valid,Y_valid,X_test,optimizer2,optimizer2_name)

submission_tensor['class'] = pred_test2

submission_tensor.to_csv('Adam_Submission.csv')
#Momentum optimizer

optimizer3 = keras.optimizers.SGD(learning_rate=0.01,momentum=0.8,name='Momentum')

optimizer3_name = 'Momentum'

output3,pred_test3 = tensor_diff_optimizer_model(X_train,Y_train,X_valid,Y_valid,X_test,optimizer3,optimizer3_name)

submission_tensor['class'] = pred_test3

submission_tensor.to_csv('Momentum_Submission.csv')
#RMSprop optimizer

optimizer4 = keras.optimizers.RMSprop(learning_rate=0.01,name='RMSprop')

optimizer4_name = 'RMSprop'

output4,pred_test4 = tensor_diff_optimizer_model(X_train,Y_train,X_valid,Y_valid,X_test,optimizer4,optimizer4_name)

submission_tensor['class'] = pred_test4

submission_tensor.to_csv('RMSprop_Submission.csv')
#define a function to reduce memory 

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
#define rotate image function

from scipy import ndimage

#rotate the data

def get_all_dat(dat_train,train_label):

    #flip the data

    data_train_flip_horizon = tf.image.flip_left_right(dat_train)

    #data_train_flip_horizon = reduce_mem_usage(data_train_flip_horizon)

    data_train_flip_vertical = tf.image.flip_up_down(dat_train)

   # data_train_flip_vertical = reduce_mem_usage(data_train_flip_horizon)

    #rotate data

    data_train_rotate_45 = ndimage.rotate(dat_train,45,axes=(1,2),reshape = False)

    #data_train_rotate_45 = reduce_mem_usage(data_train_rotate_45)

    data_train_rotate_90 = ndimage.rotate(dat_train,90,axes=(1,2),reshape = False)

    #data_train_rotate_90 = reduce_mem_usage(data_train_rotate_90)

    data_train_rotate_135 = ndimage.rotate(dat_train,135,axes=(1,2),reshape = False)

    #data_train_rotate_135 = reduce_mem_usage(data_train_rotate_135)

    data_train_rotate_180 = ndimage.rotate(dat_train,180,axes=(1,2),reshape = False)

    #data_train_rotate_180 = reduce_mem_usage(data_train_rotate_180)

    data_train_rotate_225 = ndimage.rotate(dat_train,225,axes=(1,2),reshape = False)

    #data_train_rotate_225 = reduce_mem_usage(data_train_rotate_225)

    data_train_rotate_270 = ndimage.rotate(dat_train,270,axes=(1,2),reshape = False)

    #data_train_rotate_270 = reduce_mem_usage(data_train_rotate_270)

    data_train_rotate_315 = ndimage.rotate(dat_train,315,axes=(1,2),reshape = False)

   # data_train_rotate_315 = reduce_mem_usage(data_train_rotate_315)

    dat_all = np.concatenate((dat_train,

                          data_train_flip_horizon,

                          data_train_flip_vertical,

                          data_train_rotate_45,

                          data_train_rotate_90,

                          data_train_rotate_135,

                          data_train_rotate_180,

                          data_train_rotate_225,

                          data_train_rotate_270,

                          data_train_rotate_315), axis=0)

    #dat_all = reduce_mem_usage(dat_all)

    label_all = np.tile(train_label,10)

    #label_all = reduce_mem_usage(label_all)

    X_all = dat_all.reshape(dat_all.shape[0],dat_all.shape[1]*dat_all.shape[2])

    #X_all = np.array([np.concatenate(dat_all[i,:,:]) for i in range(dat_all.shape[0])])

    Y_all = encode_labels(label_all, 10).T

    X_train_all, X_valid_all, Y_train_all, Y_valid_all = train_test_split(X_all, Y_all, test_size=0.2,random_state = 1)

    X_train_all, X_valid_all, Y_train_all, Y_valid_all = X_train_all.T, X_valid_all.T, Y_train_all.T, Y_valid_all.T

    return X_train_all, X_valid_all, Y_train_all, Y_valid_all

X_train_all, X_valid_all, Y_train_all, Y_valid_all = get_all_dat(dat_train,train_label)
#predictions of the new model

optimizer = keras.optimizers.SGD(learning_rate=0.01,momentum=0.0,name='SGD')

output5,pred_aug = tensor_diff_optimizer_model(X_train_all,Y_train_all,X_valid_all,Y_valid_all,X_test,optimizer,optimizer_name='SGD')

submission_aug = submission.copy()

submission_aug['class'] = pred_aug

submission_aug.to_csv('Submission_with_Augmentation.csv')
final_output = pd.concat([output1,output2,output3,output4,output5],axis = 1)

final_output.to_csv('output_matirx.csv')