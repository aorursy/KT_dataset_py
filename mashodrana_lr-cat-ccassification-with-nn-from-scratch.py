import os

import numpy as np

import h5py

import matplotlib.pyplot as plt

from tqdm import tqdm
path = '/kaggle/input/cat-images-dataset'



train_dataset = h5py.File( os.path.join(path,'train_catvnoncat.h5'), 'r' )

test_dataset = h5py.File( os.path.join(path,'test_catvnoncat.h5'), 'r' )



print('Column of train dataset:', list(train_dataset.keys()))

print('Column of test  dataset:', list(test_dataset.keys()))
# process train data

train_x = np.array( train_dataset['train_set_x'] ) # train features

train_y = np.array( train_dataset['train_set_y'] ) # train label

train_classes = np.array( train_dataset['list_classes'] )



# process test data

test_x = np.array( test_dataset['test_set_x'] ) # test features

test_y = np.array( test_dataset['test_set_y'] ) # test label

test_classes = np.array( test_dataset['list_classes'] ) # list of classes





# rehspae the label

train_y = train_y.reshape((1, train_y.shape[0]))

test_y = test_y.reshape( (1, test_y.shape[0]) )



# since classes are same for both train and test

classes = train_classes
plt.imshow(train_x[2])

plt.show()



plt.imshow(train_x[3])

plt.show()
# print the dimension information

print ("train_x shape: ",train_x.shape)

print ("train_y shape: ", train_y.shape)

print ("test_x shape: ", test_x.shape)

print ("test_y shape: ", test_y.shape)
# Get the necessary information of dimension of our data and print them

m_train = train_x.shape[0] # number of trainig example

m_test  = test_x.shape[0]  # number of test example

image_size = train_x[0].shape # size of each image

num_px = image_size[0] # height/width of the each image





print ("Number of training examples:", m_train)

print ("Number of testing examples:", m_test)

print ("Size of each image: ",image_size)

print ("Height/Width of each image:", num_px)

# flattened each image into a column vector

xtrain_flatten = train_x.reshape((train_x.shape[0],-1)).T

xtest_flatten =test_x.reshape((test_x.shape[0],-1)).T



# shpae after flattened

print ("xtrain_flatten shape: ", xtrain_flatten.shape)

print ("train_y shape: ", train_y.shape)

print ("xtest_flatten shape: ", xtest_flatten.shape)

print ("test_y shape: ", test_y.shape)

print ("sanity check after reshaping: ", xtrain_flatten[0:5,0])
# standarization of the dataset

xtrain = xtrain_flatten/255

xtest = xtest_flatten/255

def sigmoid(z):

    a = 1/(1+np.exp(-z))

    return a

# print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
def initialize_parameters(dim):

    w = np.zeros((dim,1))

    b = 0

    

    # Sanity check

    assert w.shape==(dim,1)

    return w,b

def propagation(w, b, X, Y):

    

    m = X.shape[1]

   

    # Forward propagation

    z = np.dot(w.T, X) + b

    A = sigmoid(z)

#     print(A.shape)

    cost = (-1/m)* np.sum(Y * np.log(A) + (1 - Y) * np.log(1-A) )

#     cost = (-1/m)* np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1-A) ))



#     print(cost)



    # Back propagation

    dw = 1/m * np.dot(X, (A-Y).T)

    db = 1/m * np.sum(A-Y)

    

    # Sanity check

    assert w.shape==dw.shape

    cost = np.squeeze(cost)

    grads = {

        'dw':dw,

        'db':db

    }

    

    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    

    costs = []

    for i in tqdm(range(num_iterations)):

        grads, cost = propagation(w=w, b=b, X=X, Y=Y)

        

        # Update the weight

        w = w - learning_rate * grads['dw']

        b = b - learning_rate * grads['db']

        

        if i%100==0:

            costs.append(cost)

        if print_cost and i%100==0:

            print("Cost after iteration %i: %f"%(i, cost))

        

    params = {

        'w':w,

        'b':b

    }



    grads = grads

        

    return params, grads, costs
def predict(w, b, X):

    m = X.shape[1]

    predcited_y = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    A = sigmoid( np.dot(w.T, X)+b )

    

    for i in range(A.shape[1]):

        if A[0][i] > 0.5:

            predcited_y[0][i]=1

        else:

            predcited_y[0][i]=0

#     predcited_y = A>0.5

    return predcited_y
def model(xtrain, ytrain, xtest, ytest, num_iterations=1500, learning_rate=0.5, print_cost=False):

    

    

    # Parameters initialization

    w, b = initialize_parameters(xtrain.shape[0])

    

    # gradient descent

    params, grads, costs = optimize(w, b, xtrain, ytrain, num_iterations, learning_rate, print_cost)

    

    # predictions

    predictted_ytrain = predict(params['w'], params['b'], xtrain) # predict train data

    predictted_ytest  = predict(params['w'], params['b'], xtest) # predict test data

    

    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(predictted_ytrain - ytrain)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(predictted_ytest - ytest)) * 100))

    

    

    d = {

     "costs": costs,

     "Y_prediction_test": predictted_ytest, 

     "Y_prediction_train" : predictted_ytrain, 

     "w" : w, 

     "b" : b,

     "learning_rate" : learning_rate,

     "num_iterations": num_iterations

    }



    return d
d = model(xtrain, train_y, xtest, test_y, 

          num_iterations = 1500, learning_rate = 0.005, print_cost = False)
# Plot learning curve (with costs)

costs = np.squeeze(d['costs'])

plt.plot(costs)

plt.ylabel('cost')

plt.xlabel('iterations (per hundreds)')

plt.title("Learning rate =" + str(d["learning_rate"]))

plt.show()
# Learning curve for differents learning rates

learning_rates = [0.01, 0.001, 0.0001]

models = {}

for learning_rate in learning_rates:

    models[str(learning_rate)] = model(xtrain, train_y, xtest, test_y,

                                      num_iterations = 2000, learning_rate = learning_rate, print_cost = False)



for learning_rate in learning_rates:

    plt.plot( np.squeeze(models[str(learning_rate)]['costs']), label=learning_rate)

plt.title('Learning curve for different learning rates')

plt.xlabel('Iterations (hundreds)')

plt.ylabel('Cost')

legend = plt.legend(loc='upper center', shadow=True)

frame = legend.get_frame()

frame.set_facecolor('0.90')

plt.show()