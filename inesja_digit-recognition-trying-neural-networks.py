import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.optimize import minimize



from subprocess import check_output



import random

import matplotlib.pyplot as pyplot
#constants

num_input = 28 * 28 #images of size 28*28

num_labels = 10 #digits 0-9



# label to vector conversion table

# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

# ...

# 9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

label2vec = np.identity(num_labels)
#helper functions for matrices



#gradient matrices needs to be packed to pass to scipy.optimize.minimize

def packMats(*Mats):

    vecs = [T.reshape(T.size) for T in Mats]

    return np.hstack(vecs)



#shapes of matrices in the packed array packMats()

def packedShapes(*Mats):

    return [T.shape for T in Mats]



#unpack matrices again

def unpackMats(packed, shapes):

    Mats = list()

    offset = 0

    for r, c in shapes:

        size = r * c

        end = offset + size

        Mats.append(packed[offset:end].reshape((r,c)))

        offset = end



    return Mats
#weights 



#randomly initalize weights of layer: #l_in incoming connections and #l_out outgoing connections

def randInitializeWeights(l_in, l_out):

    #l_out * (l_in + 1) ndarray with random values

    epsilon_init = 0.12

    return (2 * epsilon_init) * np.random.rand(l_out, l_in + 1) - epsilon_init



#sigmoid function

def sigmoid(v):

    return 1.0 / (1.0 + np.exp(-v))



#returns cost and packed gradients

def costFunction(packed, shapes, X, y, lambda_):

    Theta1, Theta2 = unpackMats(packed, shapes)

    m = X.shape[0]

    num_labels = Theta2.shape[0]

    Theta1_grad = np.zeros_like(Theta1)

    Theta2_grad = np.zeros_like(Theta2)

    a_0 = np.ones(m)

    

    # input to hidden layer

    Xt = np.vstack((a_0, X.T))

    A2 = sigmoid(Theta1.dot(Xt))



    # hidden to output layers

    A2 = np.vstack((a_0, A2))

    H_theta = sigmoid(Theta2.dot(A2))



    # calculate cost

    Y = [label2vec[y_i] for y_i in y]

    Y = np.array(Y).T

    Tmp = -Y * np.log(H_theta) - (1 - Y) * np.log(1.0 - H_theta)

    J = np.sum(Tmp) + lambda_ / 2.0 * (np.sum((Theta1 * Theta1)[:, 1:]) + np.sum((Theta2 * Theta2)[:, 1:]))

    J /= m



    # backpropagation

    Delta3 = H_theta - Y

    Delta2 = (Theta2.T.dot(Delta3) * A2 * (1 - A2))[1:,:]



    Theta2_grad = Delta3.dot(A2.T)

    Theta2_reg = np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    Theta2_grad += lambda_ * Theta2_reg



    Theta1_grad = Delta2.dot(Xt.T)

    Theta1_reg = np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))

    Theta1_grad += lambda_ * Theta1_reg



    Theta1_grad /= m

    Theta2_grad /= m



    return J, packMats(Theta1_grad, Theta2_grad)
#training

#Neural Network consists of input layer - one hidden layer - output layer

def training(X, y, num_hidden, lambda_, maxiter, initial_Theta1=None, initial_Theta2=None):

    '''

    X:          training data

    y:          training label

    lambda_:    regularization parameter

    maxiter:    maximum iterations

    initial_Theta1: initial weight matrix for input layer to hidden layer

                    if None, random initial parameter is generated

    initial_Theta2: initial weight matrix for hidden layer to output layer

                    if None, random initial parameter is generated

                    

    return: a tuple of weight matrices

    '''

    # initializing parameters

    if initial_Theta1 is None:

        initial_Theta1 = randInitializeWeights(num_input, num_hidden)

    if initial_Theta2 is None:

        initial_Theta2 = randInitializeWeights(num_hidden, num_labels)

    initial_packed = packMats(initial_Theta1, initial_Theta2)

    shapes = packedShapes(initial_Theta1, initial_Theta2)



    # now train it

    costFunc = lambda n : costFunction(n, shapes, X, y, lambda_)

    res = minimize(costFunc, initial_packed, jac=True, method='CG',

            options={'maxiter':maxiter, 'disp':True})



    # the result is packed; unpack it before return

    return unpackMats(res.x, shapes)
#predicting

#Define a function to predict the label (0-9) of a new input image using a pre-trained neural network:

def predict(Theta1, Theta2, X):

    '''

    Theta1: Trained weight matrix for input layer to hidden layer

    Theta2: Trained weight matrix for hidden layer to output layer

    X:      data

        

    return: the predicted label

    '''

    m = X.shape[0]

    num_labels = Theta2.shape[0]

    p = np.empty((m), dtype=int)  # return value

    a_0 = np.array([1])



    for i, a1 in enumerate(X):

        # input to hidden layer

        a1 = np.hstack((a_0, a1)).T

        z2 = Theta1.dot(a1)

        a2 = sigmoid(z2)



        # hidden to output layer

        a2 = np.hstack((a_0, a2)).T

        z3 = Theta2.dot(a2)

        a3 = sigmoid(z3)



        # find the index of the max prediction

        p[i] = np.argmax(a3)



    return p
#start training



# loading training data

data = pd.read_csv('../input/train.csv')

X_tr = data.values[:41949, 1:].astype(float)/255.0 #normalize training data [0...255]->[0...1]

y_tr = data.values[:41949, 0]

print('Training data loaded')



# training neural network

num_hidden = 25

lambda_ = 1     # regularization parameter

maxiter = 50    # max number of iterations

print('Training: size of hidden layer={}, lambda={}, maximum iterations={}'

    .format(num_hidden, lambda_, maxiter))

Theta1, Theta2 = training(X_tr, y_tr, num_hidden, lambda_, maxiter)





    
#make a prediction



# loading test data

X_test = data.values[41950:, 1:].astype(float)/255.0 #normalize training data [0...255]->[0...1]

y_verify = data.values[41950:, 0]

print('Test data loaded')



print('Predicting...')

y_test = predict(Theta1, Theta2, X_test)



#determine statistics and which data samples were incorrectly classified

correct=0

incorrect=0

wrong=np.zeros(y_test.shape) #store index of wrong classified here

i=0;

for x in range(0,y_test.shape[0]):

    if y_test[x]==y_verify[x]:

        correct=correct+1

    else:

        incorrect=incorrect+1

        wrong[i]=x

        i=i+1

        

total=correct+incorrect

    

#show incorrectly classified images and predicitions

print('Correctly classified: {} from {}'.format(correct,total))

print('Incorrectly classified: {} from {}'.format(incorrect,total))



for y in range(0,incorrect):

    id=int(wrong[y])

    print('Test image ID: {}'.format(id))

    print('Correct label: {}'.format(y_verify[id]))

    print('Predicted label: {}'.format(y_test[id]))

    img = np.reshape(X_test[id,:], [28,28])

    pyplot.imshow(img)