# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def get_unique_chars(input):

    

    return set(input)
def softmax(x):

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)
def one_hot(input):

    

    integer_encoded = [char_to_int[char] for char in input]

    one_hot_encoded = []

    

    for value in integer_encoded:

        

        letter = [0 for _ in range(len(unique_chars))]

        letter[value] = 1

        one_hot_encoded.append(letter)

        

    return one_hot_encoded
def softmax(x):

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)
## Simple RNN Cell - only one neuron, which takes xt and previous hidden layer o/p as inputs and returns anext and yt



def rnn_cell(xt,a_prev,parameters):

    

    # Retrieve values from parameters (dictionary)

    Wax = parameters["Wax"]

    Waa = parameters["Waa"]

    Wya = parameters["Wya"]

    ba = parameters["ba"]

    by = parameters["by"]

        

    # Compute next activation 

    a_next = np.tanh(np.dot(Wax,xt) + np.dot(Waa,a_prev) + ba)

    

    # Compute output of current cell 

    yt_pred = softmax(np.dot(Wya,a_next) + by)  

    

    cache = (a_next,a_prev,xt,parameters)

    

    return a_next,yt_pred,cache
## Main Processing block - which will call individual RNN Cells iteratively ....



def RNN_main(x,a0,parameters,y_one_hot):

    

    caches = []

    a = np.zeros((n_a, m, tx))

    y_pred = np.zeros((n_y, m, tx))

    

    a_next = a0

    loss = 0

    loss_per_timestamp = 0

    

    for t in range(tx):

        

        a_next, yt_pred, cache = rnn_cell(x[:,:,t].T,a_next,parameters)

        a[:,:,t] = a_next

                

        y_pred[:,:,t] = yt_pred.T

        

        caches.append(cache)

        loss_per_timestamp = -np.dot(y_one_hot[:,:,t] , np.log(yt_pred))

        loss = loss + loss_per_timestamp

        

    caches = (caches,x)

    

    return a, y_pred, caches , loss

## Get predicted string from y_pred values ...



def get_predicted_string(y_pred):

    

    predicted_string = ''

    idx_for_prediction = (np.squeeze(y_pred)).argmax(axis = 1)

    

    for key in idx_for_prediction:

        

        predicted_string = predicted_string + int_to_char[key]

        

    return predicted_string

        

    
## Initialize parameters for RNN ...



def initialize_parameters():

    

    np.random.seed(1234)



    a0 = np.random.randn(n_a,1) 

    Waa = np.random.randn(n_a,n_a) * 0.1

    Wax = np.random.randn(n_a,n_x) * 0.1

    Wya = np.random.randn(ty,n_a) * 0.1



    ba = np.random.randn(n_a,1) 

    by = np.random.randn(ty,1) 



    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

    

    return a0, parameters
def initialize_grads(parameters):

    

    grads = {}

    

    dWax = np.zeros((parameters.get('Wax').shape))

    dWya = np.zeros((parameters.get('Wya').shape))

    dWaa = np.zeros((parameters.get('Waa').shape))

    dba = np.zeros((parameters.get('ba').shape))

    dby = np.zeros((parameters.get('by').shape))

    

    grads = {'dWax':dWax,'dWya':dWya,'dWaa':dWaa,'dba':dba,'dby':dby }

    

    return grads
## This is a backward propagation code for one time stamp .... This block of code will be called iteratively over the entire sequence lengths to calculate gradients 



def RNN_backprop_cell(cache_t,y_t,y_pred_t):

        

    a_next_t, a_prev_t,xt,parameters = cache_t

    

    ## Loss of loss function wrt itself is 1

    dL = 1

    

    ## Change of Loss wrt y_pred(t)

    dy_pred_t = (y_pred_t - y_t) * dL

    

    ## Change of by wrt Loss function .... 

    dby_t = dy_pred_t * 1                        ## Will be used to update parameter while learning 

    

    ## Change of Wya wrt Loss function ...

    dWya_t = np.dot(a_next_t , dy_pred_t)         ## Will be used to update the parameter while learning 

    

    ## Change of a_next wrt Loss function

    da_next_t = np.dot(parameters.get('Wya').T , dy_pred_t.T) 

    

    ## Dervative of Loss wrt Tanh

    dtanh = (1 - a_next_t ** 2) * da_next_t

    

    ## Change of Wax wrt Loss function

    dWax_t = np.dot(xt , dtanh.T)                  ## Will be used to update parameter while learning 

    

    # Change of Waa wrt Loss function 

    dWaa_t = np.dot(a_prev_t , dtanh.T)            ## Will be used to update parameter while learning 

    

    ## Change of ba wrt Loss function 

    dba_t = np.sum(dtanh,axis=1,keepdims=1) 

    

    grads_t = {'dWax_t':dWax_t,'dWaa_t':dWaa_t,'dWya_t':dWya_t,'dby_t':dby_t,'dba_t':dba_t}

    

    return grads_t
## Back prop code for all time sequences ... this will iteratively call RNN_backprop_cell() function 



def RNN_backprop_main(grads,parameters,tx,caches,y_one_hot):

    

    cache,x = caches

    

    for t in reversed(range(tx)):

        

        grads_t = RNN_backprop_cell(cache[t],y_one_hot[:,:,t],y_pred[:,:,t])

        grads['dWax'] = grads.get('dWax') + grads_t.get('dWax_t').T

        grads['dWaa'] = grads.get('dWaa') + grads_t.get('dWaa_t')

        grads['dWya'] = grads.get('dWya') + grads_t.get('dWya_t').T

        grads['dby'] = grads.get('dby') + grads_t.get('dby_t').T

        grads['dba'] = grads.get('dba') + grads_t.get('dba_t')

    

    return grads

    
## Use gradient clipping to avoid exploding / vanishing gradients ..



def clip_gradients(gradients,min_val,max_val):

    

    for grad in gradients.keys():

        

        np.clip(gradients[grad],min_val,max_val,out=gradients[grad])

        

    return gradients
## This function will update the gradients basis learning rate and gradients. We will use Gradient Descent Algorithm for this



def update_gradients_GD(parameters,gradients,learning_rate):

    

    parameters['Wax'] = parameters.get('Wax') - (learning_rate * gradients.get('dWax'))

    parameters['Waa'] = parameters.get('Waa') - (learning_rate * gradients.get('dWaa'))

    parameters['Wya'] = parameters.get('Wya') - (learning_rate * gradients.get('dWya'))

    parameters['by'] = parameters.get('by') - (learning_rate * gradients.get('dby'))

    parameters['ba'] = parameters.get('ba') - (learning_rate * gradients.get('dba'))

    

    return parameters

    
# Input for RNN - this is a shuffled version of ground truth "y"

X = 'nzdyapxeoumhlvsqcwgrkibtfj'



## Ground truth (y) for the model ...

y = 'abcdefghijklmnopqrstuvwxyz'



## Get Unique characters from the output 

#alphabet = 'abcdefghijklmnopqrstuvwxyz'

unique_chars = get_unique_chars(y)



## Create 2 dictionaries which will have mapping of integers to characters and vice-versa

char_to_int = dict((c,i) for i,c in enumerate((unique_chars)))

int_to_char = dict((i,c) for i,c in enumerate((unique_chars)) )



## Hyperparameters that we need will use 

learning_rate = 0.01

epochs = 6000

losses = []

loss = 0





## Convert y to One Hot Vectors 

y_one_hot = np.array([one_hot(y)])    ## RNN module uses ony One Hot Encodec vectors for calculations.



## Convert X to One Hot Vectors 

X_one_hot = np.array([one_hot(X)])    ## RNN module uses only One Hot Encoded Vectors for calculations.



## Calculate sequence lengths to initialize Weights ....

tx =len(X)      ## Sequence length for input .... 

ty = len(y)     ## Sequence length for output .. In this example, we are keeping tx = ty 

n_a = 50        ## Hidden number of units 

m = len(X)      ## Number of inputs for learning 

n_x = len(X)    ## Input per time sequence - One hot encoded encoded vector for input at one timestamp

n_y = 1         ## Output chars





## Initialize parameters for model ... 

a0,parameters = initialize_parameters()



print("Input to system is ", X, " and the expected prediction is ", y)

print("============== Learning starts ===================================================")



## Loop through the epochs for learning ....

for epoch in range(epochs):



    ## This code executes forward propagation module - One RNN cell is called recursively for the duration of the input length

    a, y_pred, caches, loss = RNN_main(X_one_hot, a0, parameters,y_one_hot)

    print("Total Loss at epoch ",(epoch + 1 ), " is ", np.squeeze(loss))



    ## Print the predicted string based forward propagation

    TGREEN =  '\033[32m' # Green Text

    ENDC = '\033[m'

    print(TGREEN + "Predicted string at epoch ",(epoch + 1 )," is " ,get_predicted_string(y_pred), " and expected string is " , y ," and the input given was ", X , ENDC)



    ## Initialize back prop gradients 

    grads = initialize_grads(parameters)



    ## Get gradients via Back propagation method 

    gradients = RNN_backprop_main(grads,parameters,tx,caches,y_one_hot)



    ## Clip Gradients to keep them within a specific range - This will avoid Vanishing / Exploding gradients issue

    gradients = clip_gradients(gradients,-0.5,0.5)

    

    ## Update parameters based learnig rate and gradients

    parameters = update_gradients_GD(parameters,gradients,learning_rate)



    ## Append losses per epoch in a list so that they can be plotted 

    losses.append(np.sum(loss) / len(X))



print("=========================== Learning finished !!!!!!!! ==========================================")   



parameters.clear()

gradients.clear()

grads.clear()

X_one_hot = []

X = ""

y = ""

y_one_hot = []

    
## Plot losses in a graph ....

plt.rcParams['figure.figsize'] = 18,8

_ = plt.plot(losses)