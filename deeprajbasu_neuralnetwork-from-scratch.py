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
##softmax_crossentropy method to calculate the loss



def softmax_crossentropy_with_logits(logits,reference_answers):

        

        logits_for_answers = logits[np.arange(len(logits)),reference_answers]

    

        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))

    

        return xentropy

    

def grad_softmax_crossentropy_with_logits(logits,reference_answers):

        

        ones_for_answers = np.zeros_like(logits)

        ones_for_answers[np.arange(len(logits)),reference_answers] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)

        

        

        return (- ones_for_answers + softmax) / logits.shape[0]
#basic layer class

class layer : 

    

    #initialize parameters

    def __init__(self):

        pass

    

    #accept input and return output

    def forward(self,X):

        return X

    

    def backward(self,X,gradients):

        #calculate loss

        

        num_units = X.shape[1]

        

        #create 2d matrix of shape num_units

        #matrix to be all zeros and 1s diagonal

        d_layer_d_input = np.eye(num_units)

        

        return np.dot(gradients, d_layer_d_input) # chain rule

        
class relu (layer):

    def __init__(self):

        pass

    def forward(self,X):

        

        #implimenting relu function for non linearity 

        return np.maximum(0,X)

        

        #pass

        

    def backward(self,X,grad_output):

        #comput gradient of loss

        

        relu_grad = X>0

        return grad_output*relu_grad

    

    
class dense(layer):

    

    def __init__(self,input_units,output_units,lr=0.1):

        #initialise learning rate, weights and bias

        #perform weights * inputs + bias

        

        #initialize learning rate

        self.lr = lr

        

        # initializing weights using  Xavier initialization

        #weights need to be of a shape (input,output)

        self.w  = np.random.normal(loc=0.0,

                                   scale = np.sqrt(2/(input_units+output_units)),

                                   size = (input_units,output_units))

        

        #initialize biases

        #weights need to be of length output, or as many as number of outputs

        self.b = np.zeros(output_units)

        

    def forward(self,X):

        

        #perform weights * inputs + bias

        return np.dot(X,self.w)+self.b

    

    def backward(self,X,grad_output):

        ##implementing gradient dissent

        

        #calculate gradiant input

        grad_input = np.dot(grad_output,self.w.T)

        

        #calculate gradient for weights and bias

        gradient_w = np.dot(X.T,grad_output)

        gradient_b = grad_output.mean(axis=0)*X.shape[0]

        

        #make sure all the matricies are of the correct shape 

        

        assert gradient_w.shape == self.w.shape and gradient_b.shape == self.b.shape

        

        

        

        #update the weights and bias using respecting gradients

        #gradient descent

        self.w  =self.w - self.lr*gradient_w

        self.b  = self.b - self.lr *gradient_b

        

        return grad_input

    

    

    #loss function

    

    

        
import sklearn.datasets

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



#load the breast cancer data

data = sklearn.datasets.load_breast_cancer()



df = pd.DataFrame(data.data, columns = data.feature_names)

df["TARGET"] = data.target



df.head(4)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



#label and features

X = df.drop("TARGET",axis=1)

Y = df["TARGET"]



X = scaler.fit_transform(X)

X



X = pd.DataFrame(X, columns=df.drop("TARGET",axis = 1).columns)



type(X.iloc[0][0])



#split data to train and test 

X_train, X_test, y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state = 1)

#appending the layers to the network

network = []

network.append(dense(X_train.shape[1],100))

network.append(relu())

network.append(dense(100,200))

network.append(relu())

network.append(dense(200,10))



def activate(network,X):

    

    #return list of activation for each layer in network

    

    Input = X

    activations=[]

    

    #loop through network

    

    for l in network:

        

        activations.append(l.forward(Input))

        

        #update input to the last output

        

        Input = activations[-1]

    assert len(activations) == len(network)

    return activations



def predict(network,X):

    

    #make predictions using forawd method

    #use activation to predict for input X

    preds = activate(network,X)[-1]

    return preds.argmax(axis=-1)
def train(netword,X,y):

    #first run forward to get the activation for all layers

    #then back layer to optimize weight with gradient dissent

    

    layer_activations = activate(network,X)

    layer_inputs = [X]+layer_activations

    

    logits = layer_activations[-1]

    

    

    # Compute the loss and the initial gradient

    loss = softmax_crossentropy_with_logits(logits,y)

    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)

    

    

    #back propagate weights

    

    #loop in reverse

    for layer_index in range(len(network))[::-1]:

        layer = network[layer_index]

        

        #feeding the layers,back propagation method to update its weights and bias

        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad)

        

    return np.mean(loss)
for epoch in range(402):

    train(network,X_train.to_numpy(),y_train.to_numpy())
# #training loop 

# epochs = 12



# for _ in range(epochs):

#     train(network,X_train,Y_train)
# clear_output()

# print("Epoch",epoch)

# print("Train accuracy:",train_log[-1])

# print("Val accuracy:",val_log[-1])

# plt.plot(train_log,label='train accuracy')

# plt.plot(val_log,label='val accuracy')

# plt.legend(loc='best')

# plt.grid()

# plt.show()
# plt.imshow(X_train[7].reshape([28,28]),cmap='gray'),y_train[7]
from sklearn.metrics import balanced_accuracy_score

predicts = predict(network,X_test.to_numpy())



balanced_accuracy_score(predicts,Y_test)


