import seaborn as sns

import sys, pickle

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/digit-recognizer/train.csv")



train_y = list(df_train["label"])

train_X = np.array( df_train.drop("label",axis = 1))



sns.countplot(train_y)

plt.show()

plt.imshow(train_X[0].reshape((28,28)),cmap = "gray_r")
test_X = np.array(pd.read_csv("../input/digit-recognizer/test.csv"))
# Normalizing is important because it prevents gradient explosions



train_X = train_X / 255.0



test_X = test_X / 255.0
def create_batches(X,y,batch_size = 16):

    output_X = []

    output_y = []

    

    

    num_batches = int(len(X) / batch_size)

    num_mod = np.mod(len(X),batch_size)

    x = 0

    for i in range(num_batches):

        output_X.append(X[x:x+batch_size])

        output_y.append(y[x:x+batch_size])

        x = x + batch_size

    if num_mod != 0:

        if num_mod == 1:

            output_X.append(X[-1])

            output_y.append(y[-1])

        else:

            output_X.append([X[-(num_mod):-1][0],X[-1]])

            output_y.append([y[-(num_mod):-1][0],y[-1]])

        

    return output_X,output_y



train_X,train_y = create_batches(train_X,train_y)
# Layer

class layer_dense:

    def __init__(self,n_inputs,n_neurons):

        #randomly initialize the weights an biases

        self.weights = np.random.randn(n_inputs,n_neurons) * 0.1 # initialize it with the random numbers between 0 and 1

        self.biases = np.zeros((1,n_neurons)) # initialize it with 0

        self.name = "layer_dense"

    def forward(self,inputs):

        self.input = inputs # Save the input for the backpropagation algo

        self.output = np.dot(inputs,self.weights) + self.biases # a * w + b

        return self.output

# Activations

class ReLU:

    def __init__(self):

        self.name = "relu"

    def forward(self,inputs):

        self.input = inputs

        self.output = np.maximum(0,inputs) # Just looking if the number is bigger than zero

        return self.output
# seed it

np.random.seed(10)



# You can simply just add things to the network by adding objects to this list

net = [

    layer_dense(784,200),

    ReLU(),

    layer_dense(200,10),

]
# Hyperparameters

reg = 1e-3

step_size = 1e-3 # learning rate



num_samples = len(train_X[0])



# Runs the network

def forward(inputs,net):

    out = inputs

    for layer in net:

        out = layer.forward(out)

    return out



def softmax(output):

    # Get the probabilities with softmax

    exp = np.exp(output)

    probs = exp / np.sum(exp, axis=1, keepdims=True)

    return probs



def backpropagation(net,dscores):

    n = 1

    for layer in reversed(net): # The loop starts at the end of the network

        if layer.name == "relu":

            # backpropagate the ReLU non-linearity

            dscores[layer.output <= 0] = 0

        else:

            dW = 0

            dB = 0

            

            # backpropate the gradient to the parameters

            dW = np.dot(layer.input.T, dscores)

            db = np.sum(dscores, axis=0, keepdims=True)

            

            # Check if it is the last layer 

            # if it isn't it will backpropagate into the next layer

            if n != len(net):

                dscores = np.dot(dscores, layer.weights.T)

                

            #Apply regularization

            dW += reg*layer.weights

            

            #Apply gradients

            layer.weights += -step_size * dW

            layer.biases += -step_size * db

        n += 1

        

def train(epochs):

    for i in range(epochs):

        for i in range(len(train_X)-1):

            

            # Get the current batch

            feat = train_X[i]

            target = train_y[i]

            

            # Run batch trough the network

            output = forward(feat,net)

            

            # Get the probabilities with softmax

            probs = softmax(output)

            

            # Compute the gradient of the output

            dscores = probs

            dscores[range(num_samples),target] -= 1

            dscores /= num_samples

            dscores[output <= 0] = 0



            backpropagation(net,dscores)

            

            sys.stdout.write(f"\rTraining - {round(i / len(train_X) * 100)}%")

            sys.stdout.flush()
# You can train for more epochs if you want

EPOCHS = 10



train(EPOCHS)
# Predict on test data

out = forward(test_X,net)

# Apply sotmax and argmax

sub = softmax(out).argmax(axis = 1)

sns.countplot(sub)

plt.show()

# Show the distribution of the prediction
sub_df = pd.read_csv("../input/digit-recognizer/sample_submission.csv")



sub_df["Label"] = sub 



sub_df.to_csv("submission.csv",index = False)



sub_df.head()