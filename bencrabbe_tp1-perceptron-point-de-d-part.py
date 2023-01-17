# This Python 3 environment comes with many helpful analytics libraries installed# This Python 3 environment comes with many helpful analytics libraries installed

import sys

import torch        # linear algebra

import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def read_iris(filename,add_bias=False):

    """

    Reads the iris data set, and returns a set of X,Y lists of lists

    """

    X = []

    Y = []

    istream = open(filename)

    istream.readline()       #skips header

    for line in istream:

        fields = line[:-1].split(',') #also removes a newline

        x_values = [float(val) for val in fields[1:-1]]

        if add_bias:

            x_values.append(1.0)

        y        = fields[-1]

        X.append(x_values)

        Y.append(y)

    istream.close()

    return (X,Y)



irisX,irisY = read_iris('/kaggle/input/iris/Iris.csv',add_bias=True)

print(irisX[:3])

print(irisY[:3])
def code_categories(values):

    """

    Args:

        values : a list of values

    Returns. 

        A couple. with a list mapping integers to strings and dict mapping strings to integers

    """

    itos = [elt for elt in sorted(set(values))]

    stoi = dict([(elt,idx) for (idx,elt) in enumerate(itos)]) 

    return itos,stoi



#Those two variables allow to code strings to int back and forth

itos,stoi = code_categories(irisY)

#Example

print(itos)

print(stoi)
def perceptron_predict(params,x_vector):

    """

    Perceptron classification rule

    Args:

      params   (tensor): a matrix with num categories lines and dim(x_vector) columns

      x_vector (tensor): a vector with x predictors

    Returns:

        a vector coding the y scores (as real values)

    """

    return torch.matmul(params,x_vector)

    

def perceptron_argmax(y_vector):

    """

    Returns the index of the argmax category given a predicted y tensor

    Args:

      y_vector (tensor): a vector with predicted scores

    """

    return torch.argmax(y_vector).item()



def hardmax(y_vector):

    """

    Turns an arbitrary real valued vector into a 1-hot encoded vector

    where the max value is coded to 1, all the others to 0.

    Args:

      y_vector (tensor): a vector with predicted scores

    Returns:

      a vector coding the y scores (one hot encoding)

    """

    oh = torch.zeros(len(y_vector))

    oh[torch.argmax(y_vector)] = 1.0

    return oh
def perceptron_train(X,Y,epochs,alpha):

    """

    Takes a dataset, performs SGD and returns the parameters

    Args:

      X (list)    : list of list of predictor values

      Y (list)    : list of string, the predicted values

      epochs (int): number of epochs

      alpha       : float the learning rate

    """

    itos,stoi   = code_categories(Y)

    y_size = len(itos)

    def cat_as_vec(cat_value):

        y = torch.zeros(y_size)

        y[stoi[cat_value]] = 1.0

        return y

        

    X_vectors = [torch.tensor(elt) for elt in X]

    Y_vectors = [cat_as_vec(elt) for elt in Y]

    

    x_size    = len(X[0])

    Weights   = torch.zeros((y_size,x_size))             #zero init

    for e in range(epochs):

        loss = 0

        for (x,yref) in zip(X_vectors,Y_vectors):

            soft_pred   = perceptron_predict(Weights,x) #real numbers

            hard_pred   = hardmax(soft_pred)             #1-hot prediction

            Weights     += (alpha * torch.ger(yref-hard_pred,x)) #ger performs outer product

            if not torch.equal(hard_pred,yref):

                loss += 1

        print('epoch',e,'loss =',loss,'/',len(X),'lr',alpha,file=sys.stderr,flush=True)

        

    print("Weight matrix dimensions",Weights.shape)

    return Weights



params = perceptron_train(irisX,irisY,10,1.0)