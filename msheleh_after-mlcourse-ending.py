# exampe of feed-forward neural network(FFNN) from scratch with numpy

# find more different ML models in Math of Intelligence course by Siraj Raval



import numpy as np

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt





class FFNNetwork:

  

  #initialize number of networks inputs list of hidden neurons number in each layer

  def __init__(self, n_inputs, hidden_sizes=[2]):

    #intialize the inputs

    self.nx = n_inputs

    self.ny = 1

    self.nh = len(hidden_sizes)

    self.sizes = [self.nx] + hidden_sizes + [self.ny]

    

    self.W = {}

    self.B = {}

    for i in range(self.nh+1):

      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

      self.B[i+1] = np.zeros((1, self.sizes[i+1]))

  

  # just a simple sigmoid activation function

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  

  # initialize weights and biases as W and H dictionaries

  def forward_pass(self, x):

    self.A = {}

    self.H = {}

    self.H[0] = x.reshape(1, -1)

    for i in range(self.nh+1):

      # compute A and H as pre- and past- processing activation

      self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]

      self.H[i+1] = self.sigmoid(self.A[i+1])

    return self.H[self.nh+1]

  

  

  # compute sigmoid gradient

  def grad_sigmoid(self, x):

    return x*(1-x) 

  

  

  # compute the partial derivatives of the parameters with respect to the loss function

  def grad(self, x, y):

    self.forward_pass(x)

    self.dW = {}

    self.dB = {}

    self.dH = {}

    self.dA = {}

    L = self.nh + 1

    self.dA[L] = (self.H[L] - y)

    for k in range(L, 0, -1):

      self.dW[k] = np.matmul(self.H[k-1].T, self.dA[k])

      self.dB[k] = self.dA[k]

      self.dH[k-1] = np.matmul(self.dA[k], self.W[k].T)

      self.dA[k-1] = np.multiply(self.dH[k-1], self.grad_sigmoid(self.H[k-1]))

  

  

  # update weights -> add a bias -> activate

  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):

    

    # initialise w, b

    if initialise:

      for i in range(self.nh+1):

        self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

        self.B[i+1] = np.zeros((1, self.sizes[i+1]))

      

    if display_loss:

      loss = {}

    

    for e in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      dW = {}

      dB = {}

      for i in range(self.nh+1):

        dW[i+1] = np.zeros((self.sizes[i], self.sizes[i+1]))

        dB[i+1] = np.zeros((1, self.sizes[i+1]))

      for x, y in zip(X, Y):

        self.grad(x, y)

        for i in range(self.nh+1):

          dW[i+1] += self.dW[i+1]

          dB[i+1] += self.dB[i+1]

        

      m = X.shape[1]

      for i in range(self.nh+1):

        self.W[i+1] -= learning_rate * dW[i+1] / m

        self.B[i+1] -= learning_rate * dB[i+1] / m

      

      if display_loss:

        Y_pred = self.predict(X)

        loss[e] = mean_squared_error(Y_pred, Y)

    

    if display_loss:

      plt.plot(loss.values())

      plt.xlabel('Epochs')

      plt.ylabel('Mean Squared Error')

      plt.show()

  

  

  #compute the predicted value for each input by calling the forward_pass

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.forward_pass(x)

      Y_pred.append(y_pred)

    return np.array(Y_pred).squeeze()
# example of FFNN in pytorch is like a plain english



import torch.nn as nn





class FeedforwardNeuralNetModel(nn.Module):

  

    #initialize input, hidden and output dimensions size

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(FeedforwardNeuralNetModel, self).__init__()

        # fc stands for fully connected layer

        self.fc1 = nn.Linear(input_dim, hidden_dim) 



        self.sigmoid = nn.Sigmoid()



        self.fc2 = nn.Linear(hidden_dim, output_dim)  



        

    def forward(self, x):

        out = self.fc1(x)



        out = self.sigmoid(out)



        out = self.fc2(out)

        return out