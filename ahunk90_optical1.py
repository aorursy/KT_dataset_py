import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv("../input/dataset.csv")
data.head()
data.iloc[:,1024].value_counts()
rows_to_remove = np.where(data.iloc[:,1024].values==1024)

rows_to_remove
dataset = data.drop(data.index[rows_to_remove[0]])
X = dataset.iloc[:,:-1]
X.head()
Y = dataset.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
X_train = X_train.values

X_test = X_test.values

Y_train= Y_train.values

Y_test = Y_test.values
enc = OneHotEncoder()



y_OH_train = enc.fit_transform(np.expand_dims(Y_train,1)).toarray()

y_OH_val = enc.fit_transform(np.expand_dims(Y_test,1)).toarray()

print(y_OH_train.shape)
class FFSN_MultiClass:

  

  def __init__(self, n_inputs, n_outputs, hidden_sizes=[3]):

    self.nx = n_inputs

    self.ny = n_outputs

    self.nh = len(hidden_sizes)

    self.sizes = [self.nx] + hidden_sizes + [self.ny] 



    self.W = {}

    self.B = {}

    for i in range(self.nh+1):

      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

      self.B[i+1] = np.zeros((1, self.sizes[i+1]))

      

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def softmax(self, x):

    exps = np.exp(x)

    return exps / np.sum(exps)



  def forward_pass(self, x):

    self.A = {}

    self.H = {}

    self.H[0] = x.reshape(1,-1)

    for i in range(self.nh):

      self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]

      self.H[i+1] = self.sigmoid(self.A[i+1])

    self.A[self.nh+1] = np.matmul(self.H[self.nh], self.W[self.nh+1]) + self.B[self.nh+1]

    self.H[self.nh+1] = self.softmax(self.A[self.nh+1])

    return self.H[self.nh+1]

  

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.forward_pass(x)

      Y_pred.append(y_pred)

    return np.array(Y_pred).squeeze()

 

  def grad_sigmoid(self, x):

    return x*(1-x) 

  

  def cross_entropy(self,label,pred):

    yl=np.multiply(pred,label)

    yl=yl[yl!=0]

    yl=-np.log(yl)

    yl=np.mean(yl)

    return yl

 

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

    

  def fit(self, X, Y, epochs=100, initialize='True', learning_rate=0.01, display_loss=False):

      

    if display_loss:

      loss = {}

      

    if initialize:

      for i in range(self.nh+1):

        self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

        self.B[i+1] = np.zeros((1, self.sizes[i+1]))

        

    for epoch in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

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

        self.W[i+1] -= learning_rate * (dW[i+1]/m)

        self.B[i+1] -= learning_rate * (dB[i+1]/m)

        

      if display_loss:

        Y_pred = self.predict(X) 

        loss[epoch] = self.cross_entropy(Y, Y_pred)

    

    if display_loss:

      plt.plot(loss.values())

      plt.xlabel('Epochs')

      plt.ylabel('CE')

      plt.show()
ffsn_multi = FFSN_MultiClass(1024,46,[10,10])

ffsn_multi.fit(X_train,y_OH_train,epochs=100,learning_rate=.005,display_loss=True)
Y_pred_train = ffsn_multi.predict(X_train)

Y_pred_train = np.argmax(Y_pred_train,1)



#Y_pred_val = ffsn_multi.predict(X_test)

#Y_pred_val = np.argmax(Y_pred_val,1)



accuracy_train = accuracy_score(Y_pred_train, Y_train)

#accuracy_val = accuracy_score(Y_pred_val, Y_test.round(), normalize=False)



print("Training accuracy", round(accuracy_train, 2))

#print("Validation accuracy", round(accuracy_val, 2))