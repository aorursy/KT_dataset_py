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
train = pd.read_csv("../input/ee-769-assignment1/train.csv") 

test = pd.read_csv("../input/ee-769-assignment1/test.csv")
test.head()
train.shape
train.groupby('Attrition').mean()
columns_to_drop = ["ID", ]
X_train = train.drop('Attrition', axis=1)

X_train = X_train.drop('ID', axis=1)

Y_train = train['Attrition']

Y_train_ID = train['ID']
X_test = test.drop("ID", axis =1)

                   

Y_test_ID = test["ID"]
Y_train.head()
train.groupby('Attrition').mean()
column = ["BusinessTravel", "Department", "EducationField", "MaritalStatus","JobRole","Gender","OverTime"]

for col in column:

  

          X_train[col] = pd.Categorical(X_train[col], categories=X_train[col].unique()).codes

          X_test[col] = pd.Categorical(X_test[col], categories=X_test[col].unique()).codes
X_train.mean()
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[0,1])
X_train_final=X_binarised_train.values
from sklearn.metrics import accuracy_score
for x, y in zip(X_binarised_train, Y_train):

    print(x,y)
for b in range(X_train_final.shape[1] + 1):

  Y_pred_train = []

  accurate_rows = 0



  for x, y in zip(X_train_final, Y_train):

    y_pred = (np.sum(x) >= b)

    Y_pred_train.append(y_pred)

    accurate_rows += (y == y_pred)



  print(b, accurate_rows/X_train_final.shape[0])  
class Perceptron:

  

  def __init__ (self):

    self.w = None

    self.b = None

    

  def model(self, x):

    return 1 if (np.dot(self.w, x) >= self.b) else 0

    

  def predict(self, X):

    Y = []

    for x in X:

      result = self.model(x)

      Y.append(result)

    return np.array(Y)

    

  def fit(self, X, Y, epochs = 1, lr = 1):

    

    self.w = np.ones(X.shape[1])

    self.b = 0

    

    accuracy = {}

    max_accuracy = 0

    

    wt_matrix = []

    

    for i in range(epochs):

      for x, y in zip(X, Y):

        y_pred = self.model(x)

        if y == 1 and y_pred == 0:

          self.w = self.w + lr * x

          self.b = self.b - lr * 1

        elif y == 0 and y_pred == 1:

          self.w = self.w - lr * x

          self.b = self.b + lr * 1

          

      wt_matrix.append(self.w)    

          

      accuracy[i] = accuracy_score(self.predict(X), Y)

      if (accuracy[i] > max_accuracy):

        max_accuracy = accuracy[i]

        chkptw = self.w

        chkptb = self.b

        

    self.w = chkptw

    self.b = chkptb

        

    print(max_accuracy )

    print(self.w, self.b)

    

    plt.plot(accuracy.values())

    plt.ylim([0, 1])

    plt.show()

    

    return (np.array(wt_matrix ))
perceptron = Perceptron()
Y_train = Y_train.values
X_train = X_train.values
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.fit_transform(X_test)



wt_matrix = perceptron.fit(X_scaled_train, Y_train, 100, 0.01)
X_scaled_train[0]
W = [-6.47254073e-02 ,  4.16776931e-02, -3.28691849e-02, -4.60176912e-02,

  1.21628179e-01,  3.15315892e-02,  4.32755143e-03,  1.00000000e+00,

  9.66052433e-02, -1.92663140e-01,  7.24771948e-02,  3.72492650e-02,

 -8.21243185e-02, -4.59104677e-02,  9.27900277e-02, -1.24194565e-01,

 -6.27893923e-02, -1.12399270e-01, 2.41600805e-02,  1.51014372e-01,

 -2.48282073e-01, -2.24602129e-02,  2.26699295e-02, -1.11820370e-01,

 -1.26168203e-01, -2.23697862e-01, -4.74712868e-02, -3.58182049e-02,

  6.89639001e-04, -1.27930235e-01,  1.50942542e-01, -1.20095775e-01]
b = 0.7100000000000002
X_scaled_test[0]
def model(X, W):

    b = 0.7100000000000002

    return 1 if (np.dot(X, W) >= b  ) else 0
Y_pred = []

for X in X_scaled_test:

    Y_pred.append(model(X,W))
for i in range(len(Y_pred)):

    Y_pred[i] = 0

    
Y_pred
X_scaled_test.shape
len(Y_pred)
submission = {}

submission['ID'] = Y_test_ID

submission['Attrition'] = Y_pred



submission = pd.DataFrame(submission)

submission = submission[['ID', 'Attrition']]

submission.to_csv("output5.csv", index=False)
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import matplotlib.colors

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error

from tqdm import tqdm_notebook
class SigmoidNeuron:

  

  def __init__(self):

    self.w = None

    self.b = None

    

  def perceptron(self, x):

    return np.dot(x, self.w.T) + self.b

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def grad_w(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred) * x

  

  def grad_b(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred)

  

  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):

    

    # initialise w, b

    if initialise:

      self.w = np.random.randn(1, X.shape[1])

      self.b = 0

      

    if display_loss:

      loss = []

    

    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      dw = 0

      db = 0

      for x, y in zip(X, Y):

        dw += self.grad_w(x, y)

        db += self.grad_b(x, y)       

      self.w -= learning_rate * dw

      self.b -= learning_rate * db

      

      if display_loss:

        Y_pred = self.sigmoid(self.perceptron(X))

        loss.append(mean_squared_error(Y_pred, Y))

    print(self.w,self.b)

    

    if display_loss:

      plt.plot(loss)

      plt.xlabel('Epochs')

      plt.ylabel('Mean Squared Error')

      plt.show()

    

      

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.sigmoid(self.perceptron(x))

      Y_pred.append(y_pred)

    return np.array(Y_pred)
sn = SigmoidNeuron()
sn.fit(X_scaled_train, Y_train, epochs=500, learning_rate=0.005, display_loss=True)
Y_pred_train = sn.predict(X_scaled_train)

Y_pred_test = sn.predict(X_scaled_test)
scaled_threshold = 0.5

Y_pred_binarised_train = (Y_pred_train > scaled_threshold).astype("int").ravel()

Y_pred_binarised_test = (Y_pred_test > scaled_threshold).astype("int").ravel()

accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

print(Y_pred_binarised_test)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 2000, random_state = 42)
rf.fit(X_train, Y_train)
Y_pred_binarised_test = rf.predict(X_test)

Y_pred_binarised_test = Y_pred_binarised_test.round()


 
Y_pred_binarised_test_reverse = []

for item in Y_pred_binarised_test:

    if item == 0:

        Y_pred_binarised_test_reverse.append("1")

    else :

        Y_pred_binarised_test_reverse.append("0")
class FFSNNetwork:

  

  def __init__(self, n_inputs, hidden_sizes=[2]):

    self.nx = n_inputs

    self.ny = 1

    self.nh = len(hidden_sizes)

    self.sizes = [self.nx] + hidden_sizes + [self.ny]

    

    self.W = {}

    self.B = {}

    for i in range(self.nh+1):

      self.W[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])

      self.B[i+1] = np.zeros((1, self.sizes[i+1]))

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def forward_pass(self, x):

    self.A = {}

    self.H = {}

    self.H[0] = x.reshape(1, -1)

    for i in range(self.nh+1):

      self.A[i+1] = np.matmul(self.H[i], self.W[i+1]) + self.B[i+1]

      self.H[i+1] = self.sigmoid(self.A[i+1])

    return self.H[self.nh+1]

  

  def grad_sigmoid(self, x):

    return x*(1-x) 

    

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

      

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.forward_pass(x)

      Y_pred.append(y_pred)

    return np.array(Y_pred).squeeze()
X_train.shape[1]
ffsnn = FFSNNetwork(X_train.shape[1], [2])

ffsnn.fit(X_scaled_train, Y_train, epochs=1000, learning_rate=.01, display_loss=False)
from sklearn.metrics import accuracy_score, mean_squared_error

Y_pred_test = ffsnn.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()





submission = {}

submission['ID'] = Y_test_ID

submission['Attrition'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ID', 'Attrition']]

submission.to_csv("output11.csv", index=False)


