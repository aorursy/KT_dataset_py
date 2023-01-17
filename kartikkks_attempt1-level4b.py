import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.model_selection import KFold,train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

sys.path.insert(0,'../input/')

from padhai import MPNeuron, Perceptron, PerceptronWithSigmoid

import matplotlib.pyplot as plt

import statistics



np.random.seed(100)

LEVEL = 'level_4b'
class SigmoidNeuron:

  

  def __init__(self):

    self.w = None

    self.b = None

    

  def perceptron(self, x):

    return np.dot(x, self.w.T) + self.b

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def grad_w_mse(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred) * x

  

  def grad_b_mse(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred)

  

  def grad_w_ce(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    if y == 0:

      return y_pred * x

    elif y == 1:

      return -1 * (1 - y_pred) * x

    else:

      raise ValueError("y should be 0 or 1")

    

  def grad_b_ce(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    if y == 0:

      return y_pred 

    elif y == 1:

      return -1 * (1 - y_pred)

    else:

      raise ValueError("y should be 0 or 1")

  

  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):

    

    # initialise w, b

    if initialise:

      self.w = np.random.randn(1, X.shape[1])

      self.b = 0

      

    if display_loss:

      loss = {}

    

    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      dw = 0

      db = 0

      for x, y in zip(X, Y):

        if loss_fn == "mse":

          dw += self.grad_w_mse(x, y)

          db += self.grad_b_mse(x, y) 

        elif loss_fn == "ce":

          dw += self.grad_w_ce(x, y)

          db += self.grad_b_ce(x, y)

      self.w -= learning_rate * dw

      self.b -= learning_rate * db

      

      if display_loss:

        Y_pred = self.sigmoid(self.perceptron(X))

        if loss_fn == "mse":

          loss[i] = mean_squared_error(Y, Y_pred)

        elif loss_fn == "ce":

          loss[i] = log_loss(Y, Y_pred)

    

    if display_loss:

      plt.plot(loss.values())

      plt.xlabel('Epochs')

      if loss_fn == "mse":

        plt.ylabel('Mean Squared Error')

      elif loss_fn == "ce":

        plt.ylabel('Log Loss')

      plt.show()

      

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.sigmoid(self.perceptron(x))

      Y_pred.append(y_pred)

    return np.array(Y_pred)
def read_all(folder_path, key_prefix=""):

    '''

    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.

    '''

    print("Reading:")



    images = {}

    files = os.listdir(folder_path)

    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):

        file_path = os.path.join(folder_path, file_name)

        image_index = key_prefix + file_name[:-4]

        image = Image.open(file_path)

        image = image.convert("L")

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/level_4b_train/"+LEVEL+"/"+"background", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/level_4b_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_4b_test/kaggle_"+LEVEL, key_prefix='') # change the path

print(len(images_test))
list(images_test.keys())[:5]
X_train = []

Y_train = []

for key, value in images_train.items():

    X_train.append(value)

    if key[:4] == "bgr_":

        Y_train.append(0)

    else:

        Y_train.append(1)



ID_test = []

X_test = []

for key, value in images_test.items():

  ID_test.append(int(key))

  X_test.append(value)

  

        

X_train = np.array(X_train)

Y_train = np.array(Y_train)

X_test = np.array(X_test)



print(X_train.shape, Y_train.shape)

print(X_test.shape)
print(Y_train)
for i in tqdm_notebook(range(X_train.shape[0]), total=X_train.shape[0], unit="train_sample"):#range(X_train.shape[0]):

    for  j in range(X_train.shape[1]):

        if X_train[i,j]>10:

            X_train[i,j] =255



for i in tqdm_notebook(range(X_test.shape[0]), total=X_test.shape[0], unit="test_sample"):

    for  j in range(X_test.shape[1]):

        if X_test[i,j]>10:

            X_test[i,j] = 255            

scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
from mpl_toolkits import mplot3d

class SigmoidNeuronMy:

  

  def __init__(self):

    self.w = None

    self.b = None

    

  def perceptron(self, x):

    return np.dot(x, self.w.T) + self.b

  

  def sigmoid(self, x):

    return 1.0/(1.0 + np.exp(-x))

  

  def grad_w_mse(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred) * x

  

  def grad_b_mse(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    return (y_pred - y) * y_pred * (1 - y_pred)

  

  def grad_w_ce(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    if y == 0:

      return y_pred * x

    elif y == 1:

      return -1 * (1 - y_pred) * x

    else:

      raise ValueError("y should be 0 or 1")

    

  def grad_b_ce(self, x, y):

    y_pred = self.sigmoid(self.perceptron(x))

    if y == 0:

      return y_pred 

    elif y == 1:

      return -1 * (1 - y_pred)

    else:

      raise ValueError("y should be 0 or 1")

  

  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False, setting = False):

    

    # initialise w, b

    if initialise:

      self.w = np.random.randn(1, X.shape[1])

      self.b = 0

      loss = {}

    

    if display_loss:

      loss = {}

    

    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

      dw = 0

      db = 0

      for x, y in zip(X, Y):

        if loss_fn == "mse":

          dw += self.grad_w_mse(x, y)

          db += self.grad_b_mse(x, y) 

        elif loss_fn == "ce":

          dw += self.grad_w_ce(x, y)

          db += self.grad_b_ce(x, y)

      self.w -= learning_rate * dw

      self.b -= learning_rate * db

      

      if display_loss:

        Y_pred = self.sigmoid(self.perceptron(X))

        if loss_fn == "mse":

          loss[i] = mean_squared_error(Y, Y_pred)

        elif loss_fn == "ce":

          loss[i] = log_loss(Y, Y_pred)

    

      if setting:

        Y_pred = self.sigmoid(self.perceptron(X))

        if loss_fn == "mse":

          loss[i] = mean_squared_error(Y, Y_pred)

        elif loss_fn == "ce":

          loss[i] = log_loss(Y, Y_pred)

        #print(loss.values())

        

        

    if setting:

        return list(loss.values())

    if display_loss:

      plt.plot(loss.values())

      plt.xlabel('Epochs')

      if loss_fn == "mse":

        plt.ylabel('Mean Squared Error')

      elif loss_fn == "ce":

        plt.ylabel('Log Loss')

      plt.show()

  

  def optimumValue(self , X , Y, loss_fn="mse"):

        lr = np.linspace(0.000005, 0.00003, 6)

        #epochs = np.linspace(0, 10, 11)

        epochs = np.array([500])

        acc = {}

        min_loss = 1000

        lr_fin = 0

        epoch_fin = 0

        std = 0

        fin = []

        for j in tqdm_notebook(range(len(lr)), total=len(lr), unit="lr"):

            for i in tqdm_notebook(range(len(epochs)), total=len(epochs), unit="epoch"):

                for k in range (4):

                    X1_train, X1_test, y1_train, y1_test = train_test_split(X, Y, test_size=0.2, random_state=k**2)

                    self.fit(X1_train, y1_train, epochs=epochs[i].astype("int")+1, learning_rate=lr[j], loss_fn="ce", display_loss=True)

                    Y_pred_train = self.predict(X1_test)

                    Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

                    acc[k] = accuracy_score(Y_pred_binarised_train, y1_test)   

                std=statistics.stdev(list(acc.values()))

                accuracy = max(list(acc.values()))

                fin.append([accuracy,std,epochs[i].astype("int")+1,lr[j]])    



        print(fin)           

                       

                

  def predict(self, X):

    Y_pred = []

    for x in X:

      y_pred = self.sigmoid(self.perceptron(x))

      Y_pred.append(y_pred)

    return np.array(Y_pred)

#sn_mse = SigmoidNeuronMy()

#sn_mse.optimumValue(X_scaled_train, Y_train,loss_fn="mse")

#sn_mse.fit(X_scaled_train, Y_train, epochs=81, learning_rate=0.05, loss_fn="mse", display_loss=True)

#sn_mse.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.015, loss_fn="mse", display_loss=True)
sn_ce = SigmoidNeuronMy()

#sn_ce.optimumValue(X_scaled_train, Y_train,loss_fn="ce")

#sn_ce.fit(X_scaled_train, Y_train, epochs=71, learning_rate=0.03, loss_fn="ce", display_loss=True)

sn_ce.fit(X_scaled_train, Y_train, epochs=800, learning_rate=0.00005, loss_fn="ce", display_loss=True)



#acc = {}



#for k in range (5):

 #   X1_train, X1_test, y1_train, y1_test = train_test_split(X_scaled_train, Y_train, test_size=0.2, random_state=k**2)

  #  sn_ce.fit(X1_train, y1_train, epochs=700, learning_rate=0.00005, loss_fn="ce", display_loss=True)

   # Y_pred_train = sn_ce.predict(X1_test)

    #Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

   # acc[k] = accuracy_score(Y_pred_binarised_train, y1_test)



#std=statistics.stdev(list(acc.values()))

#print(std)

#accuracy = max(list(acc.values()))  

#print(accuracy)

    
def print_accuracy(sn):

  Y_pred_train = sn.predict(X_scaled_train)

  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)
#print_accuracy(sn_mse)

print_accuracy(sn_ce)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)