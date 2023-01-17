import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

from tqdm.notebook import tqdm

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt



np.random.seed(100)

LEVEL = 'level_1'

os.chdir(r'../working')
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

    

    for i in tqdm(range(epochs), total=epochs, unit="epoch"):

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

      plt.plot(np.array(list(loss.values())).astype(float))

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
class SigmoidNeuron_V:

  

  def __init__(self):

    self.W = None

    self.b = None

    

  def perceptron(self, X):

    return np.dot(X, self.W.T) + self.b

  

  def sigmoid(self, X):

    return 1.0/(1.0 + np.exp(-X))

  

  def grad_w_mse(self, X, y):

    y_pred = self.sigmoid(self.perceptron(X))

    return np.matmul(((y_pred - y.reshape(y_pred.shape[0], 1)) * y_pred * (1 - y_pred)).T, X)

  

  def grad_b_mse(self, X, y):

    y_pred = self.sigmoid(self.perceptron(X))

    return np.sum((y_pred - y.reshape(y_pred.shape[0], 1)) * y_pred * (1 - y_pred))

  

  def grad_w_ce(self, X, y):

    y_pred = self.sigmoid(self.perceptron(X))

    return np.matmul((y_pred - y.reshape(y_pred.shape[0], 1)).T, X)

    

  def grad_b_ce(self, X, y):

    y_pred = self.sigmoid(self.perceptron(X))

    return np.sum((y_pred - y.reshape(y_pred.shape[0], 1)))

  

  def fit(self, X, y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):

    

    # initialise w, b

    if initialise:

      self.W = np.random.randn(1, X.shape[1])

      self.b = 0

      

    if display_loss:

      loss = {}

    

    for i in tqdm(range(epochs), total=epochs, unit="epoch"):

      dw = 0

      db = 0

      if loss_fn == "mse":

        dw = self.grad_w_mse(X, y)

        db = self.grad_b_mse(X, y) 

      elif loss_fn == "ce":

        dw = self.grad_w_ce(X, y)

        db = self.grad_b_ce(X, y)

      

      self.W -= learning_rate * dw

      self.b -= learning_rate * db

      

      if display_loss:

        Y_pred = self.sigmoid(self.perceptron(X))

        if loss_fn == "mse":

          loss[i] = mean_squared_error(y, Y_pred)

        elif loss_fn == "ce":

          loss[i] = log_loss(y, Y_pred)

    

    if display_loss:

      plt.plot(np.array(list(loss.values())).astype(float))

      plt.xlabel('Epochs')

      if loss_fn == "mse":

        plt.ylabel('Mean Squared Error')

      elif loss_fn == "ce":

        plt.ylabel('Log Loss')

      plt.show()

    

    return np.array(list(loss.values())).astype('float')

      

  def predict(self, X):

    Y_pred = []

    Y_pred.append(self.sigmoid(self.perceptron(X)))

    return np.array(Y_pred)
def read_all(folder_path, key_prefix=""):

    '''

    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.

    '''

    print("Reading:")

    images = {}

    files = os.listdir(folder_path)

    for i, file_name in tqdm(enumerate(files), total=len(files)):

        file_path = os.path.join(folder_path, file_name)

        image_index = key_prefix + file_name[:-4]

        image = Image.open(file_path)

        image = image.convert("L")

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/train-and-test-data-sets/level_1_train/"+LEVEL+"/background", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/train-and-test-data-sets/level_1_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/train-and-test-data-sets/level_1_test/kaggle_level_1/", key_prefix='') # change the path

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
def imshow(image):

    image = image.reshape(16, 16)

    plt.axis('off')

    plt.imshow(image)
imshow(X_train[799])
scaler = MinMaxScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
sn_ce = SigmoidNeuron_V()
# scaler = StandardScaler()

# X_scaled_train = scaler.fit_transform(X_train)

# X_scaled_test = scaler.transform(X_test)
imshow(X_scaled_train[799])
sn_ce = SigmoidNeuron_V()

loss = sn_ce.fit(X_scaled_train, Y_train, epochs=1500, learning_rate=0.02, loss_fn="ce", display_loss=True)
def print_accuracy(sn, binarized = False):

  if binarized:

    Y_pred_train = sn.predict(X_train_binarized)

  else:

    Y_pred_train = sn.predict(X_scaled_train)

  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)
print_accuracy(sn_ce)
class MPNeuron:

    

    def __init__(self):

        self.b = 0

    

    def model(self,x):

        return 1 if sum(x) >= self.b else 0

    

    def predict(self,X):

        y_pred = []

        for x in X:

            y_pred.append(self.model(x))

        return np.asarray(y_pred)

    

    def fit(self,X,Y):

        accuracy_dic = {}

        for i in range(0,X.shape[1]+1):

            self.b = i

            y_pred = self.predict(X)

            accuracy = accuracy_score(y_pred,Y)

            accuracy_dic[i] = accuracy

        b = max(accuracy_dic, key = accuracy_dic.get)

        print(accuracy_dic)

        self.b = b
##Binarizing for mp neuron

X_train_binarized = []

for i in X_train:

    i = list(i)

    x = [0 if y == 255 else 1 for y in i]

    X_train_binarized.append(np.array(x))



X_test_binarized = []

for i in X_test:

    i = list(i)

    x = [0 if y == 255 else 1 for y in i]

    X_test_binarized.append(np.array(x))



X_train_binarized = np.array(X_train_binarized)

X_test_binarized = np.array(X_test_binarized)
mp = MPNeuron()

mp.fit(X_train_binarized, Y_train)

y_pred = mp.predict(X_train_binarized)

print('Accuracy on training data',accuracy_score(y_pred,Y_train))

y_pred = mp.predict(X_test_binarized)

print(mp.b)
print_accuracy(mp, True)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submission.csv", index=False)

submission.head()
submission_copy[submission['Class'] != submission_copy['Class']]
Y_pred_test = mp.predict(X_test_binarized)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission_mp = {}

submission_mp['ImageId'] = ID_test

submission_mp['Class'] = Y_pred_binarised_test



submission_mp = pd.DataFrame(submission_mp)

submission_mp = submission_mp[['ImageId', 'Class']]

submission_mp = submission_mp.sort_values(['ImageId'])

submission_mp.to_csv("submission_mp.csv", index=False)

submission_mp.head()
from IPython.display import FileLink

FileLink(r'submission.csv')