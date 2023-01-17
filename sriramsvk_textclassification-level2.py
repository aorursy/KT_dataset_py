import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt



np.random.seed(100)

LEVEL = 'level_2'
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

ImagePath = "../input/padhai-text-non-text-classification-level-2/"+LEVEL+"/"

#testPath = "../input/padhai-text-non-text-classification-level-2"+LEVEL+"_test/kaggle_"+LEVEL

allimages = read_all(ImagePath+"background", key_prefix='bgr_') # change the path

for language in languages:

  allimages.update(read_all(ImagePath+language, key_prefix=language+"_" ))

print(len(allimages))



#images_test = read_all(testPath, key_prefix='') # change the path

#print(len(images_test))
testPath = "../input/padhai-text-non-text-classification-level-2/kaggle_level_2"

images_test = read_all(testPath, key_prefix='') # change the path

print(len(images_test))
print(list(allimages.items())[650],"\n Keys",list(allimages.keys())[650],"\n value",list(allimages.values())[650])
X = []

Y = []

ID = []

i = 0 

for key,value in allimages.items():

    ID.append(i)

    i+=1

    X.append(value)

    if key[:4] == "bgr_":

        Y.append(0)

    else: 

        Y.append(1)



print(X[0],Y[0],ID[0])
from sklearn.model_selection import train_test_split
import random

idx = ID

random.shuffle(ID)

print(ID[2],idx[2])



trainset,testset = train_test_split(ID,train_size=.8,test_size=.2,stratify=Y,random_state=1)

print(len(trainset),len(testset),X[trainset[0]],Y[trainset[0]],X[testset[0]],Y[testset[0]])
X_train = []

X_test = [] 

Y_train = []

Y_test = []

print(len(trainset))



# inserting data to Train set 

for i in range(len(trainset)):

    X_train.append(X[trainset[i]])

    Y_train.append(Y[trainset[i]])

    

# inserting data to Test Set 

for i in range(len(testset)):

    X_test.append(X[testset[i]])

    Y_test.append(Y[testset[i]])

print("precentage of 1 in Y",round(Y.count(1)/len(Y),2))

print("precentage of 1 in Y Train",round(Y_train.count(1)/len(Y_train),2))

print("precentage of 1 in Y Test",round(Y_test.count(1)/len(Y_test),2))



X_train = np.array(X_train)

Y_train = np.array(Y_train)

X_test = np.array(X_test)

Y_test = np.array(Y_test)



print(X_train.shape, Y_train.shape,X_test.shape,Y_test.shape)
scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
sn_mse = SigmoidNeuron()

sn_mse.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.015, loss_fn="mse", display_loss=True)
sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.015, loss_fn="ce", display_loss=True)
def print_accuracy(sn):

  Y_pred_train = sn.predict(X_scaled_train)

  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)
print_accuracy(sn_mse)

print_accuracy(sn_ce)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()

accuracy_test = accuracy_score(Y_pred_binarised_test, Y_test)

print("Test Accuracy : ", accuracy_test)

print("-"*50)
ID_validation = []

X_validation = []

for key, value in images_test.items():

  ID_validation.append(int(key))

  X_validation.append(value)



X_validation = np.array(X_validation)

print(X_validation.shape)
X_scaled_validation = scaler.transform(X_validation)
Y_pred_validation = sn_ce.predict(X_scaled_validation)

Y_pred_binarised_validation = (Y_pred_validation >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_validation

submission['Class'] = Y_pred_binarised_validation



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)