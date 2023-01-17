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

#sys.path.insert(0,'../input/')

#import padhai

import cv2

import warnings 

warnings.filterwarnings("ignore")

np.random.seed(100)

LEVEL = 'level_3'
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

      self.w = np.zeros(X.shape[1], dtype = float)

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
#this is the same readall function given in the boiler code with some adjustments for the openCV image processing stuff

#i have tried to explain the oCV part in the snippet which follows below





def read_all(folder_path, key_prefix = ""):

    print("Reading:")

    images = {}

    files = os.listdir(folder_path)

    for i, file_name in tqdm_notebook(enumerate(files), total = len(files)):

        file_path = os.path.join(folder_path, file_name)

        image_index = key_prefix + file_name[:-4]

        im = cv2.imread(file_path)

        b,g,r = cv2.split(im)

        im1 = cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

        imh = cv2.cvtColor(im1,cv2.COLOR_RGB2HSV_FULL)

        imh1 = cv2.cvtColor(imh,cv2.COLOR_BGR2HSV_FULL)

        imh2 = cv2.cvtColor(imh1,cv2.COLOR_BGR2HSV_FULL)

        imh3 = cv2.cvtColor(imh2,cv2.COLOR_BGR2HSV_FULL)

        imh4 = cv2.cvtColor(imh3,cv2.COLOR_BGR2HSV_FULL)

        h,s,v = cv2.split(imh4)

        image = Image.fromarray(v)

        image = image.convert("L")

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images

        
'''def read_all(folder_path, key_prefix=""):

    

    print("Reading:")

    images = {}

    files = os.listdir(folder_path)

    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):

        file_path = os.path.join(folder_path, file_name)

        image_index = key_prefix + file_name[:-4]

        image = Image.open(file_path)

        image = image.convert("1")

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images'''
#this is a test to find how the images look when opened using pillow module

pil = Image.open("../input/level_3_train/level_3/hi/displayPanel_0_flip_none_c1_96.jpg")

plt.imshow(pil)

plt.show()
# this was my so called extensive test with openCV functions to improve accuracy

# processing a image opened using pillow functions in oCV(openCV) was quite complicated as they opened the file in different forms, so I opened the image in oCV

# I found oCV opens files in the BGR format rather the widely preferred RGB format

# First i converted the pic into rgb colorspace.

# now i found using the HSV colorspace could be quite useful as the background was getting eliminated step by step 

# every output from the HSV conversion was taken to be BGR and then was again converted to HSV

# in every even iteration the background was missing I found only a neutral uniform background which will be white when 

# converted to grayscale

# so i keep converting the images to HSV colorspace till the background vanishes and just the text shows up

# now i will need only the gray channel imn the HSV colorspace. there are actually three channels

# i split the channels and take just the V channel which will be a numpy array

# read the v channel as a numpy array using 'fromarray' function from the pillow module

# now i convert the image to the pure greyscale using "L"





im = cv2.imread("../input/level_3_train/level_3/background/displayPanel_0_flip_blur_d3.jpg",0)

im1 = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

imh = cv2.cvtColor(im1,cv2.COLOR_RGB2HSV_FULL)

imh1 = cv2.cvtColor(imh,cv2.COLOR_BGR2HSV_FULL)

imh2 = cv2.cvtColor(imh1,cv2.COLOR_BGR2HSV_FULL)

imh3 = cv2.cvtColor(imh2,cv2.COLOR_BGR2HSV_FULL)

imh4 = cv2.cvtColor(imh3,cv2.COLOR_BGR2HSV_FULL)

plt.imshow(imh1)

plt.show()

h,s,v = cv2.split(imh4)

img = Image.fromarray(v)

imgc = img.convert("L")

print(type(imgc))

plt.imshow(imgc)

plt.show()



#you can change the plot statements accordingly to view the results

#I have used this exact same code in the readall function i edited and it actually worked
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/level_3_train/level_3/"+"background", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/level_3_train/level_3/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_3_test/kaggle_level_3", key_prefix='') # change the path

print(len(images_test))
#list(images_test.keys())[:5]
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
scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
#sn_mse = SigmoidNeuron()

#sn_mse.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.001, loss_fn="mse", display_loss=True)
sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train, Y_train, epochs=150, learning_rate=0.0015, loss_fn="ce", display_loss=True)

def print_accuracy(sn):

  Y_pred_train = sn.predict(X_scaled_train)

  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)

  
#print_accuracy(sn_mse)

print_accuracy(sn_ce)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.85).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)