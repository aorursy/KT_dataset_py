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

import PIL

import cv2

np.random.seed(100)

LEVEL = 'level_4a'

import warnings

warnings.filterwarnings("ignore")
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

    if initialise or self.w is None:

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

#         image = Image.open(file_path)

#         image = image.convert("L")

#         image = image.filter(PIL.ImageFilter.GaussianBlur(radius=1))

#         images[image_index] = np.array(image.copy()).flatten()

        img = cv2.imread(file_path,0)

#         th3  = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#         blur = cv2.blur(img,(2,2))

#         ret1,th2 = cv2.threshold(blur,30,255,cv2.THRESH_BINARY)

        blur = cv2.blur(img,(2,2))

        ret1,th1 = cv2.threshold(img,30,255,cv2.THRESH_BINARY)

        invert =   cv2.bitwise_not(th1)

        kernel = np.ones((2,2),np.uint8)

        erosion = cv2.erode(invert,kernel,iterations = 1)

        dilation = cv2.dilate(erosion,kernel,iterations = 1)

        images[image_index] = np.array(dilation.flatten())

#         image.close()

    return images
# image = Image.open("../input/level_3_train/"+LEVEL+"/en/displayPanel_0_flip_none_c27_25.jpg")
# image.convert("L")
# from skimage.filters import threshold_adaptive
# img.shape
# from skimage import io

# img = io.imread("../input/level_3_train/"+LEVEL+"/en/displayPanel_0_flip_none_c27_25.jpg",flatten=True)

# threshold_adaptive(img.data, 35, offset=10)
# import cv2



# img = cv2.imread("../input/level_4a_train/"+LEVEL+"/en/idx1026_tent_11_c15_1026.jpg",cv2.IMREAD_GRAYSCALE)

# # th3  = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)





# # ret1,th2 = cv2.threshold(img,30,255,cv2.THRESH_BINARY)


# blur = cv2.blur(img,(2,2))

# ret1,th1 = cv2.threshold(img,30,255,cv2.THRESH_BINARY)

# invert =   cv2.bitwise_not(th1)

# kernel = np.ones((2,2),np.uint8)

# erosion = cv2.erode(invert,kernel,iterations = 1)

# dilation = cv2.dilate(erosion,kernel,iterations = 1)

# plt.imshow(th2,'gray')





# plt.show()
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/level_4a_train/"+LEVEL+"/"+"background", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/level_4a_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_4a_test/kaggle_"+LEVEL, key_prefix='') # change the path

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
scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
# scikit-learn k-fold cross-validation

from numpy import array

from sklearn.model_selection import KFold

kfold = KFold(5, True, 10)

# enumerate splits

sn_ce = SigmoidNeuron()

for train, test in kfold.split(X_scaled_train,Y_train):

#     print('train: %s, test: %s' % (Y_train[train], Y_train[test]))

    sn_ce.fit(X_scaled_train[train], Y_train[train], epochs=500, learning_rate=0.00002, loss_fn="ce", display_loss=True,initialise=False)

    

    Y_pred_train = sn_ce.predict(X_scaled_train[test])

    Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train[test])

    print("Train Accuracy : ", accuracy_train)

    print("-"*50)

    
# sn_mse = SigmoidNeuron()

# sn_mse.fit(X_scaled_train, Y_train, epochs=200, learning_rate=0.02, loss_fn="mse", display_loss=True)
# sn_ce = SigmoidNeuron()

# sn_ce.fit(X_scaled_train, Y_train, epochs=200, learning_rate=0.02, loss_fn="ce", display_loss=True)
def print_accuracy(sn):

  Y_pred_train = sn.predict(X_scaled_train)

  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)
# print_accuracy(sn_mse)

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