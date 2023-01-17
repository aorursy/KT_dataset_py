import os

import sys

import pickle

import numpy as np

import pandas as pd

import cv2

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt



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

  

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False,X_test=None,Y_test=None,display_test_loss=False):

        

        # initialise w, b

        if initialise:

            self.w = np.random.randn(1, X.shape[1])

            self.b = 0



        

        loss = {}

        loss_test = {}



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



            # display the training error

            Y_pred = self.sigmoid(self.perceptron(X))

            Y_pred_test = self.sigmoid(self.perceptron(X_test))

            if loss_fn == "mse":

                loss[i] = mean_squared_error(Y, Y_pred)

                loss_test[i] = mean_squared_error(Y_test,Y_pred_test)

            elif loss_fn == "ce":

                loss[i] = log_loss(Y, Y_pred)

                loss_test[i] = log_loss(Y_test,Y_pred_test)

            # display the test error



        if display_loss:

            plt.plot(loss.values(),label='Train Set')

            if display_test_loss:

                plt.plot(loss_test.values(),label='Test Set')

            plt.xlabel('Epochs')

            plt.legend()

            if loss_fn == "mse":

                plt.ylabel('Mean Squared Error')

            elif loss_fn == "ce":

                plt.ylabel('Log Loss')

                plt.show()

        

        return loss,loss_test

        

        



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

        img = cv2.imread(file_path)

        

        # convert it into grayscale image

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_image = 255-gray_image

        

        # Thresholding

        ret,threshold_image = cv2.threshold(gray_image,220,255,cv2.THRESH_BINARY)

        

        # eroding the image

        kernel = np.ones((2,2), np.uint8)

        erosion = cv2.erode(threshold_image,kernel,iterations = 1)

        

        # flattening the image

        images[image_index] = erosion.copy().flatten()

    return images
os.listdir("../input/")
languages = ['ta', 'hi', 'en']

images_train = read_all("../input/level_3_train/"+LEVEL+"/"+"background", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/level_3_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_3_test/kaggle_"+LEVEL, key_prefix='') # change the path

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
X_train_split, X_test_split, Y_train_split, Y_test_split = train_test_split(X_train,Y_train, random_state=0, stratify=Y_train, test_size=0.1)

X_train_split.shape, X_test_split.shape, Y_train_split.shape, Y_test_split.shape
scaler = StandardScaler()

# Standardizing the train_split and test_split set for hyperparmeter tuning

X_scaled_train_split = scaler.fit_transform(X_train_split)

X_scaled_test_split = scaler.transform(X_test_split)



# Standardizing the test set for submitting the prediction

X_scaled_test = scaler.transform(X_test)
import warnings

warnings.filterwarnings('ignore')  # Hide all warnings in ipython notebook

sn = SigmoidNeuron()

loss_train, loss_test = sn.fit(X_scaled_train_split, Y_train_split, epochs=200, learning_rate=1e-7, loss_fn="ce",display_loss=True,X_test = X_scaled_test_split,Y_test=Y_test_split,display_test_loss=True)

print(list(loss_train.values())[-1],list(loss_test.values())[-1])
# Diplay the last 10 iterations

last_range=10

plt.plot(range(epochs-last_range,epochs),list(loss_test.values())[-last_range:])

plt.show()
def print_accuracy(sn):

  Y_pred_train = sn.predict(X_scaled_train_split)

  Y_pred_binarised_train = (Y_train_split >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train_split)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)



print_accuracy(sn)
Y_pred_test = sn.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)