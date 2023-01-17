# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')





np.random.seed(100)

LEVEL = 'level_1'



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
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



images_train = read_all("../input/level_1_train/level_1"+"/"+"background", key_prefix='bgr_') # change the path

for language in languages:

    images_train.update(read_all("../input/level_1_train/level_1"+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_1_test/kaggle_level_1", key_prefix='') # change the path

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
scaler = MinMaxScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
class SigmoidNeuron:

    

    '''Sigmoid neuron class to classify texts'''

    

    # initialize weights, bias and number of epochs

    def __init__(self, epochs):

        self.epochs = epochs

        

    

    # linear combination of features

    def perceptron(self, x):

        return np.dot(self.w, x) + self.b

    

    

    # sigmoid non-linearity

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-self.perceptron(x)))

    

    

    # gradient of weights with squared error loss function

    def grad_w_mse(self, x, y):

        y_pred = self.sigmoid(x)

        return (y_pred - y) * x * y_pred * (1 - y_pred)

    

    

    # gradient of weights with cross-entropy loss function

    def grad_w_ce(self, x, y):

        y_pred = self.sigmoid(x)

        return (y_pred - y) * x

    

    

    # gradient of bias with squared error loss function

    def grad_b_mse(self, x, y):

        y_pred = self.sigmoid(x)

        return (y_pred - y) * y_pred * (1 - y_pred)

    

    

    # gradient of bias with cross-entropy loss function

    def grad_b_ce(self, x, y):

        y_pred = self.sigmoid(x)

        return (y_pred - y)

    

    

    # fit on data

    def fit(self, X, Y, lr=1, display_loss=False, loss='mse'):

        # initialize weights and bias

        self.w = [np.random.random() for _ in range(X_train.shape[1])] 

        self.b = np.random.random()

        # capture loss

        L = []

        for _ in tqdm_notebook(range(self.epochs), total=self.epochs, unit="epoch"):

            preds = []

            dw, db = 0, 0

            for x,y in zip(X, Y):

                if loss == 'mse':

                    dw += self.grad_w_mse(x, y)

                    db += self.grad_b_mse(x, y)

                else:

                    dw += self.grad_w_ce(x, y)

                    db += self.grad_b_ce(x, y)

            self.w -= lr * dw

            self.b -= lr * db

            for row in X:

                pred = self.sigmoid(row)

                preds.append(pred)

            loss = mean_squared_error(np.array(preds).reshape(-1,1), np.array(Y).reshape(-1,1))

            L.append(loss)

        if display_loss:

            plt.plot(L)

            plt.xlabel('Epochs')

            plt.ylabel('MSE Loss')

            plt.show()

        return self.w, self.b

    

    # training set accuracy

    def train_accuracy(self, X, Y, lr=1, loss='mse'):

        pred = self.predict(X, X, Y)

#         return pred.shape

        y_pred = []

        for i in pred:

            if i >= 0.5:

                i = 1

                y_pred.append(i)

            else:

                i = 0

                y_pred.append(i)

        y_pred = np.array(y_pred).reshape(-1,1)

        return accuracy_score(y_pred, Y.reshape(-1,1))

        

    

    # predict on unknown data

    def predict(self, X_train, X_test, Y_train, lr=1, display_loss=False, loss='mse'):

        self.w, self.b = self.fit(X_train, Y_train)

        Y_pred = []

        for row in X_test:

            pred = self.sigmoid(row)

            Y_pred.append(pred)

        return np.array(Y_pred)
sn_mse = SigmoidNeuron(1000)

sn_mse.fit(X_scaled_train, Y_train, lr=0.001, display_loss=True)
sn_mse = SigmoidNeuron(1000)

mse_pred = sn_mse.predict(X_scaled_train, X_scaled_test, Y_train, lr=0.001, display_loss=True)

mse_pred
mse_pred_binarized = (mse_pred >= 0.5).astype("int").ravel()

mse_pred_binarized
submission_mse = {}

submission_mse['ImageId'] = ID_test

submission_mse['Class'] = mse_pred_binarized



submission_mse = pd.DataFrame(submission_mse)

submission_mse = submission_mse[['ImageId', 'Class']]

submission_mse = submission_mse.sort_values(['ImageId'])

submission_mse.to_csv("submisision.csv", index=False)
sn_ce = SigmoidNeuron(2500)

sn_ce.fit(X_scaled_train, Y_train, lr=0.0001, display_loss=True)
sn_ce = SigmoidNeuron(2500)

ce_pred = sn_ce.predict(X_scaled_train, X_scaled_test, Y_train, lr=0.0001, display_loss=True)

ce_pred
ce_pred_binarized = (ce_pred >= 0.5).astype("int").ravel()
ce_pred_binarized
submission_ce = {}

submission_ce['ImageId'] = ID_test

submission_ce['Class'] = ce_pred_binarized



submission_ce = pd.DataFrame(submission_ce)

submission_ce = submission_ce[['ImageId', 'Class']]

submission_ce = submission_ce.sort_values(['ImageId'])

submission_ce.to_csv("submisision.csv", index=False)