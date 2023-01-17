# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix ,roc_auc_score,roc_curve

from sklearn.model_selection import StratifiedKFold

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

        return 1.0 / (1.0 + np.exp(-x))



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



images_train = read_all("../input/level_2_train/"+LEVEL+"/"+"background/", key_prefix='bgr_') # change the path

for language in languages:

    images_train.update(read_all("../input/level_2_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_2_test/kaggle_"+LEVEL, key_prefix='') # change the path

print(len(images_test))
list(images_test.keys())[:5]
X = []

y = []

for key, value in images_train.items():

    X.append(value)

    if key[:4] == "bgr_":

        y.append(0)

    else:

        y.append(1)



ID_test = []

X_test = []

for key, value in images_test.items():

    ID_test.append(int(key))

    X_test.append(value)



X = np.array(X)

y = np.array(y)

X_test = np.array(X_test)



print(X.shape, y.shape)

print(X_test.shape)
scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X)

X_scaled_test = scaler.transform(X_test)
sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train, y, epochs=1000, learning_rate=0.015, loss_fn="ce", display_loss=True)
Y_pred_train = sn_ce.predict(X_scaled_train)

accuracy_train = roc_auc_score(y,Y_pred_train)

print("Train Accuracy : ", accuracy_train)

fpr, tpr, thresholds = roc_curve(y, Y_pred_train)

print(fpr)

print(tpr)

print(thresholds)

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

# show the plot

plt.show()

Y_pred_train = (Y_pred_train >= 0.99).astype("int").ravel()

tn, fp, fn, tp = confusion_matrix(y, Y_pred_train).ravel()

print(tn, fp, fn, tp)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.99).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)