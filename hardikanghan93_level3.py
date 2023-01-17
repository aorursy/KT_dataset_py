import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix ,roc_auc_score,roc_curve

from sklearn.model_selection import StratifiedKFold,train_test_split

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import cv2



np.random.seed(100)

LEVEL = 'level_3'
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

            self.w = np.zeros(X.shape[1])

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

            key_min = min(loss.keys(), key=(lambda k: loss[k]))

            print('epochs : ' , key_min)

            print('Minimum Loss : ',loss[key_min])

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

        image = cv2.imread(file_path,1) 

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert color image to gray

        ret,th = cv2.threshold(gray,13,255,cv2.THRESH_BINARY) # convert pixels having value more then 13 to 255(White) 

        median = cv2.medianBlur(th, 3) # Remove salt and pepper noise

        images[image_index] = np.array(median.copy()).flatten()

    return images
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/level_3_train/"+LEVEL+"/"+"background/", key_prefix='bgr_') # change the path

for language in languages:

    images_train.update(read_all("../input/level_3_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_3_test/kaggle_"+LEVEL, key_prefix='') # change the path

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
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X, y):

    X_kf_train, X_kf_test = X[train_index], X[test_index]

    y_kf_train, y_kf_test = y[train_index], y[test_index]

    X_kf_scaled_train = scaler.fit_transform(X_kf_train)

    X_kf_scaled_test = scaler.transform(X_kf_test)

    sn_ce = SigmoidNeuron()

    sn_ce.fit(X_kf_scaled_train, y_kf_train, epochs=2000, learning_rate=0.00001, loss_fn="ce", display_loss=True)

    Y_pred_train = sn_ce.predict(X_kf_scaled_train)

    dic = {}

    thresholds  =  np.linspace(0,1,10000)

    # For each threshold calculate the accuracy score and store in dic

    for i in range(len(thresholds)):

        x = (Y_pred_train >= thresholds[i]).astype("int").ravel() 

        dic[i] = accuracy_score(y_kf_train,x)

    key_max = max(dic.keys(), key=(lambda k: dic[k]))

    print('Max Accuracy Score for train : ',dic[key_max])

    print('Threshold probability : ', thresholds[key_max])

    Y_pred_train = (Y_pred_train >= thresholds[key_max]).astype("int").ravel()

    tn, fp, fn, tp = confusion_matrix(y_kf_train,Y_pred_train).ravel()

    print("Confusion matrix for train set")

    print("tn, fp, fn, tp : ",tn, fp, fn, tp)

    Y_pred_test = sn_ce.predict(X_kf_scaled_test)

    Y_pred_binarised_test = (Y_pred_test >= thresholds[key_max]).astype("int").ravel()

    print('Accuracy Score for test : ',accuracy_score(y_kf_test,Y_pred_binarised_test))
sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train, y, epochs=2000, learning_rate= 0.00001, loss_fn="ce", display_loss=True)
Y_pred_train = sn_ce.predict(X_scaled_train)

thresholds  =  np.linspace(0,1,10000)

dic = {}

for i in range(len(thresholds)):

    x = (Y_pred_train >= thresholds[i]).astype("int").ravel()

    dic[i] = accuracy_score(y,x)

key_max = max(dic.keys(), key=(lambda k: dic[k]))

print('Max Accuracy Score for train : ',dic[key_max])

print('Threshold probability : ', thresholds[key_max])

Y_pred_train = (Y_pred_train >= thresholds[key_max]).astype("int").ravel()

tn, fp, fn, tp = confusion_matrix(y,Y_pred_train).ravel()

print("Confusion matrix for train set")

print("tn, fp, fn, tp :", tn, fp, fn, tp)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= thresholds[key_max]).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)