import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter, ImageOps

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt

import warnings

from sklearn.model_selection import train_test_split



np.random.seed(100)

LEVEL = 'level_1'
class SigmoidNeuron:



    def __init__(self):

        self.w = None

        self.b = None

    

    def sigmoid(self, x):

        return 1.0/(1.0 + np.exp(-(np.dot(x, self.w) + self.b)))

    

    def grad_mse(self, y, y_pred):

        return (y_pred - y) * y_pred * (1 - y_pred)

    

    def grad_ce(self, y, y_pred):

        return y_pred-y

        

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False):

    

        # initialise w, b

        if initialise:

            #self.w = np.random.randn(X.shape[1])

            np.random.seed(100)

            self.w = np.random.ranf(X.shape[1]) - 0.5

            #print(self.w)

            self.b = 0

        

        if display_loss:

            loss = {}

    

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

            dw = 0

            db = 0

            if loss_fn == 'mse':

                for x, y in zip(X, Y):

                    temp = self.grad_mse(y, self.sigmoid(x))

                    dw += (temp*x)

                    db += temp 

            elif loss_fn == "ce":

                for x, y in zip(X, Y):

                    temp = self.grad_ce(y, self.sigmoid(x))

                    dw += (temp*x)

                    db += temp

                    

            self.w -= learning_rate * dw

            self.b -= learning_rate * db

            

            if display_loss:

                Y_pred = self.sigmoid(X)

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

        return np.array(self.sigmoid(X))#np.array(Y_pred)
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

        #image = image.convert("L")    #Original Code

        image = image.convert("L") 

        image = ImageOps.invert(image)

        image = ImageOps.equalize(image)

        image = image.filter(ImageFilter.MedianFilter(5))

        image = image.filter(ImageFilter.EDGE_ENHANCE)

        #image = image.convert("P", palette=Image.ADAPTIVE, colors=64)

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/level_4a_train/level_4a/"+"background", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/level_4a_train/level_4a/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_4a_test/kaggle_level_4a/", key_prefix='') # change the path

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
warnings.filterwarnings('ignore')

scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)





x_tr,x_te, y_tr, y_te = train_test_split(X_train,Y_train, test_size = 0.3, stratify = Y_train, random_state = 7)

scaler = StandardScaler()

x_s_tr = scaler.fit_transform(x_tr)

x_s_te = scaler.transform(x_te)

x_s_test = scaler.transform(X_test)
sn_ce = SigmoidNeuron()

sn_ce.fit(x_s_tr, y_tr, epochs=500, learning_rate=0.0000145, loss_fn="ce", display_loss=True,  initialise = True)

print('w_after',sn_ce.w)

print('b_after',sn_ce.b)
#sn_ce = SigmoidNeuron()

#sn_ce.fit(X_scaled_train, Y_train, epochs=2000, learning_rate=0.000013, loss_fn="ce", display_loss=True,  initialise = True)
def print_accuracy(sn, X_scaled_train, Y_train):

    Y_pred_train = sn.predict(X_scaled_train)

    Y_pred_binarised_train = (Y_pred_train >= 0.25).astype("int").ravel()

    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

    print("Train Accuracy : ", accuracy_train)

    ones = [x for x,y in zip(Y_pred_train, Y_train) if y == 1]

    zeros = [x for x,y in zip(Y_pred_train, Y_train) if y != 1]

    print("1 avg & std: ",np.mean(ones),np.std(ones))

    print("0 avg & std: ",np.mean(zeros),np.std(zeros))

    

    print("y = 1, y` = 1: ",sum([1 for x,y in zip(Y_pred_binarised_train, Y_train) if y == 1  and x == 1]))

    print("y = 0, y` = 0: ",sum([1 for x,y in zip(Y_pred_binarised_train, Y_train) if y == 0  and x == 0]))

    print("y = 1, y` = 0: ",sum([1 for x,y in zip(Y_pred_binarised_train, Y_train) if y == 1  and x == 0]))

    print("y = 0, y` = 1: ",sum([1 for x,y in zip(Y_pred_binarised_train, Y_train) if y == 0  and x == 1]))

    print("-"*50)
#print_accuracy(sn_mse)

print_accuracy(sn_ce, x_s_tr, y_tr)

print_accuracy(sn_ce, x_s_te, y_te)
Y_pred_test = sn_ce.predict(x_s_test)

#Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.25).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)