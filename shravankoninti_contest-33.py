import os

import sys

import pickle

import numpy as np

import pandas as pd

from PIL import Image, ImageFilter, ImageEnhance

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from PIL.ImageFilter import (

    ModeFilter

    )

from PIL.ImageFilter import (

    RankFilter, MedianFilter, MinFilter, MaxFilter

    )

import pytesseract

import warnings

warnings.filterwarnings("ignore")



np.random.seed(100)

LEVEL = 'level_3'
import os

print(os.listdir("../input"))
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import mean_squared_error

from tqdm import tqdm_notebook

import operator

import json

np.random.seed(0)



class MPNeuron:

    

    def __init__(self):

        self.theta = None

        

    def mp_neuron(self, x):

        if sum(x) >= self.theta:

            return 1

        return 0

    

    def fit_brute_force(self, X, Y):

        accuracy = {}

        for theta in tqdm_notebook(range(0, X.shape[1]+1), total=X.shape[1]+1):

            self.theta = theta

            Y_pred = self.predict(X)

            accuracy[theta] = accuracy_score(Y, Y_pred)  

            

        sorted_accuracy = sorted(accuracy.items(), key=operator.itemgetter(1), reverse=True)

        best_theta, best_accuracy = sorted_accuracy[0]

        self.theta = best_theta

        

    def fit(self, X, Y, epochs=10, log=False, display_plot=False):

        self.theta = (X.shape[1]+1)//2

        if log or display_plot:

            accuracy = {}

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

            Y_pred = self.predict(X)

            tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()

            if fp > fn and self.theta <= X.shape[1]:

                self.theta += 1

            elif fp < fn and self.theta >= 1:

                self.theta -= 1

            else:

                continue

                

            if log or display_plot:

                Y_pred = self.predict(X)

                accuracy[i] = accuracy_score(Y, Y_pred)

        if log:

            with open('mp_neuron_accuracy.json', 'w') as fp:

                json.dump(accuracy, fp)

        if display_plot:

            epochs_, accuracy_ = zip(*accuracy.items())

            plt.plot(epochs_, accuracy_)

            plt.xlabel("Epochs")

            plt.ylabel("Train Accuracy")

            plt.show()

    

    def predict(self, X):

        Y = []

        for x in X:

            result = self.mp_neuron(x)

            Y.append(result)

        return np.array(Y)





class Perceptron:

    

    def __init__(self):

        self.w = None

        self.b = None

        

    def perceptron(self, x):

        return np.sum(self.w * x) + self.b

    

    def fit(self, X, Y, epochs=10, learning_rate=0.01, log=False, display_plot=False):

        # initialise the weights and bias

        self.w = np.random.randn(1, X.shape[1])

        self.b = 0

        if log or display_plot: 

            accuracy = {}

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

            for x, y in zip(X, Y):

                result = self.perceptron(x)

                if y == 1 and result < 0:

                    self.w += learning_rate*x

                    self.b += learning_rate

                elif y == 0 and result >= 0:

                    self.w -= learning_rate*x

                    self.b -= learning_rate

            if log or display_plot:

                Y_pred = self.predict(X)

                accuracy[i] = accuracy_score(Y, Y_pred)

        if log:

            with open('perceptron_accuracy.json', 'w') as fp:

                json.dump(accuracy, fp)

        if display_plot:

            epochs_, accuracy_ = zip(*accuracy.items())

            plt.plot(epochs_, accuracy_)

            plt.xlabel("Epochs")

            plt.ylabel("Train Accuracy")

            plt.show()

                    

    def predict(self, X):

        Y = []

        for x in X:

            result = self.perceptron(x)

            Y.append(int(result>=0))

        return np.array(Y)





class PerceptronWithSigmoid:

    

    def __init__(self):

        self.w = None

        self.b = None

        

    def perceptron(self, x):

        return np.sum(self.w * x) + self.b

    

    def sigmoid(self, z):

        return 1. / (1. + np.exp(-z))

    

    def grad_w(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * y_pred * (1 - y_pred) * x

    

    def grad_b(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * y_pred * (1 - y_pred)

    

    def fit(self, X, Y, epochs=10, learning_rate=0.01, log=False, display_plot=False):

        # initialise the weights and bias

        self.w = np.random.randn(1, X.shape[1])

        self.b = 0

        if log or display_plot: 

            #accuracy = {}

            mse = {}

        for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):

            dw, db = 0, 0

            for x, y in zip(X, Y):

                dw += self.grad_w(x, y)

                db += self.grad_b(x, y)

            self.w -= learning_rate*dw

            self.b -= learning_rate*db

            

            if log or display_plot:

                Y_pred = self.predict(X)

                #Y_binarized = (Y >= SCALED_THRESHOLD).astype(np.int)

                #Y_pred_binarized = (Y_pred >= SCALED_THRESHOLD).astype(np.int)

                #accuracy[i] = accuracy_score(Y_binarized, Y_pred_binarized)

                mse[i] = mean_squared_error(Y, Y_pred)

        if log:

            #with open('perceptron_with_sigmoid_accuracy.json', 'w') as fp:

                #json.dump(accuracy, fp)

            with open('perceptron_with_sigmoid_mse.json', 'w') as fp:

                json.dump(mse, fp)

        if display_plot:

            #epochs_, accuracy_ = zip(*accuracy.items())

            #plt.plot(epochs_, accuracy_)

            #plt.xlabel("Epochs")

            #plt.ylabel("Train Accuracy")

            #plt.show()

            epochs_, mse_ = zip(*mse.items())

            plt.plot(epochs_, mse_)

            plt.xlabel("Epochs")

            plt.ylabel("Train Error (MSE)")

            plt.show()

            

                    

    def predict(self, X):

        Y = []

        for x in X:

            result = self.sigmoid(self.perceptron(x))

            Y.append(result)

        return np.array(Y)



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

#     invert_clr = lambda x: 0 if x > 10 else 255

    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):

        file_path = os.path.join(folder_path, file_name)

        image_index = key_prefix + file_name[:-4]

        image = Image.open(file_path)

   

        Lim=image.convert("L")

        

        threshold = 12

        # if pixel value smaller than threshold, return 0 . Otherwise return 1.

        filter_func = lambda x: 0 if x < threshold else 1         

        image=Lim.point(filter_func, "1")  

        image = image.filter(ImageFilter.MedianFilter())

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images
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
scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
# sn_mse = PerceptronWithSigmoid()

# sn_mse.fit(X_scaled_train, Y_train, epochs=200, learning_rate=0.1, 

#                    log=False, display_plot=True)
# def print_accuracy(sn):

#   Y_pred_train = sn.predict(X_scaled_train)

#   Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

#   accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

#   print("Train Accuracy : ", accuracy_train)

#   print("-"*50)



# print_accuracy(sn_mse)
X = X_train

y = Y_train

X_train_tr, X_valid, y_train_tr, y_valid = train_test_split(X,y, random_state=17, test_size=0.33)
print(X_train_tr.shape, y_train_tr.shape)

print(X_valid.shape, y_valid.shape)
scaler = StandardScaler()

X_scaled_train_tr = scaler.fit_transform(X_train_tr)

X_scaled_valid_tr = scaler.transform(X_valid)

X_scaled_test = scaler.transform(X_test)



# sn_mse = PerceptronWithSigmoid()

# sn_mse.fit(X_scaled_train_tr, y_train_tr, epochs=500, learning_rate=0.05, 

#                    log=False, display_plot=True)



sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train_tr, y_train_tr, epochs=500,

          learning_rate=0.015, loss_fn="ce", display_loss=True)







def print_accuracy_train(sn):

  Y_pred_train_tr = sn.predict(X_scaled_train_tr)

  Y_pred_binarised_train_tr = (Y_pred_train_tr >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train_tr, y_train_tr)

  print("Train Accuracy : ", accuracy_train)

  print("-"*50)



def print_accuracy_valid(sn):

  Y_pred_valid_tr = sn.predict(X_scaled_valid_tr)

  Y_pred_binarised_valid_tr = (Y_pred_valid_tr >= 0.5).astype("int").ravel()

  accuracy_valid = accuracy_score(Y_pred_binarised_valid_tr, y_valid)

  print("Valid Accuracy : ", accuracy_valid)

  print("-"*50)
print_accuracy_train(sn_ce)

print_accuracy_valid(sn_ce)
# def runSigmoidNeuron(train_X, train_y, test_X, test_y=None, test_X2=None):

#     model = SigmoidNeuron()

#     model.fit(train_X, train_y, epochs=500, 

#               learning_rate= 0.1, loss_fn="ce", display_loss=False)    

    

#     train_preds = model.predict(train_X)    

#     train_preds = (train_preds >= 0.5).astype("int").ravel()

    

#     test_preds = model.predict(test_X)

#     test_preds = (test_preds >= 0.5).astype("int").ravel()

    

#     test_preds2 = model.predict(test_X2)

#     test_preds2 = (test_preds2 >= 0.5).astype("int").ravel()

    

#     train_accuracy = accuracy_score(train_y, train_preds)

#     test_accuracy = accuracy_score(test_y, test_preds)

#     print("Train and Test Accuracy : ", train_accuracy, test_accuracy)

#     return test_preds, test_accuracy, test_preds2
# # Necessary imports: 

# from sklearn.model_selection import cross_val_score, cross_val_predict

# from sklearn import metrics

# from sklearn.model_selection import RepeatedKFold, KFold    

# cv_scores = []

# pred_test_full = []

# kf = KFold(n_splits=5, random_state=None, shuffle=False)

# for dev_index, val_index in kf.split(X_scaled_train,Y_train):



#     dev_X, val_X = X_scaled_train[dev_index,:], X_scaled_train[val_index,:]

#     dev_y, val_y = Y_train[dev_index], Y_train[val_index]

    

#     pred_val, Accu, pred_test = runSigmoidNeuron(dev_X, dev_y, val_X, val_y, X_scaled_test)



#     cv_scores.append(Accu)

#     pred_test_full.append(pred_test)

#     print(cv_scores)

# # pred_test_full /= 5.
# data = pd.DataFrame(pred_test_full)



# data1 = (data.T)

# data1.columns = ['a', 'b', 'c', 'd', 'e']

# data1['e'] = (data1['a'] + data1['b']+data1['c']+data1['d']+data1['e'])

# data1['f'] = data1['e'].apply(lambda x: 1 if x >= 2 else 0)

# data1['f'].value_counts()
# submission = {}

# submission['ImageId'] = ID_test

# submission['Class'] = data1['f']



# submission = pd.DataFrame(submission)

# submission = submission[['ImageId', 'Class']]

# submission = submission.sort_values(['ImageId'])

# submission.to_csv("submisision.csv", index=False)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)