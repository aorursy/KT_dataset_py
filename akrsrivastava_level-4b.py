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

from sklearn.model_selection import train_test_split

#sys.path.insert(0, '../input/')

np.random.seed(1001)

LEVEL = 'level_1'

np.seterr(over='ignore')
class SigmoidNeuron:

  

  def __init__(self):

    self.w = None

    self.b = None

    self.best_epoch = 0

    

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

    #print(y_pred)

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

      

    #if display_loss:

    loss = {}

    

    best_w = self.w.copy()

    best_b = 0

    best_loss = 999

    best_epoch = 1

    

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

      

      #if display_loss:

      Y_pred = self.sigmoid(self.perceptron(X))

      if loss_fn == "mse":

          loss[i] = mean_squared_error(Y, Y_pred)

      elif loss_fn == "ce":

          

          loss[i] = log_loss(Y, Y_pred)

          #print(Y, Y_pred,loss[i])



      if best_loss == 999:

        best_loss = loss[i]

        best_w = self.w.copy()

        best_b = self.b

        best_epoch = i 

      elif loss[i] < best_loss:

        best_loss = loss[i]

        best_w = self.w.copy()

        best_b = self.b

        best_epoch = i

        

    self.w = best_w.copy()

    self.b = best_b

    self.best_epoch = best_epoch

    

    if display_loss:

      plt.plot(loss.values())

      plt.xlabel('Epochs')

      if loss_fn == "mse":

        plt.ylabel('Mean Squared Error')

      elif loss_fn == "ce":

        plt.ylabel('Log Loss')

      plt.show()

    #print("Best Epoch: ", best_epoch)

      

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

        img_array = np.array(image.copy()).flatten()

        img_array =  [(lambda x: 1 if x < 50 else 255)(x) for x in list(img_array)]

        #images[image_index] = np.array(image.copy()).flatten()

        images[image_index] = np.array(img_array)

        image.close()

    return images
a = np.array([1,2,3])

c = list(a)

print(a)

print(c)

b = [(lambda x: 0 if x < 2 else 255)(x) for x in c]

print(b)
languages = ['ta', 'hi', 'en']



images_train = read_all("../input/level_4b_train/level_4b/" + "background/", key_prefix='bgr_') # change the path

for language in languages:

  images_train.update(read_all("../input/level_4b_train/level_4b/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/level_4b_test/kaggle_level_4b/", key_prefix='') # change the path

print(len(images_test))
images_train.shape
list(images_test.keys())[:5]
a = np.array(([1,2],[3,4]), dtype = int)

print(a)

b = a.flatten()

print(b)

c = b.reshape(2,-1)

print(c)

i = images_test.get('567')

#print(i.reshape(64,-1))

z = i.reshape(64,-1,)

print(z)
plt.imshow(z)
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

  

        

X_train = np.array(X_train,dtype = float)

Y_train = np.array(Y_train)

X_test = np.array(X_test,dtype = float)



print(X_train.shape, Y_train.shape)

print(X_test.shape)
X_train[0]
scaler = StandardScaler()

X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)



# X_scaled_train = X_train

# X_scaled_test = X_test
sn_mse = SigmoidNeuron()

sn_mse.fit(X_scaled_train, Y_train, epochs=100, learning_rate=0.015, loss_fn="mse", display_loss=True)
print(Y_train[:10])
sn_ce = SigmoidNeuron()

sn_ce.fit(X_scaled_train, Y_train, epochs=10000, learning_rate=0.000005, loss_fn="ce", display_loss=True)
def print_accuracy(sn):

  Y_pred_train = sn.predict(X_scaled_train)

  #for k in range(1,8):

        

  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()

  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

  print( "Train Accuracy : ", accuracy_train)

  print("-"*50)
#print_accuracy(sn_mse)

print_accuracy(sn_ce)
# Y_pred_test = sn_ce.predict(X_scaled_test)

# Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



# submission = {}

# submission['ImageId'] = ID_test

# submission['Class'] = Y_pred_binarised_test



# submission = pd.DataFrame(submission)

# submission = submission[['ImageId', 'Class']]

# submission = submission.sort_values(['ImageId'])

# submission.to_csv("submisision.csv", index=False)
# # epochs = [50,200]

# # learning_rates = [0.001,0.02]



# epochs = [3000]

# learning_rates = [0.000005]



# best_epoch = 0

# best_lr = 0

# best_acc = 0

# for epoch in epochs:

#     for lr in learning_rates:

#         CV_Accuracy = 0

#         for i in range(5):

#             cv_train_x, cv_test_x, cv_train_y, cv_test_y = train_test_split(X_scaled_train,Y_train,test_size = 0.2, stratify = Y_train)

#             sn_ce = SigmoidNeuron()

#             sn_ce.fit(cv_train_x, cv_train_y, epochs=epoch, learning_rate=lr, loss_fn="ce", display_loss=False)

#             #optimum_epoch = sn_ce.best_epoch

#             pred_cv = sn_ce.predict(cv_test_x)

            

#             pred_cv_binarised = (pred_cv >= 0.5).astype("int").ravel()

#             CV_Accuracy += accuracy_score(pred_cv_binarised, cv_test_y)

#         CV_Accuracy /= 5

            

#         if CV_Accuracy > best_acc:

#             #best_epoch = optimum_epoch

#             best_lr = lr

#             best_acc = CV_Accuracy

#             best_model = sn_ce

            

        

#         #print("Epochs:" , epoch, " LR: ",lr, " CV Accuracy : ", accuracy_train)

# print("Best Epochs:" , best_epoch, "Best LR: ",best_lr, " Best CV Accuracy : ", best_acc)

# print("-"*50)

        

#         #print_accuracy(sn_ce)
#sn_ce.fit(X_scaled_train, Y_train, epochs=best_epoch, learning_rate= best_lr, loss_fn="ce", display_loss=True)



Y_pred_test = sn_ce.predict(X_scaled_test)

#Y_pred_test = best_model.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)