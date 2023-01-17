import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

np.random.seed(100)
LEVEL = 'level_3'
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
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
class SigmoidNeuron_V:
  
  def __init__(self):
    self.W = None
    self.b = None
    
  def perceptron(self, X):
    return np.dot(X, self.W.T) + self.b
  
  def sigmoid(self, X):
    return 1.0/(1.0 + np.exp(-X))
  
  def grad_w_mse(self, X, y):
    y_pred = self.sigmoid(self.perceptron(X))
    return np.matmul(((y_pred - y.reshape(y_pred.shape[0], 1)) * y_pred * (1 - y_pred)).T, X)
  
  def grad_b_mse(self, X, y):
    y_pred = self.sigmoid(self.perceptron(X))
    return np.sum((y_pred - y.reshape(y_pred.shape[0], 1)) * y_pred * (1 - y_pred))
  
  def grad_w_ce(self, X, y):
    y_pred = self.sigmoid(self.perceptron(X))
    return np.matmul((y_pred - y.reshape(y_pred.shape[0], 1)).T, X)
    
  def grad_b_ce(self, X, y):
    y_pred = self.sigmoid(self.perceptron(X))
    return np.sum((y_pred - y.reshape(y_pred.shape[0], 1)))
  
  def fit(self, X, y, epochs=1, learning_rate=1, initialise=True, loss_fn="mse", display_loss=False, display_weight=False):
    
    # initialise w, b
    if initialise:
      self.W = np.random.randn(1, X.shape[1])
      self.b = 0
      
    if display_loss:
      loss = {}
    
    for i in tqdm_notebook(range(epochs), total=epochs, unit="epoch"):
      dw = 0
      db = 0
      if loss_fn == "mse":
        dw = self.grad_w_mse(X, y)
        db = self.grad_b_mse(X, y) 
      elif loss_fn == "ce":
        dw = self.grad_w_ce(X, y)
        db = self.grad_b_ce(X, y)
      
      self.W -= learning_rate * dw
      self.b -= learning_rate * db
      
      if display_loss:
        Y_pred = self.sigmoid(self.perceptron(X))
        if loss_fn == "mse":
          loss[i] = mean_squared_error(y, Y_pred)
        elif loss_fn == "ce":
          loss[i] = log_loss(y, Y_pred)
        
      if display_weight:
        weight_matrix = np.array([[self.b] + list(sn_ce.W[0])[23:38]])
        weight_matrices.append(weight_matrix)
    
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
    Y_pred.append(self.sigmoid(self.perceptron(X)))
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

images_train = read_all("../input/level-3-dataset/"+LEVEL+"_train/"+LEVEL+"/background", key_prefix='bgr_') # change the path
for language in languages:
  images_train.update(read_all("../input/level-3-dataset/"+LEVEL+"_train/"+LEVEL+ "/" +language, key_prefix=language+"_" ))
print(len(images_train))

images_test = read_all("../input/level-3-dataset/level_3_test/kaggle_"+LEVEL, key_prefix='') # change the path
print(len(images_test))
list(images_test.keys())[:5]
image_filter = lambda x: 255 if x <= 10 else 0
def border_filter(img, padding =3):
    img = img.reshape(64, 64)
    img[:padding] = 0
    img[:, :padding] = 0
    img[-padding:] = 0
    img[:, -padding:] = 0
    return img
X_train = []
Y_train = []
for key, value in images_train.items():
    X_train.append(border_filter(np.array([ image_filter(x) for x in value ])).flatten())
    if key[:4] == "bgr_":
        Y_train.append(0)
    else:
        Y_train.append(1)

ID_test = []
X_test = []
for key, value in images_test.items():
  ID_test.append(int(key))
  X_test.append(border_filter(np.array([ image_filter(x) for x in value ])).flatten())
  
        
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

print(X_train.shape, Y_train.shape)
print(X_test.shape)
def imshow(image):
    image = image.reshape(64, 64)
    plt.axis('off')
    plt.imshow(image)
plt.imshow(X_train[7111].reshape(64, 64))
imshow(X_train[8556])
scaler = MinMaxScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)
imshow(X_scaled_train[8556])
sn_mse = SigmoidNeuron_V()
sn_mse.fit(X_scaled_train, Y_train, epochs=1000, learning_rate=0.009, loss_fn="mse", display_loss=True)
# Epoch - 1000, lr - 0.005
sn_ce = SigmoidNeuron_V()
sn_ce.fit(X_scaled_train, Y_train, epochs=1000, learning_rate=0.005, loss_fn="ce", display_loss=True)
def print_accuracy(sn):
  Y_pred_train = sn.predict(X_scaled_train)
  Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()
  accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
  print("Train Accuracy : ", accuracy_train)
  print("-"*50)
print_accuracy(sn_mse)
print_accuracy(sn_ce)
vals = sn_ce.W.reshape(64, 64)
plt.figure(figsize=[10, 10])
sns.heatmap(vals, cmap='coolwarm', square=True, vmin=-3, vmax=3)
Y_pred_test = sn_ce.predict(X_scaled_test)
Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()

submission = {}
submission['ImageId'] = ID_test
submission['Class'] = Y_pred_binarised_test

submission = pd.DataFrame(submission)
submission = submission[['ImageId', 'Class']]
submission = submission.sort_values(['ImageId'])
submission.to_csv("submisision.csv", index=False)
