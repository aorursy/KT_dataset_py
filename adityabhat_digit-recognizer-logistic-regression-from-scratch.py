# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
df_train = pd.read_csv('../input/digit-recognizer/train.csv')
df_test = pd.read_csv('../input/digit-recognizer/test.csv')


#Training data
print('Training data shape: ', df_train.shape)
print('Test data shape: ', df_test.shape)
df_train.head()
# Lets make it a binary classification problem. Our aim is to determine whether the digit is 1 or not. 

df_train['label'] = np.where(df_train['label'] == 1, 1, 0) 
# Sigmoid function
def sigmoid(x):
    s = (1 + np.exp(-x))**-1
    return s
X = df_train.iloc[:,1:]
y = df_train.loc[:,['label']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train_arr = X_train.T.values
X_test_arr = X_test.T.values
index = 9
plt.imshow(X_train_arr[:,index].reshape((28,28)))
X_train_st = X_train_arr/255.
X_test_st = X_test_arr/255.
X_train_st.shape
# Dimensions of arrays
m_train = X_train_st.shape[1]
m_test = X_test_st.shape[1]
n = X_train_st.shape[0]

print(m_train)
print(m_test)
print(n)
print(X_train_st.shape)
print(X_test_st.shape)
print(y_train.shape)
print(y_test.shape)
y_train = np.ravel(y_train).reshape(1,m_train)
y_test = np.ravel(y_test).reshape(1,m_test)
# Initialize the weights
#def initialize(X_train_st):
w = np.zeros(n).reshape(n,1)
b = 0
#    return w, b
w.shape
# Get the linear combinaiton of input data & weights
Z = np.dot(w.T, X_train_st)
Z.shape
A = sigmoid(Z)
A.shape
y_train.shape
J = -(1/m_train) * ( np.dot(y_train, A.T)  +   np.dot( (1-y_train), np.log(1- A.T)) )
J
def eval_metrics(y_val,y_pred):
  print('Accuracy: {:.2f}'.format(accuracy_score(y_val, y_pred)))
  print('Precision: {:.2f}'.format(precision_score(y_val, y_pred)))
  print('Recall: {:.2f}'.format(recall_score(y_val, y_pred)))
  print('F1: {:.2f}'.format(f1_score(y_val, y_pred)))
  print('AUC: {:.2f}'.format(roc_auc_score(y_val, y_pred)))
def initialize(n):
    w = np.zeros(n).reshape(n,1)
    b = 0
    return w, b
def forward_prop(X_train_st, y_train, w,b, m_train):
    # Get the linear combinaiton of input data & weights
    Z = (np.dot(w.T, X_train_st) + b)
    A = sigmoid(Z)
    J = -(1/m_train) * ( np.dot(y_train, A.T)  +   np.dot( (1-y_train), np.log(1- A.T)) )
    return Z, A, J
def backward_prop(X_train_st,y_train, w, b, A, m_train, learning_rate):
    dw = (1/m_train) * (np.dot(X_train_st, (A-y_train).T ))
    db = (1/m_train) * (np.sum(A-y_train))
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
    return w, b
def optimize(X_train_st, y_train, w, b, m_train, num_iter, learning_rate):
    for i in np.arange(num_iter):
        print("This is iteration number: ", i)
        Z, A, J = forward_prop(X_train_st, y_train, w, b, m_train)
        print("Loss is :", J)
        w, b = backward_prop(X_train_st,y_train, w, b, A, m_train, learning_rate)
    return w, b   
def train_model(X_train_st, y_train, m_train, n, learning_rate, num_iter):
    w, b = initialize(n)
    w, b = optimize(X_train_st, y_train, w, b, m_train, num_iter, learning_rate)
    return w, b
learning_rate = 0.01
num_iter = 200
w, b = train_model(X_train_st, y_train, m_train, n, learning_rate, num_iter)
def predict(X_test_st,w,b):
    y_pred = sigmoid(np.dot(w.T, X_test_st))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred
y_pred = predict(X_train_st,w,b)
eval_metrics(y_test.T, y_pred.T)
confusion_matrix(y_train.T, y_pred.T)
confusion_matrix(y_test.T, y_pred.T)
