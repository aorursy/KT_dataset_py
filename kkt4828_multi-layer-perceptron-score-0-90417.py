# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.head(1)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head(1)
X_train = train.drop(['label'], axis=1).copy()
y_train = train['label']

print(X_train.shape, y_train.shape)
# create validation set

from sklearn.model_selection import train_test_split

X_train_f, X_valid_f, y_train_f, y_valid_f = train_test_split(X_train,y_train, random_state = 5, test_size = 0.2)


print(X_train_f.shape, X_valid_f.shape)
print(y_train_f.shape, y_valid_f.shape)
from keras.utils import to_categorical

# One Hot Encoding
y_train_hot = to_categorical(y_train_f)
y_valid_hot = to_categorical(y_valid_f)

print(y_train_hot.shape, y_valid_hot.shape)
X_train_f = X_train_f.T
X_valid_f = X_valid_f.T
y_train_hot = y_train_hot.T
y_valid_hot = y_valid_hot.T

print(X_train_f.shape, X_valid_f.shape)
print(y_train_hot.shape, y_valid_hot.shape)
# Define Sigmoid

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Define Cross_entropy
def cross_entropy(actual, predict, eps=1e-15):
    
    actual = np.array(actual)
    predict = np.array(predict)
    
    
    clipped_predict = np.minimum(np.maximum(predict, eps), 1 - eps)
    
    
    loss = actual * np.log(clipped_predict) + (1 - actual) * np.log(1 - clipped_predict)
    
    return -1.0 * loss.mean()
# Define dsigmoid
def dsigmoid(A):
    
    dA = A * (1 - A)
    
    return dA
import warnings
warnings.filterwarnings(action='ignore')

num_epoch = 300
learning_rate = 3

# 우리가 학습해야하는 값들을 먼저 정의해줍니다.
w1 = np.random.uniform(low=-1.0, high=1.0, size=(600, 784))  # (num_labels, num_nodes)
b1 = np.random.uniform(low=-1.0, high=1.0, size=(600, 1)) # (num_labels, 1)

w2 = np.random.uniform(low=-1.0, high=1.0, size=(10, 600))
b2 = np.random.uniform(low=-1.0, high=1.0, size=(10, 1))
# 샘플 수도 저장해줍니다.
num_data = X_train_f.shape[1]

# 학습 시작!
for epoch in range(num_epoch):
    # Multi-layer Perceptron!
    z1 = np.dot(w1, X_train_f) + b1
    a1 = sigmoid(z1)
    
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    
    y_predict_hot = a2
    
    y_predict = np.argmax(y_predict_hot, axis=0)
    accuracy = (y_predict == y_train_f).mean()   
    
    
    if accuracy > 0.92:
        break

    
    loss = cross_entropy(y_train_hot, y_predict_hot)

    
    if epoch % 10 == 0:
        print("{0:2} accuracy = {1:.5f}, loss = {2:.5f}".format(epoch, accuracy, loss))

    # Gradient Descent
    d2 = a2 - y_train_hot
    d1 = np.dot(w2.T, d2) * dsigmoid(a1)
    
    w2 = w2 - learning_rate * np.dot(d2, a1.T) / num_data
    w1 = w1 - learning_rate * np.dot(d1, X_train_f.T) / num_data
    b2 = b2 - learning_rate * d2.mean(axis = 1, keepdims = True)
    b1 = b1 - learning_rate * d1.mean(axis = 1, keepdims=True) 

print("----" * 10)
print("{0:2} accuracy = {1:.5f}, loss = {2:.5f}".format(epoch, accuracy, loss))
# Model validation


z1 = np.dot(w1, X_train_f) + b1
a1 = sigmoid(z1)
    
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)
    
y_predict_hot = a2

y_predict = np.argmax(y_predict_hot, axis=0)
accuracy = (y_predict == y_train_f).mean()


train_result = pd.DataFrame({'actual': y_train_f, 'predict': y_predict})


train_accuracy = (train_result["actual"] == train_result["predict"]).mean(axis=0)
print("Accuracy(train) = {0:.5f}".format(train_accuracy))

print(train_result.shape)
train_result.head(10)
# Model Evaluation

z1 = np.dot(w1, X_valid_f) + b1
a1 = sigmoid(z1)
    
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)
    
y_predict_hot = a2
y_predict = np.argmax(y_predict_hot, axis=0)
accuracy = (y_predict == y_valid_f).mean()

# actual vs. predict
valid_result = pd.DataFrame({'actual': y_valid_f, 'predict': y_predict})

# accuracy는 다음과 같이 계산됩니다.
valid_accuracy = (valid_result["actual"] == valid_result["predict"]).mean(axis=0)
print("Accuracy(valid) = {0:.5f}".format(valid_accuracy))

print(valid_result.shape)
valid_result.head(10)

X_test = test.T

print(X_test.shape)
X_test.head()
z1 = np.dot(w1, X_test) + b1
a1 = sigmoid(z1)
    
z2 = np.dot(w2, a1) + b2
a2 = sigmoid(z2)
y_predict_hot = a2
y_test_predict = np.argmax(y_predict_hot, axis=0)
y_test_predict
submission = pd.read_csv('sample_submission.csv', index_col="ImageId")
submission.head()
submission["Label"]=y_test_predict
submission.head()
submission.to_csv("test_1.csv")



