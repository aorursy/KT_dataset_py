# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
from sklearn.model_selection import train_test_split

X = df.drop(['target', 'age', 'trestbps', 'thalach', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'] , axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.head()
y_train.head()
m = X_train.shape[0]

w = np.zeros(X_train.shape[1])

b = 0

learning_rate = 0.0001

# initialize error function

j = 1

# z = w * transpose(x) + b

# a = sigmoid(z)

# j = -1 * (y * log(a) + (1 - y) * log(1 - a)) 

# dz = a - y

# dw = x * dz 

(np.dot(w, np.transpose(X_train)) + b).shape

print(m)

print(w)
from scipy.special import expit

for i in range(100000):

    z = np.dot(w, np.transpose(X_train)) + b

    # Predicted probability (Forward propogation)

    a = expit(z)

    # losses for each training examples

    l = -1 * (np.dot(y_train, np.log(a)) + np.dot((1 - y_train), np.log(1 - a)))

    j = np.sum(l) / m

    # Back propogating to get partial derivative of loss function w.r.t z

    dz = a - y_train

    # Further back propogation to get partial derivaive w.r.t weights

    dw = np.dot(np.transpose(X_train), dz) / m

    db = np.sum(dz) / m

    # Resetting weights and bias

    w -= learning_rate * dw

    b -= learning_rate * db

    if(i % 10000 == 0):

        print(f"iter: {i}, loss: {j}")
print(w)

print(b)
pred_df = pd.DataFrame()

pred_df['Predicted'] = expit(np.dot(w, np.transpose(X_test)) + b)

pred_df['Actual'] = list(y_test)



correct = 0

for i in range(len(list(y_test))):

    threshold = 0.5

    if((pred_df['Predicted'][i] > threshold and pred_df['Actual'][i] == 1) or (pred_df['Predicted'][i] < threshold and pred_df['Actual'][i] == 0)):

        correct += 1

        

print(f"Accuracy: {correct / len(list(y_test))}")

print(pred_df)