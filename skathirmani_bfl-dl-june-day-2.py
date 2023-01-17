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
import pandas as pd
data = pd.read_csv('/kaggle/input/trainings/employees_attrition.csv')
data.shape
data['Attrition'].value_counts() / data.shape[0] * 100
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
X = data[['MonthlyIncome', 'Age']]
y = data['Attrition']
X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
model = LogisticRegression(solver='lbfgs', C=1e15).fit(X_std, y)
model.coef_, model.intercept_
import numpy as np
np.log(0)
def calc_logloss(y, yhat):
    yhat = yhat+0.0000001 if yhat==0 else yhat
    if y == 1:
        return -np.log(yhat)
    else:
        return -np.log(1-yhat)
import matplotlib.pyplot as plt
yhat = np.linspace(0,1, 100)
y_org = 0
errors = [calc_logloss(y_org, i) for i in yhat]
plt.plot(yhat, errors)
def calc_cross_entropy(y, yhat):
    error = y * np.log(yhat) + (1-y)*np.log(1-yhat)
    return error
print(calc_cross_entropy(1, 0.9))
#print(calc_cross_entropy())
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def forward_propogate(features, weights):
    return sigmoid(np.dot(features, weights))

def calc_gradient(features, predictions, y):
    y = y.values.reshape(len(y),1)
    gradient = (np.dot(features.T, predictions-y))
    return gradient
df_X_std = X_std.copy()
df_X_std['bias'] = 1
df_X_std.head()
weights = np.zeros((len(df_X_std.columns),1))
features = df_X_std.values
lr = 0.001
errors = []
for i in range(1000):
    predictions = forward_propogate(features, weights)
    error = log_loss(y, predictions)
    errors.append(error)
    gradient = calc_gradient(features, predictions, y)
    weights = weights - gradient * lr
weights
import matplotlib.pyplot as plt
plt.plot(errors)
from keras.layers import Dense
from keras import Sequential
X_std.shape
## One hidden layer: 64 neurons; activation: sigmoid
## one neuron for o/p: activation: sigmoid
## i/p layer no. of neurons: 2 neurons