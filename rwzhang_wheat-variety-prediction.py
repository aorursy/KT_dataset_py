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
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

data = np.loadtxt("../input/seeds_dataset.txt", delimiter="\t")
num_row, num_col = data.shape

np.random.seed(2017)
train = np.random.choice([True, False], num_row, replace=True, p=[0.5,0.5])
x_train = [data[train,:-1]]
y_train = data[train,-1].astype(int)
x_test = [data[~train,:-1]]
y_test = data[~train,-1].astype(int)
#build logistic regression
logit = linear_model.LogisticRegression(C=10)
logit.fit(x_train[0][:,[3,6]], y_train)
y_test_predict=logit.predict(x_test[0][:,[3,6]])


color = ['red', 'green','orange']
y_color = [color[i-1] for i in data[:,-1].astype(int)]

def my_linspace(min_value, max_value, steps):
    diff = max_value - min_value
    return np.linspace(min_value - 0.1 * diff, max_value + 0.1 * diff, steps)

steps = 200
x0 = my_linspace(min(data[:,3]), max(data[:,6]), steps)
x1 = my_linspace(min(data[:,3]), max(data[:,6]), steps)
xx0, xx1 = np.meshgrid(x0, x1)
mesh_data = np.c_[xx0.ravel(), xx1.ravel()]
mesh_proba = logit.predict_proba(mesh_data).reshape(steps, steps, 3)


plt.figure(figsize=(12, 12))
plt.scatter(data[:,3], data[:,6], c=y_color)
for i in range(3):
    plt.contourf(xx0, xx1, np.maximum(mesh_proba[:,:,i], 0.5), 20, cmap=plt.cm.Greys, alpha=0.5)
plt.show()

(m,)=y_test.shape
confusion_matrix = np.zeros(shape=(3,3))
for i in range(m):
    confusion_matrix[y_test[i]-1][y_test_predict[i]-1]+=1
print("confusion_matrix:","\n",confusion_matrix)
print("accuracy on test data:",np.trace(confusion_matrix)/np.sum(confusion_matrix))
print('score on test data:', logit.score(x_test[0][:,[3,6]], y_test))

