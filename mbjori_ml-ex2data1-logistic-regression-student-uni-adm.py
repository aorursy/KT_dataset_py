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
students = pd.read_csv('/kaggle/input/ex2data1/ex2data1.txt', header = None)
students.rename(columns = {0: 'Exam 1 Score', 1: 'Exam 2 Score', 2: 'Result'}, inplace = True)
students
import seaborn as sns

g =sns.scatterplot(x = 'Exam 1 Score', y = 'Exam 2 Score', hue = 'Result', data = students, style = 'Result')
X = students[['Exam 1 Score', 'Exam 2 Score']]
theta0 = np.ones((len(X),1))

X = np.array(X)
X = np.concatenate((theta0, X), axis = 1)

y = students[['Result']]
y = np.array(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
iterations = 100000

theta = np.zeros(X.shape[1]).reshape((1,3))
m = len(X_train)
alpha = 0.001

def sigmoid(tempVal):
    return 1/(1 + np.exp(- tempVal))

# def deltaJ(tempX, tempy, tempTheta):
#     dotThetaX = np.dot(tempX, tempTheta.T)
#     print (dotThetaX)
#     hyp = np.apply_along_axis(sigmoid,arr = dotThetaX, axis = 1)
#     deltaVal = np.sum(np.dot((hyp - tempy).T, tempX), axis = 0)
#     return deltaVal

JthetaArr = []
for i in range(iterations):
    dotThetaX = np.dot(X_train,theta.T)
    hyp = np.apply_along_axis(sigmoid,arr = dotThetaX, axis = 1)
    Jtheta = -1/m * (np.sum(np.multiply(y_train, np.log(hyp))+ np.multiply(1 - y_train, 1 - np.log(hyp)) ,axis = 0))
    JthetaArr.append([i,Jtheta])
    deltaVal = 1/m * np.sum(np.dot((hyp - y_train).T, X_train), axis = 0)
    deltaVal = deltaVal.reshape((1,3))
    theta = theta - (alpha * deltaVal)
theta
hyp = np.apply_along_axis(sigmoid,arr = np.dot(X_test, theta.T), axis = 1)
# print (np.round(hyp))
# hyp[hyp >= 0.5] = 1
# hyp[hyp < 0.5] = 0
# hyp

print (np.sum(np.absolute(np.round(hyp) - y_test)))
# For a student with an Exam 1 score of 45 and an Exam 2 score of 85, you should expect to see an admission probability of 0.776.

tempX_test = np.array([[1, 45, 85]])
hypTest = np.apply_along_axis(sigmoid,arr = np.dot(tempX_test, theta.T), axis = 1)
hypTest