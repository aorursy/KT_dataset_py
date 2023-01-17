# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
#reading the data

x1 = df['fixed acidity'].values

x2 = df['volatile acidity'].values

x3 = df['citric acid'].values

x4 = df['residual sugar'].values

x5 = df['chlorides'].values

x6 = df['free sulfur dioxide'].values

x7 = df['total sulfur dioxide'].values

x8 = df['density'].values

x9 = df['pH'].values

x10 = df['sulphates'].values

x11 = df['alcohol'].values

y = df['quality'].values
df.describe()
#feature scaling using Mean Normalization

X1=x1

X2=x2

X3=x3

X4=x4

X5=x5

X6=(x6-x6.mean())/x6.std()

X7=(x7-x7.mean())/x7.std()

X8=x8

X9=x9

X10=x10

X11=x11
#hypothesis

def hypothesis(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):

    return theta[0] + theta[1]*x1 + theta[2]*x2 + theta[3]*x3 + theta[4]*x4 + theta[5]*x5 + theta[6]*x6 + theta[7]*x7 + theta[8]*x8 + theta[9]*x9 + theta[10]*x10 + theta[11]*x11
#cost function

def cost(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y):

    m = x1.shape[0]

    error = 0

    for i in range(m):

        hx = hypothesis(theta, x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i], x10[i], x11[i])

        error = error + (hx - y[i])**2

    return error
#partial derivative of the cost function

def diffGradient(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y):

    m = x1.shape[0]

    grad = np.zeros((12,))

    for i in range(m):

        hx = hypothesis(theta, x1[i], x2[i], x3[i], x4[i], x5[i], x6[i], x7[i], x8[i], x9[i], x10[i], x11[i])

        grad[0] = grad[0] + (hx - y[i])

        grad[1] = grad[1] + (hx - y[i])*x1[i]

        grad[2] = grad[2] + (hx - y[i])*x2[i]

        grad[3] = grad[3] + (hx - y[i])*x3[i]

        grad[4] = grad[4] + (hx - y[i])*x4[i]

        grad[5] = grad[5] + (hx - y[i])*x5[i]

        grad[6] = grad[6] + (hx - y[i])*x6[i]

        grad[7] = grad[7] + (hx - y[i])*x7[i]

        grad[8] = grad[8] + (hx - y[i])*x8[i]

        grad[9] = grad[9] + (hx - y[i])*x9[i]

        grad[10] = grad[10] + (hx - y[i])*x10[i]

        grad[11] = grad[11] + (hx - y[i])*x11[i]

    return grad
#finding the gradient descent

def gradientDescent(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y, learning_rate = 0.0001):

    theta = np.zeros((12,),dtype = float)

    error_list = []

    theta_list = []

    max_iter = 300

    m = x1.shape[0]

    for i in range(max_iter):

        grad = diffGradient(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y)

        e = cost(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y)

        error_list.append(e)

        theta_list.append((theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9],theta[10],theta[11]))

        #simultaneously update theta values

        theta[0] = theta[0] - learning_rate*(1/m)*grad[0]

        theta[1] = theta[1] - learning_rate*(1/m)*grad[1]

        theta[2] = theta[2] - learning_rate*(1/m)*grad[2]

        theta[3] = theta[3] - learning_rate*(1/m)*grad[3]

        theta[4] = theta[4] - learning_rate*(1/m)*grad[4]

        theta[5] = theta[5] - learning_rate*(1/m)*grad[5]

        theta[6] = theta[6] - learning_rate*(1/m)*grad[6]

        theta[7] = theta[7] - learning_rate*(1/m)*grad[7]

        theta[8] = theta[8] - learning_rate*(1/m)*grad[8]

        theta[9] = theta[9] - learning_rate*(1/m)*grad[9]

        theta[10] = theta[10] - learning_rate*(1/m)*grad[10]

        theta[11] = theta[11] - learning_rate*(1/m)*grad[11]

    return theta, error_list, theta_list
final_theta, error_list, theta_list = gradientDescent(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, y)
print(final_theta)
plt.plot(error_list, label='Cost Function')

plt.xlabel("No. of iterations")

plt.ylabel("Error")

plt.legend()

plt.show()
def predict(theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):

    hx = hypothesis(final_theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)

    return hx
x1 = np.array([12.6])

x2 = np.array([0.31])

x3 = np.array([0.72])

x4 = np.array([2.2])

x5 = np.array([0.07200000000000001])

x6 = np.array([6.0])

x7 = np.array([29.0])

x8 = np.array([0.9987])

x9 = np.array([2.88])

x10 = np.array([0.82])

x11 = np.array([9.8])
prediction = hypothesis(final_theta, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11)

print(prediction)