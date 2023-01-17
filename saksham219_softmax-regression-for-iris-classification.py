import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.cross_validation import train_test_split

import math

import warnings

warnings.filterwarnings('ignore')
def phi(i,theta,x):  #i goes from 1 to k

    mat_theta = np.matrix(theta[i])

    mat_x = np.matrix(x)

    num = math.exp(np.dot(mat_theta,mat_x.T))

    den = 0

    for j in range(0,k):

        mat_theta_j = np.matrix(theta[j])

        den = den + math.exp(np.dot(mat_theta_j,mat_x.T))

    phi_i = num/den

    return phi_i
def indicator(a,b):

    if a == b: return 1

    else: return 0
def get__der_grad(j,theta):

    sum = np.array([0 for i in range(0,n+1)])

    for i in range(0,m):

        p = indicator(y[i],j) - phi(j,theta,x.loc[i])

        sum = sum + (x.loc[i] *p)

    grad = -sum/m

    return grad
def gradient_descent(theta,alpha= 1/(10^4),iters=500):

    for j in range(0,k):

        for iter in range(iters):

            theta[j] = theta[j] - alpha * get__der_grad(j,theta)

    print('running iterations')

    return theta
def h_theta(x):

    x = np.matrix(x)

    h_matrix = np.empty((k,1))

    den = 0

    for j in range(0,k):

        den = den + math.exp(np.dot(theta_dash[j], x.T))

    for i in range(0,k):

        h_matrix[i] = math.exp(np.dot(theta_dash[i],x.T))

    h_matrix = h_matrix/den

    return h_matrix
iris = pd.read_csv('../input/Iris.csv')

iris.head()
iris = iris.drop(['Id'],axis=1)

iris.head()
train, test = train_test_split(iris, test_size = 0.3)# in this our main data is split into train and test

train = train.reset_index()

test = test.reset_index()
x = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

n = x.shape[1]

m = x.shape[0]
y = train['Species']

k = len(y.unique())

y =y.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

y.value_counts()
x[5] = np.ones(x.shape[0])

x.shape
theta = np.empty((k,n+1))
theta_dash = gradient_descent(theta)
theta_dash
x_u = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

n = x_u.shape[1]

m = x_u.shape[0]
y_true = test['Species']

k = len(y_true.unique())

y_true =y_true.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})

y_true.value_counts()
x_u[5] = np.ones(x_u.shape[0])

x_u.shape
for index,row in x_u.iterrows():

    h_matrix = h_theta(row)

    prediction = int(np.where(h_matrix == h_matrix.max())[0])

    x_u.loc[index,'prediction'] = prediction
results = x_u

results['actual'] = y_true
results.head(10)
compare = results['prediction'] == results['actual']

correct = compare.value_counts()[1]

accuracy = correct/len(results)
accuracy * 100