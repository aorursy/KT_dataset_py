# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
y_l=np.load("/kaggle/input/sign-language-digits-dataset/Y.npy")

x_l=np.load("/kaggle/input/sign-language-digits-dataset/X.npy")
liste1=[]

liste2=[]

for i in range(y_l.shape[0]-1):

    if list(y_l[i])==list(y_l[i+1]):

        liste1.append(i)

    else:

        liste2.append(i)
liste2
plt.imshow(x_l[203].reshape(64,64))
plt.imshow(x_l[204].reshape(64,64))
liste2
plt.imshow(x_l[408].reshape(64,64))
liste2
plt.imshow(x_l[409].reshape(64,64))
plt.imshow(x_l[614].reshape(64,64))
plt.imshow(x_l[821].reshape(64,64))
plt.imshow(x_l[1027].reshape(64,64))
plt.imshow(x_l[1235].reshape(64,64))
plt.imshow(x_l[1442].reshape(64,64))
plt.imshow(x_l[1648].reshape(64,64))
plt.imshow(x_l[1854].reshape(64,64))
plt.imshow(x_l[2061].reshape(64,64))
y_l.shape
liste2
img_size=64

plt.subplot(1,2,1)

plt.imshow(x_l[408].reshape(img_size,img_size))

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(x_l[900].reshape(img_size,img_size))

plt.axis("off")
x=np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)

z=np.zeros(205)

o=np.ones(205)

y=np.concatenate((z,o),axis=0).reshape(x.shape[0],1)

print("x shape:", x.shape)

print("y shape:", y.shape)
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.14, random_state=42)

number_of_train=x_train.shape[0]

number_of_test=x_test.shape[0]
x_train_2d=x_train.reshape(number_of_train, x_train.shape[1]*x_train.shape[2])

x_test_2d=x_test.reshape(number_of_test, x_test.shape[1]*x_test.shape[2])

print("x_train flatten:", x_train_2d.shape)

print("y_train flatten:", x_test_2d.shape)
x_train=x_train_2d.T

x_test=x_test_2d.T

y_train=y_train.T

y_test=y_test.T
print("x_train", x_train.shape)

print("x_test", x_test.shape)

print("y_train", y_train.shape)

print("y_test", y_test.shape)
def initialize_weight_and_bias(dimension):

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b



def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head
x_train.shape
def forward_backward_propogation(w,b,x_train,y_train):

    z=np.dot(w.T,x_train)+b

    y_head=sigmoid(z)

    loss=-y_train*(np.log(y_head))-((1-y_train)*np.log(1-y_head))

    cost=np.sum(loss)/x_train.shape[1]

    

    derivative_weight=np.dot(x_train,(y_head-y_train).T)/x_train.shape[1]

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]

    gradients={"derivative_weight":derivative_weight, "derivative_bias":derivative_bias}

    return cost,gradients
def update(w,b,x_train,y_train,learning_rate,num_of_iter):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    for i in range(num_of_iter):

        cost,gradients=forward_backward_propogation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w=w-(learning_rate*gradients["derivative_weight"])

        b=b-(learning_rate*gradients["derivative_bias"])

        

        if i % 10==0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i %f"%(i,cost))

            

    parameters={"weight":w, "bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index, rotation="vertical")

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
x_test.shape
def predict(w,b,x_test):

    y_prob=sigmoid(np.dot(w.T,x_test)+b)

    y_prediction=np.zeros((1,x_test.shape[1]))

    

    for i in range(y_prob.shape[1]):

        if y_prob[0,i]>0.5:

            y_prediction[0,i]=1

        elif y_prob[0,i]<=0.5:

            y_prediction[0,i]=0

    return y_prediction
x_train.shape
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_of_iter):

    dimension=x_train.shape[0]

    w,b=initialize_weight_and_bias(dimension)

    

    parameters, gradient, cost_list=update(w,b,x_train,y_train,learning_rate, num_of_iter)

    

    y_prediction=predict(parameters["weight"], parameters["bias"], x_test)

    

    print("Test Accuracy:", 100-np.mean(np.abs(y_prediction-y_test))*100, "%")
logistic_regression(x_train, y_train, x_test, y_test, 0.01, 190)