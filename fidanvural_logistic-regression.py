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
data=pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

data.head()
x_data=data.drop(["Outcome"],axis=1)

y=data.Outcome.values



x_data.head()
x_data.tail()
x=(x_data-np.min(x_data))/((np.max(x_data)-np.min(x_data))).values

x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("x_train: {}".format(x_train.shape))

print("x_test: {}".format(x_test.shape))

print("y_train: {}".format(y_train.shape))

print("y_test: {}".format(y_test.shape))
x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T
print("x_train: {}".format(x_train.shape))

print("x_test: {}".format(x_test.shape))

print("y_train: {}".format(y_train.shape))

print("y_test: {}".format(y_test.shape))
def initialize_weights_and_bias(dimension):

    w=np.full((dimension,1),0.01) # (dimension,1) matrisinin başlangıç weightlerini 0.01 yaptık.

    b=0.0

    

    return w,b
def sigmoid(z):

    y_head=1/(1+np.exp(-z)) # Sigmoid fonksiyonunun formülü -> 1/(1+exp(-z))

    

    return y_head
sigmoid(0) # Fonksiyona bir değer vererek denedik.
def forward_and_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z=np.dot(w.T,x_train)+b # w.T dememizin sebebi matris çarpımı yapabilmemiz için uygun hale getirmek. w -> (diemnsion,1)  w.T -> (1,dimension)

    y_head=sigmoid(z)

    

    # loss ve cost functions

    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1] 

    

    # backward propagation

    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # türev aldık

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1] # türev aldık

    

    gradients={"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    

    return cost,gradients
def update(w,b,x_train,y_train,learning_rate,iteration):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    for i in range(iteration):

        

        cost,gradients=forward_and_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost) # costları tutuyorum. Çünkü ileride grafik çizdirirken gerekli olacak.

        

        # weight ve bias güncelleme

        

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        

        if i%10==0: # buradaki 10'u biz belirledik. 10 adımda bir costu göstermesini istediğim için

            cost_list2.append(cost)

            index.append(i)

        

    parameters={"weight":w,

                "bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Iteration")

    plt.ylabel("Cost")

    plt.show()

    

    return parameters,gradients,cost_list
def prediction(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test)+b)

    

    y_prediction=np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_prediction[0,i]=0

        else:

            y_prediction[0,i]=1

            

    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,iteration):

    

    dimension=x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)

    

    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,iteration)

    

    y_prediction_test=prediction(parameters["weight"],parameters["bias"],x_test)

    

    # accuracy bizim doğruluk oranımızdır.

    

    print("test accuracy: {} %".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=0.01,iteration=200)
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=2,iteration=200)
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=2,iteration=400)
from sklearn.linear_model import LogisticRegression



lr=LogisticRegression(random_state = 42,max_iter= 150)

lr.fit(x_train.T,y_train.T)



print("Train Accuracy: {}".format(lr.score(x_train.T,y_train.T)))

print("Test Accuracy: {}".format(lr.score(x_test.T,y_test.T)))