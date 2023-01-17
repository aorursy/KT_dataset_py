# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.head()
data.columns
data.info()
data.describe()
x=data.drop('Class', axis=1)

y=data.Class.values



# normalization

x=(x-np.min(x))/(np.max(x)-np.min(x))
# check if x is normalized

x.describe()
# train-test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)



x_train=x_train.T

y_train=y_train.T

x_test=x_test.T

y_test=y_test.T



print('Shape of x_train: ',x_train.shape)

print('Shape of y_train: ',y_train.shape)

print('Shape of x_test: ',x_test.shape)

print('Shape of y_test: ',y_test.shape)
# initialize w and b

def initialize_w_and_b(length):

    w=np.full(shape=(length,1), fill_value=0.01)

    b=0.0

    return w,b



def sigmoid(z):

    return 1/(1+np.exp(-z))
# forward and backward propagation

def forward_and_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    z=np.dot(w.T, x_train) + b

    y_head=sigmoid(z)

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost=np.sum(loss)/x_train.shape[1]

    

    #backward propagation

    derivative_w=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_b=np.sum(y_head-y_train)/x_train.shape[1]

    

    return cost,derivative_w,derivative_b
# update w and b

def update_w_and_b(w,b,x_train,y_train,learning_rate,number_of_iterations):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    # update parameters

    for i in range(number_of_iterations):

        cost,derivative_w,derivative_b = forward_and_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate*derivative_w

        b = b - learning_rate*derivative_b

        

        if i%100 == 0:

            cost_list2.append(cost)

            index.append(i)

            print('Cost after iteration {}: {}'.format(i,cost))

    

    #draw

    plt.plot(index, cost_list2)

    plt.xlabel('Number of Iterations')

    plt.ylabel('Cost')

    plt.show()

    

    return w,b
# logistic regression

def logistic_regression(x_train, x_test, y_train, y_test, learning_rate, number_of_iterations):

    # initialize and update parameters

    w,b = initialize_w_and_b(x_train.shape[0])

    w,b=update_w_and_b(w,b,x_train,y_train,learning_rate,number_of_iterations)

    

    # form probabilities and predict

    y_head=sigmoid(np.dot(w.T, x_test) + b)

    y_predicted=np.array([1 if each>0.5 else 0 for each in y_head[0]])

    

    #accuracy

    print('Accuracy: ',100-np.mean(np.abs(y_test-y_predicted))*100)

    

    return y_predicted
y_predicted = logistic_regression(x_train, x_test, y_train, y_test, 0.02, 1000)
from sklearn.linear_model import LogisticRegression



log_reg=LogisticRegression()



log_reg.fit(x_train.T,y_train.T)



y_predicted = log_reg.predict(x_test.T)



print('Accuracy: ',100-np.mean(np.abs(y_test-y_predicted))*100)



print('Accuracy2:',log_reg.score(x_test.T, y_test.T)*100)