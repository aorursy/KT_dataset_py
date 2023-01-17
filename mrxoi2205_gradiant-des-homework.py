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
import numpy as np

x = 2 * np.random.rand(100,1)

y = 4 +3 * x+np.random.randn(100,1)*1.5
def cal_cost(w,x,y):

    m=len(y)

    A=x.dot(w)

    cost =1/(2*m)* np.sum(np.square(A-y))

    return cost



def gradient_descent(x,y,w,learning_rate=0.01,iterations=100):



    m = len(y)

    cost_history = np.zeros(iterations)

    w_history = np.zeros((iterations,2))

    for it in range(iterations):

        A=np.dot(x,w)

        w=w-(1/m)*learning_rate*(x.T.dot((A-y)))

        w_history[it,:] =w.T

        cost_history[it]  = cal_cost(w,x,y)

        

    return w, cost_history, w_history



lr=0.1

n_inter=10000



w=np.random.randn(2,1)

x_vector=np.c_[np.ones((len(x),1)),x]



w,cost_history,w_history = gradient_descent(x_vector,y,w,lr,n_inter)



print(w)

#Dung cong thuc sau khi bien doi dao ham

x_vector=np.c_[np.ones((100,1)),x]

A=np.dot(x_vector.T,x_vector)

B=np.dot(x_vector.T,y)

g=np.dot(np.linalg.pinv(A), B)

print('g=',g)