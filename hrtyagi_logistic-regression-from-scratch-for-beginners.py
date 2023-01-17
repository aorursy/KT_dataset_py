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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
plt.style.use("seaborn")
mean_01=np.array([1,0.5])

cov_01=np.array([[1,0.1],[0.1,1.2]])

print("parameters for x1",mean_01,cov_01,sep="\n")

mean_02=np.array([4,5])

cov_02=np.array([[1.21,0.1],[0.1,1.3]])

print("parameters for x2",mean_01,cov_01,sep="\n")
dist_01=np.random.multivariate_normal(mean_01,cov_01,500)

dist_02=np.random.multivariate_normal(mean_02,cov_02,500)
plt.figure(0)

plt.scatter(dist_01[:,0],dist_01[:,1],label="class0")

plt.scatter(dist_02[:,0],dist_02[:,1],color="r",marker="^",label="class1")

plt.xlim(-3,8)

plt.ylim(-3,8)

plt.xlabel("x1")

plt.ylabel("x2")

plt.legend()

plt.show()
data=np.vstack((dist_01,dist_02))
np.random.shuffle(data)#as we created two different distributions and merged them.
print(data)
# As our output should be either 1 or 0 becoz we have two classes or we are using binary classification.

def sigmoid(x):

    training_data=[]

    mean=np.mean(x[:,-1])

    for i in x[:,-1]:

        if i > mean :

            training_data.append(1)

        elif i<mean:

            training_data.append(0)

    training_data= np.array(training_data)

    return training_data
Y=sigmoid(data)
print(Y.shape)

print(Y[0:10])
X_train=data[:800,:]

X_test=data[800:,:]

Y_train=Y[:800,]

Y_test=Y[800:,]

print(X_train.shape,X_test.shape)

print(Y_train.shape,Y_test.shape)

#we used 20 % data testing purpose and rest for training
def hypothesis(x,w,b):

    h=np.dot(x,w)+b

    return sigmoid(h)



def sigmoid(z):

    return 1.0/(1.0+np.exp(-1.0*z))



def error(y_true,x,w,b):

    m=x.shape[0]

    err=0.0

    for i in range(m):

        hx = hypothesis(x[i],w,b)

        err += y_true[i]*np.log2(hx) + (1-y_true[i])*np.log2(1-hx)

    return -err/m



def get_grad(y_true,x,w,b):

    grad_w=np.zeros(w.shape)

    grad_b=0.0

    m=x.shape[0]

    for i in range(m):

        hx=hypothesis(x[i],w,b)

        grad_w += (y_true[i]-hx)*x[i]

        grad_b += (y_true[i]-hx)

    grad_w /=m

    grad_b /=m

    return [grad_w,grad_b]



def grad_descent(x,y_true,w,b,learning_rate=0.1):

    err=error(y_true,x,w,b)

    [grad_w,grad_b] = get_grad(y_true,x,w,b)

    

    w = w +learning_rate* grad_w

    b = b + learning_rate* grad_b

    return err,w,b



def predict(x,w,b):

    confidence = hypothesis(x,w,b)

    if confidence <0.5:

        return 0

    else:

        return 1



def get_acc(x_tst,y_tst,w,b):

    y_pred=[]

    for i in range(x_tst.shape[0]):

        p=predict(x_tst[i],w,b)

        y_pred.append(p)

    y_perd=np.array(y_pred)

    return float((y_pred==y_tst).sum())/y_tst.shape[0]
w = 2*np.random.random((X_train.shape[1],))

b = 5*np.random.random()

loss=[]

acc=[]
#training 

for i in range (500):

    l,w,b = grad_descent(X_train,Y_train,w,b,learning_rate=0.7)

    acc.append(get_acc(X_test,Y_test,w,b))

    loss.append(l)
print(loss[0:20])

plt.plot(loss)

plt.xlabel("Time")

plt.ylabel("Negative of log likelihood")

plt.show()
print(acc[0:20])

plt.plot(acc)

plt.show()
print(w)

print(b)

#for Visualizing the separating boundaries we are going to use these values
plt.figure(0)

plt.scatter(dist_01[:,0],dist_01[:,1],label="class0")

plt.scatter(dist_02[:,0],dist_02[:,1],color="r",marker="^",label="class1")

plt.xlim(-3,8)

plt.ylim(-3,8)

plt.xlabel("x1")

plt.ylabel("x2")

x=np.linspace(-3,8,10)

y= - ( w[0] * x + b ) / w[1]

plt.plot(x,y,color="k")

plt.legend()

plt.show()