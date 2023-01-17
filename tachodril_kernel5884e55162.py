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
dfo=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_set=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df=dfo
df.head()

df.shape
X=df.drop(labels='label',axis=1)

X.head()
y=df['label']

y.head()
X.describe()
y=pd.get_dummies(y)

y.head()
ip_layer_size=784

hidden_layer_size=50

op_layer_size=10

lmbda=2
epsilon=0.12

theta1=np.random.rand(50,785)  *2*epsilon-epsilon

theta2=np.random.rand(10,51)  *2*epsilon-epsilon
pd.DataFrame(theta1).head()
#unrolling the params

unrolled_params=np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))
theta1.ravel(order='F').shape

unrolled_params.shape
for i in range(5):

    print(f'jwheb {i}')
def sigmoid(z):

    return 1.0/(1.0+np.exp(-z))
q=np.array([[1],[2],[3]])

pd.DataFrame(q)
def costFunction(unrolled_params, ip_layer_size, hidden_layer_size, op_layer_size, X, y, lmbda):

    param1=np.reshape(unrolled_params[:hidden_layer_size*(1+ip_layer_size)], 

                      (hidden_layer_size, 

                      1+ip_layer_size), 

                      'F')

    param2=np.reshape(unrolled_params[hidden_layer_size*(1+ip_layer_size):],

                      (op_layer_size,  

                      1+hidden_layer_size), 

                      'F')

    

    m=len(y)

    ones=np.ones((m,1))

    a1=np.hstack((ones, X))

    a2=sigmoid(a1 @ param1.T)

    a2=np.hstack((ones, a2))

    h=sigmoid(a2 @ param2.T)

    

    print(a1.dtype)

    print(y.shape)

    print(h.shape)

    temp1=y.T @ np.log(h)

    temp2=(1-y).T @ np.log(1-h)

    temp3=np.sum(temp1+temp2)

    

    sum1=np.sum(np.sum(np.power(param1[:,1:], 2), axis=1))

    sum2=np.sum(np.sum(np.power(param2[:,1:], 2), axis=1))

    

    jcost = np.sum(temp3/(-m)) + (sum1+sum2)*(lmbda)/(2*m)

    

    print(f'testing the cost function ... {jcost}')

    

    return jcost
costFunction(unrolled_params, 

             ip_layer_size, 

             hidden_layer_size, 

             op_layer_size, 

            X, y, lmbda)
def sigmoidGrad(z):

    return np.multiply(sigmoid(z), 1-sigmoid(z))
print(58)

len(X)

X.shape[0]

np.ones(1)

y.iloc[1,:][:,np.newaxis]

print(X.iloc[225])

ones=np.ones(1)

lp=np.hstack((ones, X.iloc[225]))
def costGrad(unrolled_params, ip_layer_size, hidden_layer_size, op_layer_size, X, y, lmbda):

    param1=np.reshape(unrolled_params[:hidden_layer_size*(1+ip_layer_size)], 

                      (hidden_layer_size, 

                      1+ip_layer_size), 

                      'F')

    param2=np.reshape(unrolled_params[hidden_layer_size*(1+ip_layer_size):],

                      (op_layer_size,  

                      1+hidden_layer_size), 

                      'F')

    

    delta1=np.zeros(param1.shape)

    delta2=np.zeros(param2.shape)

    m=len(y)

    print(f'testing the cost gradient that is the differential...1')

    

    for i in range(X.shape[0]):

        ones=np.ones(1)

        a1=np.hstack((ones, X.iloc[i]))

        z2=a1 @ param1.T

        a2=sigmoid(z2)

        a2=np.hstack((ones,a2))

        h=sigmoid(a2 @ param2.T)

        

        d3=h-y.iloc[i,:][np.newaxis,:]

        z2=np.hstack((ones,z2))

        d2=np.multiply(param2.T @ d3.T, sigmoidGrad(z2).T[:, np.newaxis])

        delta1=delta1+d2[1:,:] @ a1[np.newaxis,:]

        delta2=delta2+d3.T @ a2[np.newaxis,:]

        

    delta1/=m

    delta2/=m

    delta1[:,1:]=delta1[:,1:]+param1[:,1:]*lmbda/m

    delta2[:,1:]=delta2[:,1:]+param2[:,1:]*lmbda/m

    

    print(f'testing the cost gradient that is the differential...2')

    

    return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))
costGrad(unrolled_params, 

             ip_layer_size, 

             hidden_layer_size, 

             op_layer_size, 

            X, y, lmbda)
import scipy.optimize as opt
theta_opt=opt.fmin_cg(maxiter=50, f=costFunction, 

                     x0=unrolled_params, fprime=costGrad,

                     args=(ip_layer_size, hidden_layer_size, op_layer_size,

                          X, y, lmbda))



theta1_opt=np.reshape(theta_opt[:hidden_layer_size*(ip_layer_size+1)],

                     (hidden_layer_size, ip_layer_size+1),

                     'F')

theta2_opt=np.reshape(theta_opt[hidden_layer_size*(ip_layer_size+1):],

                     (op_layer_size, hidden_layer_size+1),

                     'F')
theta2_opt.shape
def predict(theta1_opt, theta2_opt, X_test):

    m=len(X_test)

    ones=np.ones((m,1))

    a1=np.hstack((ones,X_test))

    a2=sigmoid(a1 @ theta1_opt.T)

    a2=np.hstack((ones, a2))

    h=sigmoid(a2 @ theta2_opt.T)

    return h
h=predict(theta1_opt, theta2_opt, test_set)

h=pd.DataFrame(h)

hg=h.idxmax(axis=1)
hg.shape
print(hg.head())

hg.shape
op=pd.DataFrame({"ImageId":test_set.index+1, 'Label': hg})

op.head()
op.to_csv('my_submission.csv', index=False)

print("Submission saved !!")