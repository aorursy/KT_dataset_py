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
salary = pd.read_csv('/kaggle/input/adult-income-dataset/adult.csv')

salary.head()         
y = salary['income']

X = salary.iloc[:,[0,1,3,5,6,8,9,-2,-3]]
X.head()
import numpy as np

import pandas as pd

from pandas import Series,DataFrame

from sklearn.neighbors import KNeighborsClassifier
cancer = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

cancer.shape
cancer.columns
cancer.head()
X = cancer.iloc[:,2:-1]



y = cancer.diagnosis
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train.shape
X_test.shape
knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train,y_train)



y_ = knn.predict(X_test)

display(y_,y_test.values)
knn.score(X_test,y_test)
accuracy_score(y_test,y_)
pd.crosstab(index=y_test,columns=y_,rownames=['确诊'],colnames=['预测'],margins=True)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_)
from sklearn.metrics import classification_report
report = classification_report(y_test,y_)

print(report)
# precision

69/75
# recall

69/74
knn = KNeighborsClassifier(n_neighbors=10,weights='distance')



knn.fit(X_train,y_train)



y_ = knn.predict(X_test)

accuracy_score(y_test,y_)
X.head()
from sklearn.preprocessing import StandardScaler
s = StandardScaler()

X2 = s.fit_transform(X)

X2[:5]
# Not normalized

score = 0

for i in range(100):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=10)



    knn.fit(X_train,y_train)



    y_ = knn.predict(X_test)

    score += accuracy_score(y_test,y_)

print(score)
# normalized

score = 0

for i in range(100):

    X_train,X_test,y_train,y_test = train_test_split(X2,y,test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=10)



    knn.fit(X_train,y_train)



    y_ = knn.predict(X_test)

    score += accuracy_score(y_test,y_)

print(score)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)



knn.fit(X_train,y_train)



# training set(higher)

display(knn.score(X_train,y_train))



# test set

display(knn.score(X_test,y_test))
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# the relationship between X and y is linear

y = np.array([1,3,5,7,9])



X = np.array([0,1,2,3,4])



plt.plot(X,y)

# f = lambda x: 2*x + 1
A = np.random.randint(0,10,size=(3,2))

A
# transpose

A.T
# x + y + z = 3

# 2x + y - z = 3

# 3x - y + 4z = 6 

X = np.array([[1,1,1],[2,1,-1],[3,-1,4]])

X

# w = [X,y,z]
a = np.random.randint(0,30,size=(4,5))

a
a.dot(a.T)
np.linalg.inv(a.dot(a.T))
import sklearn.datasets as datasets
# House prices in Boston that have a lot to do with it

data = datasets.load_boston()

data
X = data['data']

X.shape
y = data['target']

y.shape
data['feature_names']
y
# regression problem --------> apropriate equation: Xw=y

# predict

# In reality, There must be a relationship between the data and the target value, not necessarily a linear one

# simplify complex problems, consider relation between X and y to be linear

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train.shape
lr = LinearRegression(fit_intercept=False)



lr.fit(X_train, y_train)
# provide reference

lr.predict(X_test).round(2)
y_test
np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
w_ = lr.coef_

w_
# bias

b_ = lr.intercept_

b_
lr.predict(X_test).round(2)
# this is the equation that the algorithm came up with

# f(x) = Xw_ + b_

# predict X_test

X_test.dot(w_).round(2)
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
f = lambda x:(x-4.3)**2 + 7.5*x +6
# derivation ------> min

g = lambda x:2*x-1.1
x = np.linspace(-5,9,100)



plt.plot(x,f(x))
result = []

# starting value

v_min = np.random.randint(-10,10,size=1)[0]



# This value records the value of the previous step in the gradient descent

v_min_last = v_min + 1



result.append(v_min)

# when to stop

precision = 0.0001



# another exit criteria, the cycle index

max_time = 3000



# step

step = 0.01

count = 0

# print('-------------------------random min:', v_min)

while True:

    if np.abs(v_min - v_min_last) < precision:

        break

    if count > max_time:

        break

    v_min_last = v_min

    

    v_min = v_min - g(v_min)*step

    result.append(v_min)

#     print('--------------------->gradient update value:',v_min)

    count += 1
plt.figure(figsize=(12,9))

x = np.linspace(-8,5,100)

plt.plot(x,f(x))



# minimum point

plt.scatter(result,f(np.array(result)),marker='*',color='red')
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LinearRegression
X = np.linspace(2.5,12,25)



w = np.random.randint(2,10,size=1)[0]



b= np.random.randint(-5,5,size=1)[0] 



y = X*w +b + np.random.randn(25)*2
plt.scatter(X,y)
lr = LinearRegression()



lr.fit(X.reshape(-1,1),y)
lr.coef_
lr.intercept_
w
b
# if we want the scope,LinearRegression also use gradient descent
for xi,yi in zip(X,y):

    print(xi,yi)
# the object of Class is to obtain slope and intercept

class Linear_model(object):

    def __init__(self):

        self.w = np.random.randn(1)[0]

        self.b = np.random.randn(1)[0]

#         print('---------starting-------->',self.w,self.b)

    

    # model -------> equation -------> f(x)=wx+b

    def model(self,x):

        return self.w*x+self.b

    

    # The principle of linear problems is the least square method

    def loss(self,x,y):

        # how many unknowns

        cost = (y-self.model(x))**2

        

        # partial derivatives

        # derivative is a special form of p-d

        g_w = 2*(y-self.model(x))*(-x)

        g_b = 2*(y-self.model(x))*(-1)

        

        return g_w,g_b

    

    # gradient descent

    def gradient_descent(self,g_w,g_b,step=0.01):

        # update new scope and intercept

        self.w = self.w - g_w*step

        self.b = self.b - g_b*step

#         print('--------------------->',self.w,self.b)

    

    def fit(self,X,y):

        

        w_last = self.w + 1

        b_last = self.b + 1

        

        precision = 0.00001

        max_count = 3000

        count = 0

        while True:

            if (np.abs(self.w-w_last) < precision) and (np.abs(self.b-b_last) < precision):

                break

            if count > max_count:

                break

            

            g_w=0

            g_b=0

            size = X.shape[0]

            # update new

            for xi,yj in zip(X,y):

                g_w += self.loss(xi,yj)[0]/size

                g_b += self.loss(xi,yj)[1]/size

            

            w_last = self.w

            b_last = self.b

            self.gradient_descent(g_w,g_b)

            count += 1

        

    def coef_(self):

            return self.w

        

    def intercept_(self):

            return self.b
lm = Linear_model()
lm.fit(X,y)
lm.coef_()
lm.intercept_()
import numpy as np



from sklearn.linear_model import LinearRegression



import sklearn.datasets as datasets
diabetes = datasets.load_diabetes()

diabetes
X = diabetes['data']



y = diabetes['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) 
X_train.shape
X_train[:5]
# The data is positive and negative, indicating that the data has been processed

# normalization

X_train.std(axis=0)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_ = lr.predict(X_test)

y_.round(2)
y_test
# r^2 the bigger the better

lr.score(X_test,y_test)
u = ((y_test - y_)**2).sum()

u
v = ((y_test - y_test.mean())**2).sum()

v
1 - u/v
1- np.var(y_test-y_)/np.var(y_test)
np.abs(y_test-y_).mean()
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error
mean_absolute_error(y_test,y_)
import numpy as np
X = np.array([[1,1,1],[2,1,1]])

X
np.linalg.inv(X)
np.linalg.matrix_rank(X)
X2 = np.array([[1,1,1],[2,1,1],[2,-1,1]])

np.linalg.matrix_rank(X2)
X3 = np.array([[1,1,1],[2,1,1],[2,2,2]])

np.linalg.matrix_rank(X3)
I = np.eye(3)

I
X4 = X3 + 0.01 * I

X4
np.linalg.matrix_rank(X4)
import numpy as np

from sklearn.linear_model import LinearRegression,Ridge



import sklearn.datasets as datasets
boston = datasets.load_boston()
X = boston['data']

y = boston['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
lr = LinearRegression()



lr.fit(X_train,y_train)



display(lr.coef_,lr.score(X_test,y_test))
lr.predict(X_test).round(2)
# precision

# tol tolerate

ridge = Ridge(alpha=100,tol=0.00001)



ridge.fit(X_train,y_train)



display(ridge.coef_,ridge.score(X_test,y_test))
ridge.predict(X_test).round(2)
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LinearRegression,Ridge,Lasso
# generate datasets

# Size: sample(50) < characteristic(200) 



X = np.random.randn(50,200)

X
# Xw = y

w = np.random.randn(200)

# randomly choose 190 to be 0

w
index = np.arange(200)

np.random.shuffle(index)

index
w[index[:190]] = 0

w
X.shape
y = X.dot(w)

y
lr = LinearRegression(fit_intercept=False)



ridge = Ridge(alpha=1,fit_intercept=False)



lasso = Lasso(alpha=0.1,fit_intercept=False)



lr.fit(X,y)

ridge.fit(X,y)

lasso.fit(X,y)



lr_w = lr.coef_

ridge_w = ridge.coef_

lasso_w = lasso.coef_





# Four subviews

plt.figure(figsize=(12,9))



ax = plt.subplot(2,2,1)

ax.plot(w)

ax.set_title('True')



ax = plt.subplot(2,2,2)

ax.plot(lr_w)

ax.set_title('Lr')



ax = plt.subplot(2,2,3)

ax.plot(ridge_w)

ax.set_title('Ridge')



ax = plt.subplot(2,2,4)

ax.plot(lasso_w)

ax.set_title('Lasso')
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LinearRegression,Ridge,Lasso



from sklearn.neighbors import KNeighborsRegressor



import sklearn.datasets as datasets
data = np.load('/kaggle/input/fetchfaces/images.npy')

data = data.reshape(400,64,64)
data.shape
index = np.random.randint(400,size=1)[0]



plt.imshow(data[index],cmap=plt.cm.gray)
# The top half of the face

X = data[:,:32].reshape(400,-1)



# The lower half of the face

y = data[:,32:].reshape(400,-1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=10)
X_train.shape
y_train.shape
X_test.shape
index = np.random.randint(400,size=1)[0]



face_up = X_train[index].reshape(32,64)

face_down = y_train[index].reshape(32,64)



ax = plt.subplot(131)

ax.imshow(face_up,cmap=plt.cm.gray)



ax = plt.subplot(132)

ax.imshow(face_down,cmap=plt.cm.gray)



ax = plt.subplot(133)

ax.imshow(np.concatenate([face_up,face_down],axis=0),cmap='gray')
estimators = {}



estimators['KNN'] = KNeighborsRegressor(n_neighbors=5)

estimators['LR'] = LinearRegression()

estimators['Ridge'] = Ridge(alpha=1)

estimators['Lasso'] = Lasso(alpha=1)
predict_ = {}

for key,model in estimators.items():

#     print(key,model)

    model.fit(X_train,y_train)

    

#     predict up----->down

    y_ = model.predict(X_test)

    predict_[key] = y_
for i,key in enumerate(predict_):

    print(i,key)
# visualization



# 10rows,6columns

plt.figure(figsize=(6*2,10*2))



for i in range(10):

#     first column

    ax = plt.subplot(10,6,1+i*6)

    face_up = X_test[i].reshape(32,64)

    face_down = y_test[i].reshape(32,64)

    ax.imshow(np.concatenate([face_up,face_down],axis=0),cmap='gray')

    ax.axis('off')

    if i ==0:

        ax.set_title('True')

    

#     second column

    ax = plt.subplot(10,6,2+i*6)

    ax.imshow(face_up,cmap='gray')

    ax.axis('off')

    if i ==0:

        ax.set_title('Face_up')

    

#     third column 3 + i * 6

    for j,key in enumerate(predict_):

        ax = plt.subplot(10,6,3+j+i*6)

        

        y_ = predict_[key]

        

        face_down = y_[i].reshape(32,64)

        

        ax.imshow(np.concatenate([face_up,face_down],axis=0),cmap='gray')

        ax.axis('off')

        if i ==0:

            ax.set_title(key)
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import sklearn.datasets as datasets
iris = datasets.load_iris()

iris
X = iris['data']



y = iris['target']
"""

four attrs:

sepal length (cm)

sepal width (cm)

petal length (cm)

petal width (cm)

"""

X.shape
X_train,X_test,y_train,y_test = train_test_split(X,y)
lg = LogisticRegression()



lg.fit(X_train,y_train)
y_ = lg.predict(X_test)
display(y_,y_test)
lg.score(X_test,y_test)
lg.predict_proba(X_test)
np.bincount(y_train)
X_train.shape
lg.coef_
lg.intercept_
cond = y!=2

X = X[cond]

y = y[cond]
y
lg.fit(X,y)
lg.coef_
lg.intercept_