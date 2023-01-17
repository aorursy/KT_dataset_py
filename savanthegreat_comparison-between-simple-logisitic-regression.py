# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by t|he kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

df=pd.read_csv("../input/heart.csv")
df.head()
df.index
df.dtypes
df.isnull().sum()
df.describe()
df.head()
fig, ax = plt.subplots(figsize=(5, 8))

sns.countplot(df['target'])

plt.title('Target values')
fig, ax = plt.subplots(figsize=(5, 8))

sns.countplot(df['sex'])

plt.title('male and female distribution')
sns.distplot(df['age'])

sns.jointplot(x = 'age', y = 'oldpeak', kind = 'kde', color = 'red', data = df)

sns.set(font_scale=2)

fig, ax = plt.subplots(figsize=(20, 10))

plt.title('Target values with different cp in male and female')

sns.barplot(x='sex',y='target',hue='cp',data=df)
ages=['age']

bins = [29,35,45,55,65,77]

labels = [ '25-40','41-50', '51-60', '61-70', '70+']

df['agerange'] = pd.cut(df.age, bins, labels = labels,include_lowest = True)

fig, ax = plt.subplots(figsize=(20, 10))

plt.title('maximum heart rate as the agerange with and without excersize induced angina ')

sns.barplot(x='agerange',y='thalach',hue='exang',data=df)
sns.set(font_scale=1)

fig, ax = plt.subplots(figsize=(20, 20))

plt.title('correlation between features')

sns.heatmap(df.corr(), robust=True, fmt='.2f',

                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
fig, ax = plt.subplots(figsize=(20, 10))

plt.title('change in cholestrol with the age in male and female')

sns. barplot(x="agerange", y="chol",hue='sex' ,data=df)

ax1=sns.catplot(x="agerange", y="chol",hue="sex",kind="violin",split=True,data=df)

plt.title('change in cholestrol with the age in men and women')

fig1, ax2 = plt.subplots(figsize=(10, 5))

ax2=sns.boxplot(x="sex", y="chol",data=df)

plt.title('change in cholestrol with the age')

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

X=df.drop(columns=["target","agerange"],axis=1)

y=df['target']
df.head()
test_size=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]

for i in test_size:

  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=i)

  clf=LogisticRegression()

  clf.fit(X_train,y_train)

  clf.predict(X_test)

  accuracy=clf.score(X_test,y_test)

  print("accuracy for test size",i,"is",accuracy )
from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)

from sklearn.preprocessing import StandardScaler

# scale the train and test ages

df_num=['age', 'trestbps', 'chol','thalach', 'oldpeak']

scaler = StandardScaler()

X_train.age = scaler.fit_transform(X_train.age.values.reshape(-1,1))

X_test.age = scaler.transform(X_test.age.values.reshape(-1,1))

X_train.trestbps = scaler.fit_transform(X_train.trestbps.values.reshape(-1,1))

X_test.trestbps = scaler.transform(X_test.trestbps.values.reshape(-1,1))

X_train.chol = scaler.fit_transform(X_train.chol.values.reshape(-1,1))

X_test.chol = scaler.transform(X_test.chol.values.reshape(-1,1))

X_train.thalach = scaler.fit_transform(X_train.thalach.values.reshape(-1,1))

X_test.thalach = scaler.transform(X_test.thalach.values.reshape(-1,1))

X_train.oldpeak = scaler.fit_transform(X_train.oldpeak.values.reshape(-1,1))

X_test.oldpeak = scaler.transform(X_test.oldpeak.values.reshape(-1,1))

clf=LogisticRegression()

clf.fit(X_train,y_train)

clf.predict(X_test)

accuracy=clf.score(X_test,y_test)

print("accuracy with feauture scaling is",accuracy )

import sklearn

import sklearn.datasets

from sklearn import preprocessing

df=pd.read_csv("../input/heart.csv")

df.columns
X=df.drop(['target'],axis=1)

Y=df.as_matrix(columns=['target'])

shape_X = X.shape

shape_Y = Y.shape

m = X.shape[0]

print ('The shape of X is: ' + str(shape_X))

print ('The shape of Y is: ' + str(shape_Y))

print ('I have m = %d training examples!' % (m))




np.random.seed(42)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

from sklearn.preprocessing import StandardScaler

# scale the train and test ages

df_num=['age', 'trestbps', 'chol','thalach', 'oldpeak']

scaler = StandardScaler()

X_train.age = scaler.fit_transform(X_train.age.values.reshape(-1,1))

X_test.age = scaler.transform(X_test.age.values.reshape(-1,1))

X_train.trestbps = scaler.fit_transform(X_train.trestbps.values.reshape(-1,1))

X_test.trestbps = scaler.transform(X_test.trestbps.values.reshape(-1,1))

X_train.chol = scaler.fit_transform(X_train.chol.values.reshape(-1,1))

X_test.chol = scaler.transform(X_test.chol.values.reshape(-1,1))

X_train.thalach = scaler.fit_transform(X_train.thalach.values.reshape(-1,1))

X_test.thalach = scaler.transform(X_test.thalach.values.reshape(-1,1))

X_train.oldpeak = scaler.fit_transform(X_train.oldpeak.values.reshape(-1,1))

X_test.oldpeak = scaler.transform(X_test.oldpeak.values.reshape(-1,1))

X_train=X_train.T

print(X_train.shape)

Y_train=Y_train.T

print(Y_train.shape)

X_test=X_test.T

print(X_test.shape)

Y_test=Y_test.T

print(Y_test.shape)
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
def sigmoid(z):

    s = 1/(1+np.exp(-z))

    return s
def layer_sizes(X, Y):

    

    n_x = X.shape[0] # size of input layer

    n_h = 4

    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)
def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.

    

    W1 = np.random.randn(n_h,n_x) * 0.01

    b1 = np.zeros((n_h,1))

    W2 = np.random.randn(n_y,n_h) * 0.01

    b2 = np.zeros((n_y,1))

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
def forward_propagation(X, parameters):

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    Z1 = np.dot(W1,X)+b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2,A1)+b2

    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
def compute_cost(A2, Y, parameters):

    m = Y.shape[1] # number of example

    logprobs = (Y*np.log(A2))+((1-Y)*np.log(1-A2))

    cost = (-1/m)*np.sum(logprobs)

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 

    return cost
# GRADED FUNCTION: backward_propagation



def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    W1= parameters["W1"]

    W2 = parameters["W2"]

    A1 = cache["A1"]

    A2 = cache["A2"]

    dZ2 = A2-Y

    dW2 = (1/m)*np.dot(dZ2,A1.T)

    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))

    dW1 = (1/m)*np.dot(dZ1,X.T)

    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW1": dW1,

             "db1": db1,

             "dW2": dW2,

             "db2": db2}

    

    return grads
def update_parameters(parameters, grads, learning_rate = 1.2):

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    dW1 = grads["dW1"]

    db1 = grads["db1"]

    dW2 = grads["dW2"]

    db2 = grads["db2"]

    W1 = W1-(learning_rate*dW1)

    b1 = b1-(learning_rate*db1)

    W2 = W2-(learning_rate*dW2)

    b2 = b2-(learning_rate*db2)

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):

    np.random.seed(3)

    n_x = layer_sizes(X, Y)[0]

    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate = 1.2)

  

        if print_cost and i % 100000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))



    return parameters
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)

    predictions = (A2>0.5)

    

    return predictions
parameters = nn_model(X_train, Y_train, n_h = 4, num_iterations = 1000000 , print_cost=True)

predictions = predict(parameters, X_test)

predictions
print(predictions.shape)

print(Y_test.shape)
predictions = predict(parameters, X_test)

print ('Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-(Y_test),1-predictions.T))/float(((Y_test)).size)*100) + '%')