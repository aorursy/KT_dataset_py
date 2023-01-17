import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras import optimizers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from numpy.random import seed

seed(2019)

from tensorflow import set_random_seed

set_random_seed(2019)



from matplotlib import pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## Data Matrix

df_XOR_X = np.array([

    [0, 0, 0, 0],

    [1, 0, 0, 0],

    [0, 1, 0, 0], 

    [0, 0, 1, 0],

    [0, 0, 0, 1],

    [1, 1, 0, 0],

    [1, 0, 1, 0],

    [1, 0, 0, 1],

    [0, 1, 1, 0],

    [0, 1, 0, 1],

    [0, 0, 1, 1],

    [0, 1, 1, 1],

    [1, 0, 1, 1],

    [1, 1, 0, 1],

    [1, 1, 1, 0],

    [1, 1, 1, 1]

])

## Corresponding Target Variable

df_XOR_Y = []

for i in range(len(df_XOR_X)):

    if (sum(df_XOR_X[i]) == 1):

        df_XOR_Y.append([1])

    else:

        df_XOR_Y.append([0])

df_XOR_Y = np.array(df_XOR_Y)
## Build network architecture

model_XOR = Sequential()

nnodes1 = 16

model_XOR.add(Dense(nnodes1, activation = 'relu', input_dim = 4))

model_XOR.add(Dense(1, activation = 'sigmoid'))

## Compile

model_XOR.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

## Fit Model

model_XOR.fit(df_XOR_X, df_XOR_Y, epochs=3000, batch_size=4)
pred_XOR = model_XOR.predict(df_XOR_X)
df_XOR_Yhat = (pred_XOR > 0.5)*1
sum((abs(df_XOR_Yhat - df_XOR_Y)).flatten())
# fix (x1,x2) = (0,0), (1,0), (0,1), (1,1)

df_XOR_00vv = np.zeros((10201,4))

df_XOR_01vv = np.zeros((10201,4))

df_XOR_01vv[:,1] = 1

df_XOR_10vv = np.zeros((10201,4))

df_XOR_10vv[:,0] = 1

df_XOR_11vv = np.zeros((10201,4))

df_XOR_11vv[:,[0,1]] = 1



df_XOR_00vv[:,2] = np.repeat(range(101),101)/100.

df_XOR_01vv[:,2] = np.repeat(range(101),101)/100.

df_XOR_10vv[:,2] = np.repeat(range(101),101)/100.

df_XOR_11vv[:,2] = np.repeat(range(101),101)/100.



df_XOR_00vv[:,3] = np.tile(range(101),101)/100.

df_XOR_01vv[:,3] = np.tile(range(101),101)/100.

df_XOR_10vv[:,3] = np.tile(range(101),101)/100.

df_XOR_11vv[:,3] = np.tile(range(101),101)/100.

        

# Predict

df_Yhat_00vv = model_XOR.predict(df_XOR_00vv)

df_Yhat_01vv = model_XOR.predict(df_XOR_01vv)

df_Yhat_10vv = model_XOR.predict(df_XOR_10vv)

df_Yhat_11vv = model_XOR.predict(df_XOR_11vv)
fig, axes = plt.subplots(2, 2, figsize=(7.5,7.5))

# Plot for (x1,x2) = (0,0)

axes[0,0].scatter(np.repeat(range(101),101)[(df_Yhat_00vv <= 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_00vv <= 0.5).flatten()]/100., color = "pink")

axes[0,0].scatter(np.repeat(range(101),101)[(df_Yhat_00vv > 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_00vv > 0.5).flatten()]/100., color = "lightgreen")

axes[0,0].set_title("(x1,x2) = (0,0)")

axes[0,0].set_xlabel("x3")

axes[0,0].set_ylabel("x4")

# Plot for (x1,x2) = (0,1)

axes[0,1].scatter(np.repeat(range(101),101)[(df_Yhat_01vv <= 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_01vv <= 0.5).flatten()]/100., color = "pink")

axes[0,1].scatter(np.repeat(range(101),101)[(df_Yhat_01vv > 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_01vv > 0.5).flatten()]/100., color = "lightgreen")

axes[0,1].set_title("(x1,x2) = (0,1)")

axes[0,1].set_xlabel("x3")

axes[0,1].set_ylabel("x4")

# Plot for (x1,x2) = (1,0)

axes[1,0].scatter(np.repeat(range(101),101)[(df_Yhat_10vv <= 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_10vv <= 0.5).flatten()]/100., color = "pink")

axes[1,0].scatter(np.repeat(range(101),101)[(df_Yhat_10vv > 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_10vv > 0.5).flatten()]/100., color = "lightgreen")

axes[1,0].set_title("(x1,x2) = (1,0)")

axes[1,0].set_xlabel("x3")

axes[1,0].set_ylabel("x4")

# Plot for (x1,x2) = (1,1)

axes[1,1].scatter(np.repeat(range(101),101)[(df_Yhat_11vv <= 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_11vv <= 0.5).flatten()]/100., color = "pink")

axes[1,1].scatter(np.repeat(range(101),101)[(df_Yhat_11vv > 0.5).flatten()]/100., np.tile(range(101),101)[(df_Yhat_11vv > 0.5).flatten()]/100., color = "lightgreen")

axes[1,1].set_title("(x1,x2) = (1,1)")

axes[1,1].set_xlabel("x3")

axes[1,1].set_ylabel("x4")



plt.tight_layout()

plt.show()
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
iris_features = iris['feature_names']

iris_features
fig, axes = plt.subplots(1,4, figsize=(15,5))

for i in range(4):

    axes[i].boxplot([iris['data'][iris['target']==0,i], iris['data'][iris['target']==1,i], iris['data'][iris['target']==2,i]], labels = iris['target_names'])

    axes[i].set_title(iris_features[i])

plt.tight_layout()

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(10,10))

index = [[[0,1],[0,2]],[[0,3],[1,2]],[[1,3],[2,3]]]



for i in range(3):

    for j in range(2):

        axes[i,j].scatter(iris['data'][iris['target']==0,index[i][j][0]],iris['data'][iris['target']==0,index[i][j][1]], color = 'red')

        axes[i,j].scatter(iris['data'][iris['target']==1,index[i][j][0]],iris['data'][iris['target']==1,index[i][j][1]], color = 'orange')

        axes[i,j].scatter(iris['data'][iris['target']==2,index[i][j][0]],iris['data'][iris['target']==2,index[i][j][1]], color = 'green')

        axes[i,j].set_title(iris_features[index[i][j][1]] + " vs " + iris_features[index[i][j][0]])

        axes[i,j].set_xlabel(iris_features[index[i][j][0]])

        axes[i,j].set_ylabel(iris_features[index[i][j][1]])



plt.tight_layout()

plt.show()
scaler = StandardScaler()

df_iris_X = scaler.fit_transform(iris['data'])

df_iris_X = np.concatenate((df_iris_X, np.array([df_iris_X[:,2]*df_iris_X[:,3]]).T), axis=1)



df_iris_X_train, df_iris_X_test, df_iris_Y_train, df_iris_Y_test  = train_test_split(df_iris_X, np.array([iris['target']]).T, test_size=1/3, random_state=2019)

target_names = iris['target_names']
## Build network architecture

model_iris = Sequential()

nnodes1 = 50

model_iris.add(Dense(nnodes1, activation = 'relu', input_dim = 5))

model_iris.add(Dense(3, activation = 'softmax'))

## Compile

#rmsprop = optimizers.RMSprop(lr=0.0008)

model_iris.compile(optimizer="adam",

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

## Fit Model

model_iris.fit(df_iris_X_train, df_iris_Y_train, epochs=3000, batch_size=32)
pred_iris_train = model_iris.predict(df_iris_X_train)

pred_iris_test = model_iris.predict(df_iris_X_test)
df_iris_Yhat_train = np.array([np.argmax(vec) for vec in pred_iris_train])

df_iris_Yhat_test = np.array([np.argmax(vec) for vec in pred_iris_test])
df_iris_Yhat_train
df_iris_Y_train.flatten()
## Number of wrong predictions for training set

sum([(df_iris_Yhat_train[i] - df_iris_Y_train.flatten()[i] != 0) for i in range(len(df_iris_Yhat_train))])
df_iris_Yhat_test
df_iris_Y_test.flatten()
## Number of wrong predictions for test set

sum([(df_iris_Yhat_test[i] - df_iris_Y_test.flatten()[i] != 0) for i in range(len(df_iris_Yhat_test))])
wrong = np.where([(df_iris_Yhat_test[i] - df_iris_Y_test.flatten()[i] != 0) for i in range(len(df_iris_Yhat_test))])[0]

wrong_c = ['red', 'orange', 'green']



fig, axes = plt.subplots(3, 2, figsize=(10,10))

index = [[[0,1],[0,2]],[[0,3],[1,2]],[[1,3],[2,3]]]



for i in range(3):

    for j in range(2):

        axes[i,j].scatter(df_iris_X[iris['target']==0,index[i][j][0]],df_iris_X[iris['target']==0,index[i][j][1]], color = 'red', alpha = 0.2)

        axes[i,j].scatter(df_iris_X[iris['target']==1,index[i][j][0]],df_iris_X[iris['target']==1,index[i][j][1]], color = 'orange', alpha = 0.2)

        axes[i,j].scatter(df_iris_X[iris['target']==2,index[i][j][0]],df_iris_X[iris['target']==2,index[i][j][1]], color = 'green', alpha = 0.2)

        

        for k in range(len(wrong)):

            axes[i,j].scatter(df_iris_X_test[wrong[k],index[i][j][0]],df_iris_X_test[wrong[k],index[i][j][1]], color = wrong_c[df_iris_Yhat_test[wrong[k]]], marker = 'x', s = 100)

        

        axes[i,j].set_title(iris_features[index[i][j][1]] + " vs " + iris_features[index[i][j][0]])

        axes[i,j].set_xlabel(iris_features[index[i][j][0]])

        axes[i,j].set_ylabel(iris_features[index[i][j][1]])



plt.tight_layout()

plt.show()