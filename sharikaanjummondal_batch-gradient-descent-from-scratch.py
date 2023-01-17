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
df=pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
df.head()
df.shape
df.drop(columns=['User ID'], inplace=True)
df.shape
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
df['Gender']= encoder.fit_transform(df['Gender'])
df.head()
X=df.iloc[:,:-1].values

y=df.iloc[:,-1].values
X
y
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X=scaler.fit_transform(X)
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
import tensorflow

from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(1,activation='sigmoid',input_dim=X_train.shape[1]))
model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=36,epochs=100,verbose=1)
model.evaluate(X_test,y_test)
model.get_weights()
#initializing my weights and bias



weights=np.ones(X_train.shape[1])

bias = 0
print(weights)

print(bias)
def sigmoid(z):

    

    return 1/(1+np.exp(-z))
y_hat = sigmoid(np.dot(X_train,weights)+bias)
y_hat
y_hat.shape
#finding the loss for every point



loss = np.mean(-y_train*np.log(y_hat)- (1-y_train)*np.log(1-y_hat))
loss
lr = 0.1 #learning rate
w_derivative = -(np.dot((y_train - y_hat),X_train))/X_train.shape[0] 
w_derivative
b_derivative = -np.mean(y_train - y_hat)
b_derivative
weights = weights-(lr*w_derivative)

bias = bias - (lr*b_derivative)
weights 
bias
for i in range(100):

    y_hat = sigmoid(np.dot(X_train,weights)+bias)

    w_derivative = -(np.dot((y_train - y_hat),X_train))/X_train.shape[0] 

    b_derivative = -np.mean(y_train - y_hat)

    weights = weights-(lr*w_derivative)

    bias = bias - (lr*b_derivative)

    y_hat = sigmoid(np.dot(X_train,weights)+bias)

    loss = np.mean(-y_train*np.log(y_hat)- (1-y_train)*np.log(1-y_hat))

    print("After {} epoch the loss is".format(i+1),loss)
#finding the final weights and bias of ours



print(bias)

print(weights)