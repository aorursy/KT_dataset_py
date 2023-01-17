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
from sklearn.linear_model import LogisticRegression
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_res = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_test.head()
df_train.head()
df_train = df_train.drop(['Ticket','Name','PassengerId','Cabin'],axis=1)
df_test = df_test.drop(['Ticket','Name','Cabin'],axis=1)
df_train.head()

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
df_train['Embarked'] = df_train['Embarked'].map(embarked_mapping)
df_test['Embarked'] = df_test['Embarked'].map(embarked_mapping)
#on analysing the data , it can be infered that the two passangers which have NA as embarked, belonging to class 1, paid a fare similar to thoes belonging to class 1 embarking from 'C'.Hence replacing NA by 2.
df_train = df_train.fillna({"Embarked": 2})
map1 = {"male": 0, "female": 1}
df_train['Sex'] = df_train['Sex'].map(map1)
df_test['Sex'] = df_test['Sex'].map(map1)
df_train.head()
df_train['Age'].fillna(df_train['Age'].median(), inplace = True)
df_train['Fare'].fillna(df_train['Fare'].median(), inplace = True)

df_test['Age'].fillna(df_test['Age'].median(), inplace = True)
df_test['Fare'].fillna(df_test['Fare'].median(), inplace = True)
df_train.loc[ df_train['Age'] <= 16, 'Age'] = 0
df_train.loc[ (df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1
df_train.loc[ (df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2
df_train.loc[ (df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3
df_train.loc[ df_train['Age'] > 64, 'Age'] = 4
df_test.loc[ df_test['Age'] <= 16, 'Age'] = 0
df_test.loc[ (df_test['Age'] > 16) & (df_test['Age'] <= 32), 'Age'] = 1
df_test.loc[ (df_test['Age'] > 32) & (df_test['Age'] <= 48), 'Age'] = 2
df_test.loc[ (df_test['Age'] > 48) & (df_test['Age'] <= 64), 'Age'] = 3
df_test.loc[ df_test['Age'] > 64, 'Age'] = 4

df_train.head()
df_train['Survived'].value_counts()
Y = df_train['Survived'] 
X = df_train.drop(['Survived'],axis=1)
X['Fare'] = X['Fare']/max(X['Fare']) 
PassengerId=df_test['PassengerId']
df_test=df_test.drop(['PassengerId'],axis=1)
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X,Y = sm.fit_sample(X,Y.values.ravel()) 
from sklearn.model_selection import train_test_split
X_train,X_dev,Y_train,Y_dev = train_test_split(X,Y,test_size =0.15)
Y_dev=pd.DataFrame(Y_dev)
X_train
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.models import Sequential
classifier = Sequential()
classifier.add(Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=7))
classifier.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
history = classifier.fit(X_train,Y_train, batch_size=32, epochs=500,validation_data=(X_dev,Y_dev))
import matplotlib.pyplot as plt
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,500)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
predictions = rf.predict(df_test)

output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
output.to_csv('my_submission_random.csv', index=False)
print("Submission was successfully saved!")
df_test.head()
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
import torch
torch.cuda.is_available()
import torch
def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))
weights = weights.reshape(5,1)
z=torch.mm(features,weights)
z.size()
z= sum(z,bias)
z
activation(z)
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
H1 = activation(sum(torch.mm(features,W1),B1))
H2 = activation(sum(torch.mm(H1,W2),B2))
H2
