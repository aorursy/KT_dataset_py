# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt
data = pd.read_csv("../input/"+os.listdir("../input")[0])
data.head()

import matplotlib.pyplot as plt

data.target.value_counts()

data[['age','target']].hist()



plt.show()

plt.scatter(x=data.age[data.target==1], y=data.thalach[data.target==1], c='r')

plt.scatter(x=data.age[data.target==0], y=data.thalach[data.target==0], c='g')



plt.show()
#knn classifier



X =np.array( data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']])

Y = np.array(data[['target']])



#normalizing the data 



for i in range(0,np.shape(X)[1]):

    X[:,i]= (X[:,i]-np.average(X[:,i]))/np.std(X[:,i])





import torch

import torchvision

import torch.nn as nn

import torchvision.transforms as transforms

import numpy as np

import torch.optim as optim

import torch.nn.functional as F

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,shuffle= True)



from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(X_train, y_train) 

accuracy_kkn = neigh.score(X_test, y_test)

accuracy_kkn_train = neigh.score(X_train, y_train)



Z = neigh.predict(X_test)



print ('The train accuracy obtained is: {0:1.2f}'.format(accuracy_kkn_train))

print ('The test accuracy obtained is: {0:1.2f}'.format(accuracy_kkn))



n=[]

for i in range(0,15):

    neigh = KNeighborsClassifier(n_neighbors=i+1)

    neigh.fit(X_train, y_train) 

    p = neigh.score(X_test, y_test)

    n.append(p)

    

    

plt.plot(range (0,np.size(n)),n)

plt.xlabel('Number of nearest neighbours ')

plt.ylabel('Accuracy in test data')

plt.show()
#Logistic regression







from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression().fit(X_train, y_train)

y_prediction = logistic.predict(X_test)



print (' The accuracy by logistic regression is: ')

accuracy_logistic = logistic.score(X_test, y_test)

accuracy_logistic_train = logistic.score(X_train, y_train)



print ('The train accuracy obtained is: {0:1.2f}'.format(accuracy_logistic_train))

print ('The test accuracy obtained is: {0:1.2f}'.format(accuracy_logistic))
import torch

import torchvision

import torch.nn as nn

import torchvision.transforms as transforms

import numpy as np

import torch.optim as optim

import torch.nn.functional as F

from torch.autograd import Variable









train_data = torch.FloatTensor(np.array(X_train))

train_y = np.array(y_train)



train_y =  torch.FloatTensor(train_y)

x,y = Variable(train_data), Variable (train_y)





#######Neural network





n_in, n_h1,n_h2, n_out, batch_size = 13, 13,13, 1, 20







# Creating a model with 2 hidden layer

model = nn.Sequential(nn.Linear(n_in, n_h1, bias = True), nn.ReLU(),nn.Linear(n_h1, n_h2, bias =True),

                     nn.ReLU(),nn.Linear(n_h2, n_out), nn.ReLU())





criterion = torch.nn.MSELoss()



# Construct the optimizer (Stochastic Gradient Descent in this case)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# Gradient Descent

for epoch in range(10000):

    # Forward pass: Compute predicted y by passing x to the model

    y_pred = model(x)



    # Compute and print loss

    loss = criterion(y_pred, y)

    if epoch%1000 == 0:

        print('epoch: ', epoch,' loss: ', loss.item())



    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()

    

    # perform a backward pass (backpropagation)

    loss.backward()

    

    # Update the parameters

    optimizer.step()

test_data = torch.FloatTensor(np.array(X_test))



y_pred_test = model(test_data)

y1_pred = y_pred_test.detach().numpy()



p1= (y1_pred>0.5)





print ('The test accuracy obtained is {0:1.2f}'.format(1-(np.sum(y_test-p1)/np.size (p1))))
