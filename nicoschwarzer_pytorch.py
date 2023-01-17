
##
## Using Pytorch on self-generated data 
##



import torch
import numpy as np
import pandas as pd

# first data type 
a = []

for i in range(0,999):
    a.append(i + 1)
    

# second data type 

b = []

for i in range(0,999):
    x = i ** 2 - 2 * i
    b.append(x)
    

# third data type 
    
c = []

for i in range(0,999):
    x = i ** 0.5  + 3 * i
    c.append(x)
    

df1 = pd.DataFrame(a)
df1["b"] = b
df1["c"] = c


# fourth data type - to be predicted 

import random

randomlist = []

for i in range(0,999):
    n = random.randint(0,110) * 0.01
    randomlist.append(n)

df1["rand"] = randomlist

df1["d"] = 0.4 * df1[0] - 0.8* df1["b"] + 0.2*df1["c"] + 100 * df1["rand"]


## splitting the data 


from sklearn.model_selection import train_test_split

x = df1.drop(["rand", "d"], axis = 1)

y = pd.DataFrame(df1["d"])



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#x_train.info()

#y_train.info()


def to_torch(x):
    x1 = np.array(x)
    x2 = torch.from_numpy(x1)
    return x2

x_train = to_torch(x_train)
y_train = to_torch(y_train)
x_test = to_torch(x_test)
y_test = to_torch(y_test)

## the model

cols = x_train.shape[1]


model = torch.nn.Sequential(
    torch.nn.Linear(cols, 50),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(50, 1))


loss_MSE = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


l_iterations = []
l_loss = [] 

for i in range(3000):
    y_pred = model(x_train.float())

    loss = loss_MSE(y_pred, y_train.float())
   
        
    l_iterations.append(i)
    l_loss.append(loss)
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


import matplotlib.pyplot as plt

plt.plot(l_iterations, l_loss)


## Evaluating the model

y_pred2 = model(x_test.float())
y_pred2

loss = loss_MSE(y_pred2, y_test)
loss




