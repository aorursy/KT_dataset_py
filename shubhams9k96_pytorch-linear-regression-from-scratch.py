import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import torch

import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Advertising.csv')

df.info()
# removing the inbuilt index column

df.drop('Unnamed: 0', axis = 1, inplace=True)

df.head()
sns.pairplot(df)

plt.show()
x = df.drop('Sales', axis =1).values

y = df[['Sales']].values
# n = 200, d = 3

print(x.shape)

print(y.shape)
# Converting the numpy array features to pytorch tensors. 

inputs = torch.from_numpy(x)

targets = torch.from_numpy(y)
print(inputs.shape)

print(targets.shape)
w = torch.randn(1,3, requires_grad=True, dtype = torch.double)

b = torch.randn(1, requires_grad=True,  dtype = torch.double)

print(w,b)
'''Function to apply y = X.WT + b'''

def model(x):

    return torch.mm(x, w.t()) + b



'''Function to find the mean squared error between predicted and actual labels.  '''



def mse(t1, t2):

    diff = t1-t2

    return torch.sum(diff*diff)/diff.numel()
epochs = 1000

lr = 1e-5

loss_tr = []

for i in range(epochs):

    preds = model(inputs)    

    loss = mse(preds, targets)

    loss.backward()

    

    with torch.no_grad():

        w -= w.grad * lr

        b -= b.grad * lr

        w.grad.zero_()

        b.grad.zero_() 

    if i%100==0:

        print('[{:3}/{}] Loss : {:3.3f}'.format(i, epochs, loss.item()))

    loss_tr.append(loss.item())
print('Final Loss :', min(loss_tr))

print('Weights    :' , w)

print('Bias       :' , b)
sns.set(style='darkgrid')

plt.plot(loss_tr)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss per epoch')

plt.show()
y_preds = model(inputs)
sns.residplot(y_preds.data.numpy(), targets.data.numpy())

plt.title("Residuals")

plt.show()
plt.figure(figsize=(10,15))



plt.subplot(311)

sns.regplot(x="TV", y="Sales", data=df, label = 'Actual')

plt.scatter( df[['TV']].values, y_preds.data.numpy(),color = 'r', marker = '+', label='Predicted')

plt.legend()



plt.subplot(312)

sns.regplot(x="Radio", y="Sales", data=df, label = 'Actual')

plt.scatter( df[['Radio']].values, y_preds.data.numpy(),color = 'r', marker = '+', label='Predicted')

plt.legend()



plt.subplot(313)

sns.regplot(x="Newspaper", y="Sales", data=df, label = 'Actual')

plt.scatter( df[['Newspaper']].values, y_preds.data.numpy(),color = 'r', marker = '+', label='Predicted')

plt.legend()



plt.show()
