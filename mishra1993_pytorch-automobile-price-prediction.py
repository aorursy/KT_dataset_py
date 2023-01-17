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
automobile_data = pd.read_csv('../input/Automobile_data.csv')
automobile_data.head()
import numpy as np
automobile_data = automobile_data.replace('?', np.nan)
automobile_data.head()
automobile_data = automobile_data.dropna()
automobile_data.head()

col = ['make', 'fuel-type', 'body-style', 'horsepower']
automobile_features = automobile_data[col]
automobile_features.head()
automobile_target = automobile_data[['price']]
automobile_target.head()

automobile_features['horsepower'].describe()
pd.options.mode.chained_assignment = None
automobile_features['horsepower'] = \
                pd.to_numeric(automobile_features['horsepower'])
automobile_features['horsepower'].describe()
automobile_target['price'].describe()
automobile_target = automobile_target.astype(float)
automobile_target['price'].describe()
automobile_features = pd.get_dummies(automobile_features, 
                                     columns= ['make', 'fuel-type', 'body-style'])
automobile_features.head()
automobile_features.columns

from sklearn import preprocessing
automobile_features[['horsepower']] = \
                preprocessing.scale(automobile_features[['horsepower']])
automobile_features[['horsepower']].head()


from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(automobile_features,
                                                    automobile_target,
                                                    test_size=0.2,
                                                    random_state=0)
import torch
dtype = torch.float
X_train_tensor = torch.tensor(X_train.values, dtype = dtype)
x_test_tensor = torch.tensor(x_test.values, dtype = dtype)

Y_train_tensor = torch.tensor(Y_train.values, dtype = dtype)
y_test_tensor = torch.tensor(y_test.values, dtype = dtype)
X_train_tensor.shape
Y_train_tensor.shape
inp = 26
out = 1

hid = 100

loss_fn = torch.nn.MSELoss()

learning_rate = 0.0001
model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(hid, out),
)
for iter in range(10000):
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, Y_train_tensor)

    if iter % 1000 ==0:
        print(iter, loss.item())
    
    model.zero_grad()
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

sample = x_test.iloc[23]
sample
sample_tensor = torch.tensor(sample.values, 
                             dtype = dtype)
sample_tensor
y_pred = model(sample_tensor)
print("Predicted price of automobile is : ", int(y_pred.item()))
print("Actual price of automobile is : ", int(y_test.iloc[23]))

y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()
import matplotlib.pyplot as plt

plt.scatter(y_pred, y_test.values)
plt.xlabel("Actual Price")
plt.ylabel("Predicted price")

plt.title("Predicted prices vs Actual prices")
plt.show()
torch.save(model, 'my_model')
saved_model = torch.load('my_model')

y_pred_tensor = saved_model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()
plt.figure(figsize=(15,6))

plt.plot(y_pred, label='Predicted Price')
plt.plot(y_test.values, label='Actual Price')

plt.legend()
plt.show()

