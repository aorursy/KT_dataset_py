import numpy as np
import pandas as pd
import os
%matplotlib inline
PATH = '../input'
os.listdir(PATH)
train_df = pd.read_csv(f'{PATH}/train.csv', nrows=10000000)
print(train_df.isnull().sum())
print('Old size %d'% len(train_df))
train_df = train_df.dropna(how='any',axis='rows')
print('New size %d' % len(train_df))
train_df[:5]
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
add_travel_vector_features(train_df)
train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
train_df = train_df[(train_df.abs_diff_longitude<5) & (train_df.abs_diff_latitude<5)]
print(len(train_df))
import torch
import torch.nn as nn
from torch.autograd import Variable
model = nn.Sequential(nn.Linear(2, 10),
                     nn.Linear(10, 5),
                      nn.Linear(5, 1))
                    

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

X = np.stack((train_df.abs_diff_latitude.values,train_df.abs_diff_longitude.values)).T
X = torch.from_numpy(X)
X = X.type(torch.FloatTensor)
y = torch.from_numpy(train_df.fare_amount.values.T)
y = y.type(torch.FloatTensor)
y.unsqueeze_(-1)
for epoch in range(90):
    # Forward Propagation
    y_pred = model(X)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for epoch in range(700):
    # Forward Propagation
    y_pred = model(X)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()
    
    # perform a backward pass (backpropagation)
    loss.backward()
    
    # Update the parameters
    optimizer.step()
y[:10]
y_pred[:10]
test_df = pd.read_csv(f'{PATH}/test.csv')
test_df

add_travel_vector_features(test_df)
X_test = np.stack((test_df.abs_diff_latitude.values,test_df.abs_diff_longitude.values)).T
X_test = torch.from_numpy(X_test)
X_test = X_test.type(torch.FloatTensor)
y_test = model(X_test)
y_test[:20]
test_df.key
y_test = y_test.detach().numpy()
y_test = y_test.reshape(-1)
y_test
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': y_test},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
