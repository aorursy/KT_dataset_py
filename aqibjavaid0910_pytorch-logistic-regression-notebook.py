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
# Set your own project id here
PROJECT_ID = 'kaggle_notebooks'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Loading the Breast Cancer Dataset and distributing it across (X and y)
dataset = datasets.load_breast_cancer()
X,y = dataset.data, dataset.target
n_samples, n_features = X.shape
print(n_samples,'  ', n_features)
# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12)
print(X_train.shape[0], X_test.shape[0])
# Applying Standardization on the Training Data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
# Converting Numpy Arrays to Torch Tensors
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_test_tensor  = torch.from_numpy(X_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))
# Changing the dimension of the 'y_train_tensor' and 'y_test_tensor'
y_train_tensor = y_train_tensor.view(y_train_tensor.shape[0],1)
y_test_tensor = y_test_tensor.view(y_test_tensor.shape[0],1)
y_train_tensor.shape
# Knowing the unique classes of the breast_cancer dataset
set(y_train)
# Setting input and output attributes of the network
input_size  = X_train_tensor.shape[1]
output_size =  1

# Defining custom LR class
class LogisticRegression(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        
        self.lin1 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.lin1(x))
        return y_pred
model = LogisticRegression(input_size, output_size)
lr = 0.01
epochs = 200
# Defining Loss and Optimizer function for the LR model
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
# Training Loop for the Logistic Regression Model
for i in range(epochs):
    
    # forward pass
    y_preds = model.forward(X_train_tensor)
    
    # loss function
    loss = criterion(y_preds, y_train_tensor)
    
    # Backward pass
    loss.backward()
    
    # updation of parameters
    optimizer.step()
    
    # Zeroing parameters gradients
    optimizer.zero_grad()
    
    if (i+1) % 10 == 0:
        print(f'Epochs : {i+1}, Loss : {loss:.4f}')

# Detaching the tensor from the Computational Graph 
with torch.no_grad():
    y_test_preds = model.forward(X_test_tensor)
    
    # Rounding the probabality of the class to predict the class either (0 or 1) 
    y_test_preds_cls = y_test_preds.round()
    
    # Finding the Accuracy of the Model
    acc = y_test_preds_cls.eq(y_test_tensor).sum() / float(y_test_tensor.shape[0])
    print(f'Accuracy of a LR model is : {acc:.4f}')
! pip install jovian
import jovian
jovian.commit(project = 'pytorch-logistic-regression-notebook')
