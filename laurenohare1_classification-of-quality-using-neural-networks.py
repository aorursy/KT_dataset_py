import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, scale

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# get the directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data["quality"].unique() #printed in the order they appear in the dataset
sns.countplot(x = "quality", data = data)
plt.title("Number of observations per category of wine")
plt.show()
# Returns an array of the text files
# Skipping the first row which is the column names

wine = np.loadtxt("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv", delimiter= ",", skiprows= 1)

# These options determine the way floating point numbers, arrays and other NumPy objects are displayed.

# Number of digits of precision for floating point output
np.set_printoptions(precision = 2)

# If True, always print floating point numbers using fixed point notation, 
# in which case numbers equal to zero in the current precision will print as zero.
np.set_printoptions(suppress = True)

print("# of instances:", len(wine))

# print the first five rows
print(wine[:5])

# print the last five rows
print(wine[-5:])
# define the test percentage = 20%
test_per = 0.2

# number of input features
n_features = 11

# X is all the rows in the dataset and the first 11 columns, ie the 11 input features
X = wine[:,:n_features]

# Standardize the input features
X = scale(X)

# Now we have X - a scaled dataset with all the features

# label is all the rows but only from the 11th column. 
# Ie this is just the target values
label = wine[:, n_features:]

# Encode categorical features as a one-hot numeric array.
oneHot = OneHotEncoder()

# Fit OneHotEncoder to X, then transform X. Then to an array
label = oneHot.fit_transform(label).toarray()

# Split the data for testing and training
# Using X which are the features
# Using label which is the oneHot encoded targets
X_train , X_test , Y_train , Y_test = train_test_split(X , label, test_size = test_per)

# check the lengths of the data 
print(len(X_train))
print(len(X_test))
from IPython.display import Image
Image("/kaggle/input/fcnimage/FCN model.JPG")
class FCN(nn.Module):
    def __init__(self):
        super(FCN , self).__init__()
        
        # Input to 11 features (n_features defined as 11 above) to 11 hidden nodes. 
        self.layer1 = nn.Linear(n_features , n_features)
        
        # 11 hidden nodes to 6 output classes.
        self.layer2 = nn.Linear(n_features , 6)  # 6 output classes
        
        # Forward pass
    def forward(self , data):
        
        # apply layer one to input 
        activation1 = self.layer1(data)
        
        # sigmoid activation on the first layer
        activation1 = torch.sigmoid(activation1)
        
        # layer two 
        activation2 = self.layer2(activation1)
        
        # return the output activation on the sigmoid
        return torch.sigmoid(activation2)
# define the model instance
model = FCN()

# define the criterion 
# Creates a criterion that measures the Binary Cross Entropy between the target and the output
criterion = nn.BCELoss()

# gradient decent
optimizer = optim.SGD(model.parameters() , lr = 0.2)

# convert the numpy to tensor
# variable wraps a tensor
X = Variable(torch.from_numpy(X_train).float())
Y = Variable(torch.from_numpy(Y_train).float())

# set n to 0
# create an empty array to hold the losses, actual class, predicted class and training accuracy.
n = 0
losses = []
act_class = []
pred_class = []
train_acc = []

# iterate the data 10,000 times
for epoch in range(10000):
    
    # forward pass
    out = model(X)
    loss = criterion(out , Y)
    losses.append(loss.data)
    
    optimizer.zero_grad()
    
    # back propagation
    loss.backward()
    optimizer.step()
    
    # save the training accuracy to the train_acc array
    train_acc.append(accuracy_score(oneHot.inverse_transform(Y), oneHot.inverse_transform(model(X).data.numpy())))
    
    if epoch % 500 == 0:
        print(epoch, loss.data)
        

# Training Over.

# add the losses to the losses array
losses = np.array(losses , dtype = np.float)

# Convert the data back to the original representation.
train_out = oneHot.inverse_transform(model(X).data.numpy())

print('Training accuracy', accuracy_score(oneHot.inverse_transform(Y), train_out))

test_out=oneHot.inverse_transform(model(torch.from_numpy(X_test).float()).data.numpy())

print('prediction accuracy', accuracy_score(oneHot.inverse_transform(Y_test), test_out))

# gather class results
act_class.append(oneHot.inverse_transform(Y_test))
pred_class.append(test_out)
%matplotlib inline
plt.plot(losses)
plt.title("Loss during training")
plt.show()
plt.plot(train_acc)
plt.title("Training Accuracy")
plt.show()
confusion_matrix(oneHot.inverse_transform(Y_test), test_out)