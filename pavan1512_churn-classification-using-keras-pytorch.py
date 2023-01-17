import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd
# Read the data

data_churn = pd.read_csv('/kaggle/input/churndataset/Churn.csv')
data_churn.head(2)
# Check data types and missing values

data_churn.info()
data_churn.shape
# for detailed report

# pandas_profiling.ProfileReport(data_churn)
# Drop the columns which are not useful

data_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)
# Create dummies for columns 

df = pd.get_dummies(data_churn, ['Geography', 'Gender'], drop_first = True)
df.head(2)
df.shape
feature_columns = df.columns.difference(['Exited'])

feature_columns
# Separate X variables and Y variable

X = df[feature_columns]

y = df['Exited']
X.shape
y.shape
# Split the data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 111)
# Apply scaling transformation to converge quickly

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test  = sc.transform(X_test)
X_train.shape
X_train
X_test
# Import the necessary libraries

import keras



from keras.models import Sequential

from keras.layers import Dense
# Create the model with network [11, 6, 6, 1]



model = Sequential()
# Model with network (11, 6, 6, 1)

# Input layers = 11 inputs, two hidden layers with 6 neurons each and one output layer



# Input layer and Hidden layer1

model.add(Dense(6, activation = 'relu', input_dim = 11))



# Hidden layer2

model.add(Dense(6, activation = 'relu'))



# Output layer

model.add(Dense(1, activation = 'sigmoid'))

# Above, we have defined the model



# Now lets compile the model with loss function/Optimizer

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# model.fit?
# fit the model

model.fit(X_train, y_train, batch_size = 10, epochs = 20)
# predict the test data

y_test_pred = model.predict(X_test)

y_test_pred
# calculate the score

import sklearn.metrics as metrics



metrics.roc_auc_score(y_test, y_test_pred)
# how to save the model

# model.save('pavan')



# how to load the model

# from keras.models import load_model

# model = load_model('pavan')



model.summary()
# Import necessary libraries



import torch

import torch.nn as nn

from torch.autograd import Variable

import torch.nn.functional as F

import torch.utils.data

import torch.optim as optim
X_train
type(X_train)
# Convert numpy array to tensor



# Train data

X_train = torch.from_numpy(X_train)

y_train = torch.from_numpy(y_train.values).view(-1,1)



# Test data

X_test  = torch.from_numpy(X_test)

y_test = torch.from_numpy(y_test.values).view(-1,1)
type(X_train)
type(X_test)
X_train
type(y_train)
y_train
class ANN(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(ANN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 6)           # Input layer and Hidden layer1

        self.fc2 = nn.Linear(6,6)                    # Hidden layer1 and Hidden layer2

        self.output_layer = nn.Linear(6, output_dim) # Hidden layer2 and Output layer



    # feed forward function    

    def forward(self,x):

        x = F.relu(self.fc1(x))                      # Activation function for hidden layer1

        x = F.relu(self.fc2(x))                      # Activation function for hidden layer2

        x = F.sigmoid(self.output_layer(x))          # Activation function for output layer

        return x



# Neural network model



model = ANN(input_dim = 11, output_dim = 1)



print(model)

# Loss function (Binary Cross Entropy loss)

criterion = nn.BCELoss()



# Optimizer  

optimizer = optim.Adam(model.parameters(), lr = 0.01)
model.eval()
# calculate the loss of test data before weight updates



data_test   = Variable(X_test).float()

target_test = Variable(y_test).type(torch.FloatTensor)



# predict test data

y_pred_test = model(data_test)

before_train = criterion(y_pred_test.squeeze(), target_test)



# Print the loss before training the test data

print('Test loss before training' , before_train.item())
model.train()



for epoch in range(1000):

    data = Variable(X_train).float()

    target = Variable(y_train).type(torch.FloatTensor)

    

    # forward pass

    output = model(data)

    loss   = criterion(output, target)

    

    # Backward pass (set grad to zero, pass the loss backward, apply optimizer and update weights)

    optimizer.zero_grad() # Sets gradients to zero for each iteration

    loss.backward()       # Perform backward pass to compute gradients

    optimizer.step()      # Update weights 

    

    if (epoch+1) % 10 == 0:

        print ('epoch [{}/{}], Loss: {:.3f}'.format(epoch+1, 1000, loss.item()))
# Calculate the test loss after updating the weights



data_test   = Variable(X_test).float()

target_test = Variable(y_test).type(torch.FloatTensor)



# predict test data

y_pred_test = model(data_test)

after_train = criterion(y_pred_test.squeeze(), target_test)



# Print the loss after training the test data

print('Test loss after training' , after_train.item())
metrics.roc_auc_score(y_test, y_pred_test.squeeze().detach().numpy())