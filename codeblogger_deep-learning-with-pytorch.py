# IMPORT THE NECESSARY LIBRARIES

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from torch.utils.data import DataLoader

from torch.autograd import Variable

import matplotlib.pyplot as plt

import torch.nn as nn

import pandas as pd

import numpy as np

import torch
car_prices_array = [3,4,5,6,7,8,9]

car_price_np = np.array(car_prices_array, dtype=np.float32)

car_price_np = car_price_np.reshape(-1,1)

car_price_tensor = Variable(torch.from_numpy(car_price_np))



number_of_car_sell_array = [7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]

number_of_car_sell_np = np.array(number_of_car_sell_array, dtype=np.float32)

number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)

number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))



# visualize

plt.scatter(car_prices_array, number_of_car_sell_array)

plt.xlabel("Car Price")

plt.ylabel("Number of Car Sell")

plt.title("Car Price & Number of Car Sell")

plt.show()
class LinearRegression(nn.Module):

    def __init__(self, input_size, output_size):

        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    

    def forward(self, x):

        return self.linear(x)

    





# DEFINE MODEL

input_dim = 1

output_dim = 1

model = LinearRegression(input_dim, output_dim)





# MSE

mse = nn.MSELoss()





# OPTIMIZATION

learning_rate = 0.02

optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)





loss_list = []

iteration_number = 1000

for iteration in range(iteration_number):

    

    # OPTIMIZATION

    optimizer.zero_grad()

    

    # FORWARD TO GET OUTPUT

    results = model(car_price_tensor)

    

    # CALCULATE LOSS

    loss = mse(results, number_of_car_sell_tensor)

    

    # BACKWARD PROPAGATION

    loss.backward()

    

    # UPDATING PARAMETERS

    optimizer.step()

    

    # STORE LOSS

    loss_list.append(loss.data)

    

    # PRINT LOSS

    if(iteration % 100 == 0):

        print('epoch {}, loss {}'.format(iteration, loss.data))

        

        

plt.plot(range(iteration_number),loss_list)

plt.xlabel("Number of Iterations")

plt.ylabel("Loss")

plt.show()
predicted = model(car_price_tensor).data.numpy()

plt.scatter(car_prices_array, number_of_car_sell_array, label="original data", color="red")

plt.scatter(car_prices_array, predicted, label="predicted  data", color="blue")



plt.legend()

plt.xlabel("Car Price")

plt.ylabel("Number of Car Sell")

plt.title("Original vs Predicted values")

plt.show()
# PREPARE DATASET

train = pd.read_csv(r"../input/fashionmnist/fashion-mnist_train.csv", dtype=np.float32)



targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 



features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42) 



featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long



batch_size = 100

n_iters = 10000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)



plt.imshow(features_numpy[20].reshape(28,28),cmap="gray")

plt.axis("off")

plt.title(str(targets_numpy[20]))

plt.savefig('graph.png')

plt.show()
# Create Logistic Regression Model

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LogisticRegressionModel, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    

    def forward(self, x):

        out = self.linear(x)

        return out



input_dim = 28*28 

output_dim = 10 



model = LogisticRegressionModel(input_dim, output_dim)



error = nn.CrossEntropyLoss()



learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
count = 0

loss_list = []

iteration_list = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        

        # Define variables

        train = Variable(images.view(-1, 28*28))

        labels = Variable(labels)

        

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and cross entropy loss

        loss = error(outputs, labels)

        

        # Calculate gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

        

        count += 1

        

        # Prediction

        if count % 50 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            # Predict test dataset

            for images, labels in test_loader: 

                test = Variable(images.view(-1, 28*28))

                

                # Forward propagation

                outputs = model(test)

                

                # Get predictions from the maximum value

                predicted = torch.max(outputs.data, 1)[1]

                

                # Total number of labels

                total += len(labels)

                

                # Total correct predictions

                correct += (predicted == labels).sum()

            

            accuracy = 100 * correct / float(total)

            

            # store loss and iteration

            loss_list.append(loss.data)

            iteration_list.append(count)

        if count % 500 == 0:

            # Print Loss

            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
# visualization

plt.figure(figsize=(25,6))

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("Logistic Regression: Loss vs Number of iteration")

plt.show()
# IMPORT LIBRARIES

import torch

import torch.nn as nn

from torch.autograd import Variable
# CREATE ANN MODEL

class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super(ANNModel, self).__init__()

        # Linear Function 1: 784 --> 150

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity 1

        self.relu1 = nn.ReLU()

        

        # Linear Function 2: 150 --> 150

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Non-linearity 2

        self.tanh2 = nn.Tanh()

        

        # Linear Function 3: 150 --> 150

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # Non-linearity 3

        self.elu3 = nn.ELU()

        

        # Linear Function 4 (readout) 150 --> 10

        self.fc4 = nn.Linear(hidden_dim, output_dim)

        

    def forward(self, x):

        out = self.fc1(x)

        out = self.relu1(out)

        

        out = self.fc2(out)

        out = self.tanh2(out)

        

        out = self.fc3(out)

        out = self.elu3(out)

        

        out = self.fc4(out)

        return out

    

input_dim = 28*28

hidden_dim = 150

output_dim = 10



# Craete ANN Model

model = ANNModel(input_dim, hidden_dim, output_dim)



# Cross Entropy Loss

error = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# ANN model training

count = 0

loss_list = []

iteration_list = []

accuracy_list = []



for epoch in range(num_epochs):

    for i,(images, labels) in enumerate(train_loader):

        train = Variable(images.view(-1, 28*28))

        labels = Variable(labels)

        

        # clear gradients

        optimizer.zero_grad()

        

        # forward propagation

        outputs = model(train)

        

        # calculate softmax and ross entropy loss

        loss = error(outputs, labels)

        

        # calculating gradients

        loss.backward()

        

        # update parameters

        optimizer.step()

        

        count += 1

        

        if count % 50 == 0:

            correct = 0

            total = 0

            for images, labels in test_loader:

                test = Variable(images.view(-1,28*28))

                outputs = model(test)

                predicted = torch.max(outputs.data, 1)[1]

                total += len(labels)

                correct += (predicted == labels).sum()

                

            accuracy = 100 * correct / float(total)

            loss_list.append(loss.data)

            iteration_list.append(count)

            accuracy_list.append(accuracy)

        if count % 500 == 0:

            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
# visualization loss

plt.figure(figsize=(25,6))

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("ANN: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.figure(figsize=(25,6))

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("ANN: Accuracy vs Number of iteration")

plt.show()
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable
# Create CNN Model

class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        # Convolution 1

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)

        self.relu1 = nn.ReLU()

        

        # Max pool 1

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

     

        # Convolution 2

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)

        self.relu2 = nn.ReLU()

        

        # Max pool 2

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        

        # Fully connected 1

        self.fc1 = nn.Linear(64 * 4 * 4, 10) 

        

    def forward(self, x):

        out = self.cnn1(x)

        out = self.relu1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)

        out = self.relu2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        return out

    

batch_size = 100

n_iters = 2500

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    

model = CNNModel()



error = nn.CrossEntropyLoss()



learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
count = 0

loss_list = []

iteration_list = []

accuracy_list = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        

        train = Variable(images.view(100,1,28,28))

        labels = Variable(labels)

        

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and ross entropy loss

        loss = error(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

        

        count += 1

        

        if count % 50 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            for images, labels in test_loader:

                test = Variable(images.view(100,1,28,28))

                outputs = model(test)

                predicted = torch.max(outputs.data, 1)[1]

                total += len(labels)

                correct += (predicted == labels).sum()

            

            accuracy = 100 * correct / float(total)

            loss_list.append(loss.data)

            iteration_list.append(count)

            accuracy_list.append(accuracy)

        if count % 500 == 0:

            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
# visualization loss 

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("CNN: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("CNN: Accuracy vs Number of iteration")

plt.show()
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
class RNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):

        super(RNNModel, self).__init__()

        

        # Number of hidden dimensions

        self.hidden_dim = hidden_dim

        

        # Number of hidden layers

        self.layer_dim = layer_dim

        

        # RNN

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        

        # Readout layer

        self.fc = nn.Linear(hidden_dim, output_dim)

    

    def forward(self, x):

        

        # Initialize hidden state with zeros

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

            

        # One time step

        out, hn = self.rnn(x, h0)

        out = self.fc(out[:, -1, :]) 

        return out



batch_size = 100

n_iters = 5000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)

    

# Create RNN

input_dim = 28    # input dimension

hidden_dim = 100  # hidden layer dimension

layer_dim = 1     # number of hidden layers

output_dim = 10   # output dimension



model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)



# Cross Entropy Loss 

error = nn.CrossEntropyLoss()



# SGD Optimizer

learning_rate = 0.05

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
seq_dim = 28  

loss_list = []

iteration_list = []

accuracy_list = []

count = 0

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):



        train  = Variable(images.view(-1, seq_dim, input_dim))

        labels = Variable(labels )

            

        # Clear gradients

        optimizer.zero_grad()

        

        # Forward propagation

        outputs = model(train)

        

        # Calculate softmax and ross entropy loss

        loss = error(outputs, labels)

        

        # Calculating gradients

        loss.backward()

        

        # Update parameters

        optimizer.step()

        

        count += 1

        

        if count % 500 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            for images, labels in test_loader:

                

                images = images.view(-1, seq_dim, input_dim)



                # Forward pass only to get logits/output

                outputs = model(images)



                # Get predictions from the maximum value

                _, predicted = torch.max(outputs.data, 1)



                # Total number of labels

                total += labels.size(0)



                # Total correct predictions

                correct += (predicted == labels).sum()



            accuracy = 100 * correct / total

            

            loss_list.append(loss.data.item())

            iteration_list.append(count)

            accuracy_list.append(accuracy)

            

            # Print Loss

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(count, loss.data.item(), accuracy))
# visualization loss 

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("RNN: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("RNN: Accuracy vs Number of iteration")

plt.show()
class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):

        super(LSTMModel, self).__init__()

        

        # Hidden dimensions

        self.hidden_dim = hidden_dim



        # Number of hidden layers

        self.layer_dim = layer_dim



        # LSTM

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # batch_first=True (batch_dim, seq_dim, feature_dim)



        # Readout layer

        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, x):

        # Initialize hidden state with zeros

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()



        # Initialize cell state

        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()



        # 28 time steps

        # We need to detach as we are doing truncated backpropagation through time (BPTT)

        # If we don't, we'll backprop all the way to the start even after going through another batch

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))



        # Index hidden state of last time step

        # out.size() --> 100, 28, 100

        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 

        out = self.fc(out[:, -1, :]) 

        # out.size() --> 100, 10

        return out

    

input_dim = 28

hidden_dim = 100

layer_dim = 1

output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)



error = nn.CrossEntropyLoss()



learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
seq_dim = 28  

loss_list = []

iteration_list = []

accuracy_list = []

count = 0



for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        # Load images as a torch tensor with gradient accumulation abilities

        images = images.view(-1, seq_dim, input_dim).requires_grad_()



        # Clear gradients w.r.t. parameters

        optimizer.zero_grad()



        # Forward pass to get output/logits

        # outputs.size 100, 10

        outputs = model(images)



        # Calculate Loss: softmax --> cross entropy loss

        loss = error(outputs, labels)



        # Getting gradients

        loss.backward()



        # Updating parameters

        optimizer.step()



        count += 1



        if count % 500 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            for images, labels in test_loader:

                

                images = images.view(-1, seq_dim, input_dim)



                # Forward pass only to get logits/output

                outputs = model(images)



                # Get predictions from the maximum value

                _, predicted = torch.max(outputs.data, 1)



                # Total number of labels

                total += labels.size(0)



                # Total correct predictions

                correct += (predicted == labels).sum()



            accuracy = 100 * correct / total

            

            loss_list.append(loss.data.item())

            iteration_list.append(count)

            accuracy_list.append(accuracy)

            

            # Print Loss

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(count, loss.data.item(), accuracy))
# visualization loss 

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("LSTM: Loss vs Number of iteration")

plt.show()



# visualization accuracy 

plt.plot(iteration_list,accuracy_list,color = "red")

plt.xlabel("Number of iteration")

plt.ylabel("Accuracy")

plt.title("LSTM: Accuracy vs Number of iteration")

plt.show()