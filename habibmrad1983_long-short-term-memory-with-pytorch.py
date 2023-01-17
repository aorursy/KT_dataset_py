# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries

import torch

import torch.nn as nn

from torch.autograd import Variable

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset
# load data

train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv",dtype = np.float32)



# split data into features(pixels) and labels(numbers from 0 to 9)

targets_numpy = train.label.values

features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization



# train test split. Size of train data is 80% and size of test data is 20%. 

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,

                                                                             targets_numpy,

                                                                             test_size = 0.2,

                                                                             random_state = 42) 



# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable

featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long



# create feature and targets tensor for test set.

featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long



# batch_size, epoch and iteration

batch_size = 100

n_iters = 6000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)

print("Epoch Number: ",num_epochs)



# Pytorch train and test sets

train = TensorDataset(featuresTrain,targetsTrain)

test = TensorDataset(featuresTest,targetsTest)



# data loader

train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)



# visualize one of the images in data set

plt.imshow(features_numpy[10].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[10]))

plt.savefig('graph.png')

plt.show()



print(len(train_loader.dataset))

print(len(test_loader.dataset))

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):

        super(LSTMModel, self).__init__()

        

        # Hidden dimensions

        self.hidden_dim = hidden_dim



        # Number of hidden layers

        self.layer_dim = layer_dim



        # LSTM

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) # batch_first=True (batch_dim, seq_dim, feature_dim)

#         batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False

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

        # detach() is to detach a tensor from the network graph, making the tensor no gradient, while '. 

        #'data' is only to obtain tensor-data from Variable. They have different functions, which are used in different cases.

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
# Number of steps to unroll

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

plt.savefig('graph.png')

plt.show()