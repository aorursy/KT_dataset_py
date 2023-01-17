import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.datasets as dsets

import argparse

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from torch.autograd import Variable









train_dataset = pd.read_csv('../input/digit-recognizer/train.csv',dtype = np.float32)



test_dataset = pd.read_csv('../input/digit-recognizer/test.csv',dtype = np.float32)



targets_numpy = train_dataset.label.values

features_numpy = train_dataset.loc[:,train_dataset.columns != "label"].values/255 # normalization



# Using SKLEARN we train test split. Size of train data is 80% and size of test data is 20%. 

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

n_iters = 10000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)





#print(train_dataset.train_data.size())



#print(train_dataset.train_labels.size())

#Here we would have 10k testing images of the same size, 28 x 28 pixels.





#print(test_dataset.test_data.size())



#print(test_dataset.test_labels.size())







train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 

                                           batch_size=batch_size, 

                                           shuffle=True)



test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 

                                          batch_size=batch_size, 

                                          shuffle=False)



# Create RNN Model



class RNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):

        super(RNNModel, self).__init__()

        # Hidden dimensions

        self.hidden_dim = hidden_dim



        # Number of hidden layers

        self.layer_dim = layer_dim



        # Building your RNN

        # batch_first=True causes input/output tensors to be of shape

        # (batch_dim, seq_dim, input_dim)

        # batch_dim = number of samples per batch

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')



        # Readout layer

        self.fc = nn.Linear(hidden_dim, output_dim)



    def forward(self, x):

        # Initialize hidden state with zeros

        # (layer_dim, batch_size, hidden_dim)

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()



        # We need to detach the hidden state to prevent exploding/vanishing gradients

        # This is part of truncated backpropagation through time (BPTT)

        out, hn = self.rnn(x, h0.detach())



        # Index hidden state of last time step

        # out.size() --> 100, 28, 10

        # out[:, -1, :] --> 100, 10 --> just want last time step hidden states! 

        out = self.fc(out[:, -1, :]) 

        # out.size() --> 100, 10

        return out



# batch_size, epoch and iteration

batch_size = 100

n_iters = 3000

num_epochs = n_iters / (len(features_train) / batch_size)

num_epochs = int(num_epochs)



# Pytorch train and test sets

train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest,targetsTest)



# data loader

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    

# Create RNN

input_dim = 28    # input dimension

hidden_dim = 100  # hidden layer dimension

layer_dim = 3     # number of hidden layers

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

min_val_loss = np.Inf

val_array = []

correct = 0

iter = 0

count = 0

iter_array = []

loss_array = []

total = 0

accuracy_array = []

n_epochs_stop = 6

epochs_no_improve = 0

early_stop = False
for epoch in range(num_epochs):

    val_loss = 0

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

        val_loss += loss

        val_loss = val_loss / len(train_loader)

        # If the validation loss is at a minimum

        if val_loss < min_val_loss:

  # Save the model

             #torch.save(model)

             epochs_no_improve = 0

             min_val_loss = val_loss

  

        else:

            epochs_no_improve += 1

        iter += 1

        if epoch > 5 and epochs_no_improve == n_epochs_stop:

            print('Early stopping!' )

            early_stop = True

            break

        else:

            continue

        break

        if iter % 336 == 0:

            # Calculate Accuracy         

            correct = 0

            total = 0

            #print(iter)

            # Iterate through test dataset

  # Check early stopping condition

        

    if early_stop:

        print("Stopped")

        break

        

        

        

    for images, labels in test_loader:

       

                # Resize images

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

        

        #Print Loss

        count = count +1

        if iter % 336 == 0 and count % 100 == 0  : 

            iter_array.append(iter)

            loss_array.append(loss.item())

            accuracy_array.append(accuracy.item())

            print('Epoch: {}. Iteration: {}. Loss: {}. Accuracy: {}, Count: {}'.format(epoch,iter, loss.item(),accuracy.item(),count))
examples = enumerate(test_loader)

batch_idx, (images, labels) = next(examples)

images = images.numpy()

labels = labels.numpy()



import matplotlib.pyplot as plt



fig = plt.figure()

for i in range(6):

  plt.subplot(2,3,i+1)

  plt.tight_layout()

  plt.imshow(images[i].reshape(28,28), cmap='gray', interpolation='none')

  plt.title("Number: {}".format(labels[i]))

  plt.xticks([])

  plt.yticks([])

print(fig)
df = pd.DataFrame({'Iterations': iter_array, 'Loss': loss_array, 'Accuracy': accuracy_array})

df['Index'] = range(1, len(iter_array) + 1)
from bokeh.plotting import figure, output_file, show

from bokeh.io import output_notebook

from bokeh.models import CustomJS, ColumnDataSource, Select,HoverTool,LinearInterpolator,Column

from bokeh.layouts import column

from bokeh.models.widgets import Div

output_notebook()

source_CDS = ColumnDataSource(df)
df
hover = HoverTool(tooltips = '@Loss= Loss')

Loss_line = figure(plot_width=700, plot_height=300,tools = [hover])



Loss_line.line('Iterations','Loss',source = source_CDS, line_width=2)

Loss_line.background_fill_color = '#fffce6'



title_div = Div(text="<b> Loss vs Iterations </b>", style={'font-size': '400%', 'color': '#FF6347'})

p2 = column(title_div,Loss_line)



show(p2)
hover = HoverTool(tooltips = ' Accuracy: @Accuracy%')

Accuracy_line = figure(plot_width=700, plot_height=300,tools = [hover])

Accuracy_line.line('Iterations','Accuracy',source = source_CDS, line_width=2)

title_div2 = Div(text="<b> Accuracy vs Iterations </b>", style={'font-size': '400%', 'color': '#008080'})

Accuracy_line.background_fill_color = '#fffce6'

p2 = column(title_div2,Accuracy_line)

show(p2)
test_dataset = pd.read_csv('../input/digit-recognizer/test.csv',dtype = np.float32)

test_dataset.shape
test_dataset = torch.from_numpy(test_dataset.values)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 

                                          shuffle=False)

model.eval()

test_pred = torch.LongTensor()

for i, data in enumerate(test_loader):

    

    

    predict = data.view(-1,seq_dim,input_dim)

    predict = Variable(predict)

    output = model(predict)

    pred = output.data.max(1, keepdim=True)[1]

    test_pred =  torch.cat((test_pred,pred),dim =0)

        
test_pred.size()
Submission_df = pd.DataFrame(np.c_[np.arange(1, len(test_pred.numpy())+1)[:,None], test_pred.numpy()], 

                      columns=['ImageId', 'Label'])

print(Submission_df.head())







Submission_df.to_csv('submission.csv', index=False)
test_dataset2 = pd.read_csv('../input/digit-recognizer/test.csv',dtype = np.float32)

test_dataset2 = test_dataset2.values



plt.imshow(test_dataset2[4].reshape(28,28))

print(test_pred[4])
test_pred
from collections import Counter

list = Counter(Submission_df['Label'].values)

list
train_dataset2 =  pd.read_csv('../input/digit-recognizer/train.csv',dtype = np.float32)



train_dataset2 = train_dataset2["label"].values # normalization







list = Counter(train_dataset2)

list