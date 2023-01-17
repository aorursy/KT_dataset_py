import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch 

import pandas as pd

import numpy as np

from torch import nn,optim

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F

from sklearn.preprocessing import LabelBinarizer
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test  = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

print(train.shape)

print(test.shape)
# Splitting the labels from data

xtrain = np.array(train.iloc[:,1:])/255

ytrain = np.array(train.iloc[:,0])

xtest = np.array(test.iloc[:,1:])/255

ytest = np.array(test.iloc[:,0])



print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)
# Train loader

tensor_x = torch.Tensor(xtrain) # transform to torch tensor

tensor_y = torch.Tensor(ytrain)

tensor_y = tensor_y.type(torch.LongTensor)





my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset

train_loader = DataLoader(my_dataset,batch_size = 64) # create your dataloader for train data



# Test Loader

tensor_xtest = torch.Tensor(xtest) # transform to torch tensor

tensor_ytest = torch.Tensor(ytest)

tensor_ytest = tensor_ytest.type(torch.LongTensor)





my_dataset = TensorDataset(tensor_xtest,tensor_ytest) # create your datset

test_loader = DataLoader(my_dataset,batch_size = 64,shuffle=True) # create your dataloader for test data
class Net(nn.Module):

    

    def __init__(self):

        super().__init__()

        

        # Fully-connected layer-1

        self.fc1 = nn.Linear(784, 256)

        # Fully-connected layer-2

        self.fc2 = nn.Linear(256, 128)

        # Fully-connected layer-3

        self.fc3 = nn.Linear(128,64)

        # Fully-connected layer-4 or output-layer

        self.fc4 = nn.Linear(64,10)

        

        #Lets create a Dropout to prevent overfitting

        self.dropout = nn.Dropout(p=0.2)

        

    def forward(self,x):

        x = self.dropout(F.relu(self.fc1(x))) # ReLU and dropout is applied to layer-1

        x = self.dropout(F.relu(self.fc2(x))) # ReLU and dropout is applied to layer-1

        x = self.dropout(F.relu(self.fc3(x))) # ReLU and dropout is applied to layer-1

        x = F.log_softmax(self.fc4(x),dim=1)  # Softmax is applied to output layer

        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
model = Net()



# Loss 

criterion = nn.NLLLoss()



# Optimizer

optimizer = optim.Adam(model.parameters(),lr=0.003)
# Shift model to GPU

model = model.to(device)
epochs = 100



loss_train,acc_train,losstest,acctest = [],[],[],[]



for e in range(1,epochs + 1):

    

    # Setting model to train mode

    model.train()

    

    running_loss, total, correct = 0,0,0

    

    for images,labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        

        # Forward Pass

        outputs = model(images)

        loss = criterion(outputs,labels)

        

        # Backward Pass

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        #Metrics

        _, predicted = torch.max(outputs.data, 1)

        running_loss += loss.item()

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    else:

        # Validation Forward pass

        with torch.no_grad():

            

            # setting model to test mode

            model.eval()    # will switch off dropouts during test

            

            running_loss_test, total_test, correct_test = 0,0,0

            for images_test, labels_test in test_loader:

                images_test, labels_test = images_test.to(device), labels_test.to(device)

                # Forward Pass

                outputs_test = model(images_test)

                loss_test = criterion(outputs_test,labels_test)



                #Metrics

                _, predicted_test = torch.max(outputs_test.data, 1)

                running_loss_test += loss_test.item()

                total_test += labels_test.size(0)

                correct_test += (predicted_test == labels_test).sum().item()



        # Logs per Epoch

        loss_train.append(running_loss)

        acc_train.append(correct/total * 100)

        losstest.append(running_loss_test)

        acctest.append(correct_test/total_test * 100)

        print(f'Epoch {e} Train: Loss: {running_loss}, Accuracy = {correct/total * 100 :.2f} Validation: Loss: {running_loss_test}, Accuracy = {correct_test/total_test * 100 :.2f}')

        
import plotly.graph_objs as go



# Plot Loss

fig = go.Figure()

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = loss_train,

    mode = 'lines',

    name = 'Train loss'   

))

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = losstest,

    mode = 'lines',

    name = 'Validation loss'   

))



fig.update_layout(

    title = 'Train and Validation Loss'

)

fig.show()
# Plot Accuracy

fig = go.Figure()

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = acc_train,

    mode = 'lines',

    name = 'Train Accuracy'   

))

fig.add_trace(go.Scatter(

    x = list(range(epochs)),

    y = acctest,

    mode = 'lines',

    name = 'Validation Accuracy'   

))



fig.update_layout(

    title = 'Train and Validation Accuracy'

)

fig.show()