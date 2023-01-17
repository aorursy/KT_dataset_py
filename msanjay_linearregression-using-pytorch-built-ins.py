import torch.nn as nn
import numpy as np
import torch
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')
 
# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], 
                    [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype='float32')
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
#import tensor dataset and data loader
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(inputs,targets)
train_ds[0:5]
#define data loadder
batch_size = 4
train_dl = DataLoader(train_ds,batch_size,shuffle=True)
next(iter(train_dl))
#Define the model, nn.Linear takes care of intializng the weights
model = nn.Linear(3,2) #3 inputs and 2 outputs
print(model.weight)
print(model.bias)
#Define Optimizer
opt = torch.optim.SGD(model.parameters(),lr=1e-4)
import torch.nn.functional as F
#Use the builtin loss function mse_loss
loss_fn = F.mse_loss
loss = loss_fn(model(inputs),targets)
print(loss)
#Define a utility functin to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            #Generate Predictions
            pred = model(xb)
            loss = loss_fn(pred,yb)
            #Perform Gradient Descent
            loss.backward()
            opt.step()
            #set the gradients to zero
            opt.zero_grad()
    print('Training Loss: ', loss_fn(model(inputs),targets))
#Train the model for 100 epochs
fit(100,model,loss_fn,opt)
#Generate predictions
preds = model(inputs)
preds
#compare with targets
targets
#Feed forward neural network
class SimpleNet(nn.Module):
    #initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(3,3)
        self.act1 = nn.ReLU()#Activation funcction
        self.linear2 = nn.Linear(3,2)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


# Now we can define model, optimizer and loss function like bfore
model = SimpleNet()
opt = torch.optim.SGD(model.parameters(),1e-5)
loss_fn = F.mse_loss
fit(200,model,loss_fn,opt)