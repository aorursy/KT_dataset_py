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
from pathlib import Path

import requests



DATA_PATH = Path("data")

PATH = DATA_PATH / "mnist"



PATH.mkdir(parents=True, exist_ok=True)



URL = "http://deeplearning.net/data/mnist/"

FILENAME = "mnist.pkl.gz"



if not (PATH / FILENAME).exists():

        content = requests.get(URL + FILENAME).content

        (PATH / FILENAME).open("wb").write(content)
import pickle

import gzip



with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:

        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
## Check if data works well



from matplotlib import pyplot

import numpy as np



pyplot.imshow(x_train[4].reshape((28, 28)), cmap="spring")

print(x_train.shape)
import torch



x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid) )
n, c  = x_train.shape; n,c
print(x_train, y_train)

print(x_train.shape)

print(y_train.min(), y_train.max())
x1, x2 = torch.Size([2,3]); type(x1)
import math



weights = torch.rand(784,10)/math.sqrt(784)  ## This normalization is Xavier Normalization

weights.requires_grad_()
weights
bias = torch.zeros(10,requires_grad=True);bias
def log_softmax(x):

    return x - x.exp().sum(-1).log().unsqueeze(-1)



#NOTE: Actual log softmax also has a maximum value for evaluation, but since the values are not so high over here we are excluding max(x)
log_softmax(x_train[0]).shape
pyplot.imshow(log_softmax(x_train[0]).reshape((28, 28)))

print(x_train.shape)
pyplot.imshow(x_train[0].reshape((28, 28)))



#OBS: Pyplot shows same image even if we subtract a constant value from the image
def model(xb):

    return log_softmax(xb @ weights + bias)



#NOTE: `@` is for dot product between these two matrices
bs = 64  # batch size



xb = x_train[0:bs]  # a mini-batch from x

preds = model(xb)  # predictions

preds[0], preds.shape

print(preds[0], preds.shape)
LOGGER = False
def nll(input, target):

    """Calculates Negative Log Likelihood

    

    This function calculates NLL of the targets and take mean. 

    The function treats input as vector and selects only target 

    variable log likelihood and take a mean of the complete 

    batch.

    

    """

    if LOGGER:

        print("Range of Input:",range(target.shape[0]))

        print("Target Values:",target)

        print("Input:",input)

        print("Log Likelihood:",-input[range(target.shape[0]), target])

    return -input[range(target.shape[0]), target].mean()



loss_func = nll
yb = y_train[0:bs]

print(loss_func(preds, yb))
def accuracy(out, yb):

    """Accuracy of the Predicted Value"""

    preds = torch.argmax(out, dim=1) #Find out the argument of the max value from dimension 1

    return (preds == yb).float().mean()
print(accuracy(preds, yb))
from IPython.core.debugger import set_trace

from tqdm import tqdm





lr = 0.5 # Learning Rate

epochs = 5 # Number of Iterations



for epoch in tqdm(range(epochs)):

    for i in range((n-1)//bs+1):

        start_i = i * bs

        end_i = start_i + bs

        xb = x_train[start_i:end_i]

        yb = y_train[start_i:end_i]

        pred = model(xb)

        loss = loss_func(pred,yb)

        

        loss.backward()

        with torch.no_grad():

            weights -= weights.grad * lr

            bias -= bias.grad * lr

            weights.grad.zero_()

            bias.grad.zero_()
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
import torch.nn.functional as F



loss_func = F.cross_entropy



def model(xb):

    return xb @ weights + bias
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
from torch import nn



class Mnist_Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.weights = nn.Parameter(torch.rand(784,10)/math.sqrt(784))  ## This normalization is Xavier Normalization

        self.bias = nn.Parameter(torch.zeros(10))

        print("Initialized MNIST Model.")

        

    def forward(self,xb):

        return xb@self.weights + self.bias
model = Mnist_Model()
#NOTE: nn.Module objects are used as if they are functions (i.e they are callable), 

#but behind the scenes Pytorch will call our forward method automatically.



print(loss_func(model(xb),yb))
def fit():

    for epoch in tqdm(range(epochs)):

        for i in range((n - 1) // bs + 1):

            start_i = i * bs

            end_i = start_i + bs

            xb = x_train[start_i:end_i]

            yb = y_train[start_i:end_i]

            pred = model(xb)

            loss = loss_func(pred, yb)



            loss.backward()

            with torch.no_grad():

                for p in model.parameters():

                    p -= p.grad * lr

                model.zero_grad()



fit()
print(loss_func(model(xb), yb))
class Mnist_Logistic(nn.Module):

    def __init__(self):

        super().__init__()

        self.lin = nn.Linear(784,10)

        

    def forward(self,xb):

        return self.lin(xb)
model = Mnist_Logistic()

print(loss_func(model(xb), yb))
fit()



loss_func(model(xb), yb)
from torch import optim



def get_model():

    model = Mnist_Logistic()

    return model, optim.SGD(model.parameters(),lr=lr) # A simple SGD is used on model parameters with learning rate, lr
model, opt = get_model()



def fit():

    for epoch in tqdm(range(epochs)):

        for i in range((n - 1) // bs + 1):

            start_i = i * bs

            end_i = start_i + bs

            xb = x_train[start_i:end_i]

            yb = y_train[start_i:end_i]

            pred = model(xb)

            loss = loss_func(pred, yb)



            loss.backward()

            opt.step()

            opt.zero_grad()
fit()



loss_func(model(xb), yb)
from torch.utils.data import TensorDataset



train_ds = TensorDataset(x_train,y_train)
model, opt = get_model()



for epoch in tqdm(range(epochs)):

    for i in range((n - 1) // bs + 1):

        xb, yb = train_ds[i * bs: i * bs + bs]

        pred = model(xb)

        loss = loss_func(pred, yb)



        loss.backward()

        opt.step()

        opt.zero_grad()



print(loss_func(model(xb), yb))
from torch.utils.data import DataLoader



train_ds = TensorDataset(x_train,y_train)

train_dl = DataLoader(train_ds,batch_size=bs)
model, opt = get_model()



for epoch in tqdm(range(epochs)):

    for xb, yb in train_dl:

        loss = loss_func(model(xb),yb)

        loss.backward()

        

        opt.step()

        opt.zero_grad()
print(loss_func(model(xb), yb))
train_ds = TensorDataset(x_train, y_train)

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)



valid_ds = TensorDataset(x_valid, y_valid)

valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
model, opt = get_model()



validation_results = list()



for epoch in tqdm(range(epochs)):

    model.train()

    for xb, yb in train_dl: #Shuffle is ON here.

        loss = loss_func(model(xb), yb)



        loss.backward()

        opt.step()

        opt.zero_grad()



    model.eval()

    with torch.no_grad():

        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)



    validation_results.append((epoch, valid_loss / len(valid_dl)))

    

print(validation_results)
def loss_batch(xb,yb,loss_func,model,optimizer=None):

    """Returns loss and size of Batch"""

    loss = loss_func(model(xb),yb)

    

    if optimizer is not None:

        loss.backward()

        opt.step()

        opt.zero_grad()

    

    return loss.item(),len(xb)
def fit(model,optimizer,loss_func,train_dl,valid_dl,epochs=5):

    """Fit Model"""

    for epoch in range(epochs):

        # Training

        model.train()

        for xb,yb in train_dl:

            loss, _ = loss_batch(xb,yb,loss_func,model,optimizer=optimizer)

            

        # Validation

        model.eval()

        with torch.no_grad():

            losses, num = zip(*[loss_batch(xb,yb,loss_func,model,optimizer=None) for xb,yb in valid_dl])

        val_loss = np.sum(np.multiply(losses,num))/np.sum(num)

        

        print(epoch,val_loss.item())
def get_data(train_ds,valid_ds,batch_size=128):

    return(

        DataLoader(train_ds,batch_size=batch_size,shuffle=True),

        DataLoader(valid_ds,batch_size=2*batch_size)

    )
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

model, opt = get_model()

fit(model, opt,loss_func, train_dl, valid_dl,epochs)
class Mnist_CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layer_1 = nn.Conv2d( 1,16,kernel_size=3, stride=2, padding=1)

        self.conv_layer_2 = nn.Conv2d(16,16,kernel_size=3, stride=2, padding=1)

        self.conv_layer_3 = nn.Conv2d(16,16,kernel_size=3, stride=2, padding=1)

        

    def forward(self,xb):

        xb = xb.view(-1,1,28,28) ## Reshape the unspecified input dimension[Single or Multiple IP] to 1 Channel * 28 Unit Height * 28 Unit Width

        xb = F.relu(self.conv_layer_1(xb))

        xb = F.relu(self.conv_layer_2(xb))

        xb = F.relu(self.conv_layer_3(xb))

        xb = F.avg_pool2d(xb,4)

#         print(xb.shape) # Size of xb = (-1,16,1,1)

        return xb.view(-1,xb.size(1))
lr = 0.1



#Changing Learning Rate
# Momentum is added for faster training so it also its some previous updates into account



model = Mnist_CNN()

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)



fit(model, opt,loss_func, train_dl, valid_dl,epochs)
class Lambda(nn.Module):

    def __init__(self,func):

        super().__init__()

        self.func = func

        

    def forward(self,xb):

        return self.func(xb)

    

def preprocess(xb):

    return xb.view(-1,1,28,28)
model = nn.Sequential(

    Lambda(preprocess),

    nn.Conv2d( 1,16,kernel_size=3, stride=2, padding=1),

    nn.ReLU(),

    nn.Conv2d(16,16,kernel_size=3, stride=2, padding=1),

    nn.ReLU(),

    nn.Conv2d(16,16,kernel_size=3, stride=2, padding=1),

    nn.ReLU(),

    nn.AvgPool2d(4),

    Lambda(lambda x: x.view(x.size(0),-1))

)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)



fit(model, opt,loss_func, train_dl, valid_dl,epochs)
loss_func(model(xb[0]),torch.tensor([yb[0]]))



# Check this link for more info https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
model(xb[0]).shape
model(xb[1:3]).shape



#OBS: Output is in batch format only
def preprocess(x, y):

    return x.view(-1, 1, 28, 28), y





class WrappedDataLoader:

    def __init__(self, data_loader, func):

        self.data_loader = data_loader

        self.func = func



    def __len__(self):

        return len(self.data_loader)



    def __iter__(self):

        batches = iter(self.data_loader)

        for b in batches:

            yield (self.func(*b))



train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

train_dl = WrappedDataLoader(train_dl, preprocess)

valid_dl = WrappedDataLoader(valid_dl, preprocess)
model = nn.Sequential(

    nn.Conv2d( 1,16,kernel_size=3, stride=2, padding=1),

    nn.ReLU(),

    nn.Conv2d(16,16,kernel_size=3, stride=2, padding=1),

    nn.ReLU(),

    nn.Conv2d(16,16,kernel_size=3, stride=2, padding=1),

    nn.ReLU(),

    nn.AvgPool2d(4),

    Lambda(lambda x: x.view(x.size(0),-1))

)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)



fit(model, opt,loss_func, train_dl, valid_dl,epochs)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Data moved to device



def preprocess(x, y):

    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

train_dl = WrappedDataLoader(train_dl, preprocess)

valid_dl = WrappedDataLoader(valid_dl, preprocess)
# Model and Parameters moved to Device



model.to(dev)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
for layer in model.children():

    if hasattr(layer, 'reset_parameters'):

        layer.reset_parameters()
fit(model, opt,loss_func, train_dl, valid_dl,epochs)