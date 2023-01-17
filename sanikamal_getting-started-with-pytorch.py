# import PyTorch

import torch

# import torchvision

import torchvision



# get PyTorch version

print(torch.__version__)

# get torchvision version

print(torchvision.__version__)
# checking if cuda is available

torch.cuda.is_available()
# get number of cuda/gpu devices

torch.cuda.device_count()
# get cuda/gpu device id

torch.cuda.current_device()
# get cuda/gpu device name

torch.cuda.get_device_name(0)
x = torch.rand(2,3)

print(x)
# Define a tensor with a default data type:

x = torch.ones(2,3)

print(x)

print(x.dtype)
# define a tensor with specific data type

x = torch.ones(2,3,dtype=torch.int16)

print(x)

print(x.dtype)
x = torch.ones(3,3,dtype=torch.int8)

print(x.dtype)

# Change the tensor datatype

x = x.type(torch.float32)

print(x.dtype)
s_val = torch.full((3,4),3.1416)

print(s_val)
e_val = torch.empty((3,4))

print(e_val)
r_val = torch.randn((4,5))

print(r_val)
rng_val = torch.randint(10,20,(3,4))

print(rng_val)
# Define a tensor

x = torch.rand(2,3)

print(x)

print(x.dtype)

# convert tensor into numpy array

y = x.numpy()

print(y)

print(y.dtype)
import numpy as np

# define a numpy array

x = np.ones((2,3),dtype=np.float32)

print(x)

print(x.dtype)

# convert to pytorch tensor

y = torch.from_numpy(x)

print(y)

print(y.dtype)
# Define a tensor in cpu

x = torch.tensor([2.3,5.8])

print(x)

print(x.device)



# Define a CUDA device

if torch.cuda.is_available():

    device= torch.device("cuda:0")

    

# Move the tensor onto CUDA device

x = x.to(device)

print(x)

print(x.device)
# Similarly, we can move tensors to CPU:

# define a cpu device

device = torch.device("cpu")

x = x.to(device) 

print(x)

print(x.device)
# We can also directly create a tensor on any device:

# define a tensor on device

device = torch.device("cuda:0")

x = torch.ones(2,2, device=device) 

print(x)
from torchvision import datasets

# path to store data and/or load from

data_path="./data"

# loading training data

train_data=datasets.MNIST(data_path, train=True, download=True)
# extract data and targets

x_train, y_train=train_data.data,train_data.targets

print(x_train.shape)

print(y_train.shape)
# loading validation data

val_data=datasets.MNIST(data_path, train=False, download=True)
# extract data and targets

x_val,y_val=val_data.data, val_data.targets

print(x_val.shape)

print(y_val.shape)
# add a dimension to tensor to become B*C*H*W

if len(x_train.shape)==3:

    x_train=x_train.unsqueeze(1)

print(x_train.shape)



if len(x_val.shape)==3:

    x_val=x_val.unsqueeze(1)

print(x_val.shape)
from torchvision import utils

import matplotlib.pyplot as plt

%matplotlib inline



# define a helper function to display tensors as images

def show(img):

    # convert tensor to numpy array

    npimg = img.numpy()

    # Convert to H*W*C shape

    npimg_tr=np.transpose(npimg, (1,2,0))

    plt.imshow(npimg_tr,interpolation='nearest')



    

# make a grid of 40 images, 8 images per row

x_grid=utils.make_grid(x_train[:40], nrow=8, padding=2)

print(x_grid.shape)

# call helper function

show(x_grid)
from torchvision import transforms

# define transformations

data_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(p=1),

        transforms.RandomVerticalFlip(p=1),

        transforms.ToTensor(),

    ])
# get a sample image from training dataset

img = train_data[0][0]



# transform sample image

img_tr=data_transform(img)



# convert tensor to numpy array

img_tr_np=img_tr.numpy()



# show original and transformed images

plt.subplot(1,2,1)

plt.imshow(img,cmap="gray")

plt.title("original")

plt.subplot(1,2,2)

plt.imshow(img_tr_np[0],cmap="gray");

plt.title("transformed")
# define transformations

# data_transform = transforms.Compose([

#         transforms.RandomHorizontalFlip(1),

#         transforms.RandomVerticalFlip(1),

#         transforms.ToTensor(),

#     ])



# Loading MNIST training data with on-the-fly transformations

# train_data=datasets.MNIST(path2data, train=True, download=True, transform=data_transform )
from torch.utils.data import TensorDataset



# wrap tensors into a dataset

train_ds = TensorDataset(x_train, y_train)

val_ds = TensorDataset(x_val, y_val)



for x,y in train_ds:

    print(x.shape,y.item())

    break
from torch.utils.data import DataLoader



# create a data loader from dataset

train_dl = DataLoader(train_ds, batch_size=8)

val_dl = DataLoader(val_ds, batch_size=8)



# iterate over batches

for xb,yb in train_dl:

    print(xb.shape)

    print(yb.shape)

    break
from torch import nn



# input tensor dimension 64*1000

input_tensor = torch.randn(64, 1000) 

# linear layer with 1000 inputs and 10 outputs

linear_layer = nn.Linear(1000, 10) 

# output of the linear layer

output = linear_layer(input_tensor) 

print(output.size())
# implement and print the model using nn.Sequential

from torch import nn



# define a two-layer model

model = nn.Sequential(

    nn.Linear(4, 5),

    nn.ReLU(), 

    nn.Linear(5, 1),

)

print(model)
import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)

        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 500)

        self.fc2 = nn.Linear(500, 10)

    

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4*4*50)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

#     Net.__init__ = __init__

#     Net.forward = forward
model = Net()    

print(model)
print(next(model.parameters()).device)
device = torch.device("cuda:0")

model.to(device)

print(next(model.parameters()).device)
!pip install torchsummary
# model summary using torchsummary

from torchsummary import summary

summary(model, input_size=(1, 28, 28))
from torch import nn

loss_func = nn.NLLLoss(reduction="sum")
for xb, yb in train_dl:

    # move batch to cuda device

    xb=xb.type(torch.float).to(device)

    yb=yb.to(device)

    # get model output

    out=model(xb)

    # calculate loss value

    loss = loss_func(out, yb)

    print (loss.item())

    break
# define the Adam optimizer

from torch import optim

opt = optim.Adam(model.parameters(), lr=1e-4)
# update model parameters

opt.step()
# set gradients to zero

opt.zero_grad()
#  helper function to compute the loss value per mini-batch

def loss_batch(loss_func, xb, yb,yb_h, opt=None):

    # obtain loss

    loss = loss_func(yb_h, yb)

    

    # obtain performance metric

    metric_b = metrics_batch(yb,yb_h)

    

    if opt is not None:

        loss.backward()

        opt.step()

        opt.zero_grad()



    return loss.item(), metric_b
# helper function to compute the accuracy per mini-batch

def metrics_batch(target, output):

    # obtain output class

    pred = output.argmax(dim=1, keepdim=True)

    

    # compare output class with target class

    corrects=pred.eq(target.view_as(pred)).sum().item()

    return corrects
# helper function to compute the loss and metric values for a dataset

def loss_epoch(model,loss_func,dataset_dl,opt=None):

    loss=0.0

    metric=0.0

    len_data=len(dataset_dl.dataset)

    for xb, yb in dataset_dl:

        xb=xb.type(torch.float).to(device)

        yb=yb.to(device)

        

        # obtain model output

        yb_h=model(xb)



        loss_b,metric_b=loss_batch(loss_func, xb, yb,yb_h, opt)

        loss+=loss_b

        if metric_b is not None:

            metric+=metric_b

    loss/=len_data

    metric/=len_data

    return loss, metric
def train_val(epochs, model, loss_func, opt, train_dl, val_dl):

    for epoch in range(epochs):

        model.train()

        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)

  

        model.eval()

        with torch.no_grad():

            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)

        

        accuracy=100*val_metric



        print("epoch: %d, train loss: %.6f, val loss: %.6f, accuracy: %.2f" %(epoch, train_loss,val_loss,accuracy))
# call train_val function

num_epochs=5

train_val(num_epochs, model, loss_func, opt, train_dl, val_dl)
# define path to weights

weights_path ="weights.pt"

 

# store state_dict to file

torch.save(model.state_dict(), weights_path)
# define model: weights are randomly initiated

_model = Net()

weights=torch.load(weights_path)

_model.load_state_dict(weights)

_model.to(device)
n=100

x= x_val[n]

y=y_val[n]

print(x.shape)

plt.imshow(x.numpy()[0],cmap="gray")
# we use unsqueeze to expand dimensions to 1*C*H*W

x= x.unsqueeze(0)



# convert to torch.float32

x=x.type(torch.float)



# move to cuda device

x=x.to(device)
# get model output

output=_model(x)



# get predicted class

pred = output.argmax(dim=1, keepdim=True)

print (pred.item(),y.item())