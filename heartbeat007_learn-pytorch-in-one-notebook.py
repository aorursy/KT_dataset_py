import  torch



## create a tensor

your_first_tensor = torch.tensor([3,3])

## for random tensor

your_first_tensor = torch.rand(3,3)
## finding tensor size

tensor_size = your_first_tensor.size()
# printing the element

print(your_first_tensor)

print(tensor_size)
tensor_of_ones = torch.ones(3,3)

## create a identity tensor

identity_tensor = torch.eye(3)
print(tensor_of_ones)

print(identity_tensor)
### matrix multiplication

matrices_multiplied = torch.matmul(tensor_of_ones,identity_tensor)
print(matrices_multiplied)
#### elementwise multiplication

element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)
## forward propargation

## implement a forward propagation 

## we apply this computational graph
import torch

a = torch.Tensor([2])

b = torch.Tensor([-4])

c = torch.Tensor([-2])

d = torch.Tensor([-4])

e = a+b

f = c+d

g = e*f

print(e,f,g)
## second computation graph
## assigning three random matrix with

## shape (1000,1000)

import torch

x = torch.rand(1000,1000)

y = torch.rand(1000,1000)

z = torch.rand(1000,1000)
## multiply (matrix multiplication)

q = torch.matmul(x,y)
## now elementwise multiplication

f = z*q
## find the mean

mean_f = torch.mean(f)
print(mean_f)
### back propagation algorithm
## we use the word gradient in terms of derivative

## the derivatives / gradient in calculated by the reverse mode of auto

## differentiation call backpropagation

## lets calculate the derivaties of this computational graph

import torch



## we initializa the tensor x y and z

## remember you need to give a . to make it float

## Only Tensors of floating point dtype can require gradients

x = torch.tensor(-3.,requires_grad=True)

y = torch.tensor(5.,requires_grad=True)

z = torch.tensor(-2.,requires_grad=True)



# doing the operation

q = x+y

f = q*z



## finally we compute the derivatices 

## using the backward() method



f.backward()



print("Gradient of z is: "+str(z.grad))

print("Gradient of y is: "+str(y.grad))

print("Gradient of x is: "+str(x.grad))

## practice derivatives
import torch



## we initializa the tensor x y and z

## remember you need to give a . to make it float

## Only Tensors of floating point dtype can require gradients

x = torch.tensor(4.,requires_grad=True)

y = torch.tensor(-3.,requires_grad=True)

z = torch.tensor(5.,requires_grad=True)



# doing the operation

q = x+y

f = q*z



## finally we compute the derivatices 

## using the backward() method



f.backward()



print("Gradient of z is: "+str(z.grad))

print("Gradient of y is: "+str(y.grad))

print("Gradient of x is: "+str(x.grad))

## another practice

# Initialize x, y and z to values 4, -3 and 5

x = torch.tensor(4.,requires_grad=True)

y = torch.tensor(-3.,requires_grad=True)

z = torch.tensor(5.,requires_grad=True)



# Set q to sum of x and y, set f to product of q with z

q = x+y

f = q*z



# Compute the derivatives

f.backward()



# Print the gradients

print("Gradient of x is: " + str(x.grad))

print("Gradient of y is: " + str(y.grad))

print("Gradient of z is: " + str(z.grad))
### find this graphs gradient 
import torch

x = torch.rand(1000,1000,requires_grad=True)

y = torch.rand(1000,1000,requires_grad=True)

z = torch.rand(1000,1000,requires_grad=True)

## multiply (matrix multiplication)

q = torch.matmul(x,y)

## now elementwise multiplication

f = z*q

## find the mean

mean_f = torch.mean(f)

mean_f.backward()
# weight_1 = torch.rand(784,200)

# weight_2 = torch.rand(200, 10)



# # Multiply input_layer with weight_1

# hidden_1 = torch.matmul(input_layer, weight_1)



# # Multiply hidden_1 with weight_2

# output_layer = torch.matmul(hidden_1,weight_2)

# print(output_layer)
import torch

input_layer = torch.rand(10)

w1 = torch.rand(10,20)

w2 = torch.rand(20,20)

w3 = torch.rand(20,4)



hidden_layer_1 = torch.matmul(input_layer,w1)

hidden_layer_2 = torch.matmul(hidden_layer_1,w2)

output_layer = torch.matmul(hidden_layer_2,w3)

print(output_layer)

### same neural network object oriented way

import torch

import torch.nn as nn



class Net(nn.Module):

  def __init__(self):

    ## calling the base constructor

    super(Net,self).__init__()

    ## instantiate all two linear layers

    ## input neurron and output neuron

    self.fc1 = nn.Linear(784,200)

    ## output neuron of the second will be the input of the first

    self.fc2 = nn.Linear(200,10)



  def forward(self,x):

    x = self.fc1(x)

    x = self.fc2(x)

    return x

    
net = Net()
net
## simple implementation of relu

import torch.nn as nn

relu = nn.ReLU()

tensor_1 = torch.tensor([2.,-4.])

print(relu(tensor_1))
# # Calculate the first and second hidden layer

# hidden_1 = torch.matmul(input_layer, weight_1)

# hidden_2 = torch.matmul(hidden_1, weight_2)



# # Calculate the output

# print(torch.matmul(hidden_2, weight_3))



# # Calculate weight_composed_1 and weight

# weight_composed_1 = torch.matmul(weight_1, weight_2)

# weight = torch.matmul(weight_composed_1, weight_3)



# # Multiply input_layer with weight

# print(torch.matmul(input_layer,weight))
# # Apply non-linearity on hidden_1 and hidden_2

# hidden_1_activated = relu(torch.matmul(input_layer, weight_1))

# hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))

# print(torch.matmul(hidden_2_activated, weight_3))



# # Apply non-linearity in the product of first two weights. 

# weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))



# # Multiply `weight_composed_1_activated` with `weight_3

# weight = torch.matmul(weight_composed_1_activated, weight_3)

# # Multiply input_layer with weight

# print(torch.matmul(input_layer, weight))

input_layer = torch.tensor([[ 0.0401, -0.9005,  0.0397, -0.0876]])

# Instantiate ReLU activation function as relu

relu = nn.ReLU()



# Initialize weight_1 and weight_2 with random numbers

weight_1 = torch.rand(4, 6)

weight_2 = torch.rand(6, 2)



# Multiply input_layer with weight_1

hidden_1 = torch.matmul(input_layer, weight_1)



# Apply ReLU activation function over hidden_1 and multiply with weight_2

hidden_1_activated = relu(hidden_1)

print(torch.matmul(hidden_1_activated, weight_2))
## an example of the cross Cross Entropy Loss 


logits = torch.tensor([[3.2,5.1,-1.7]])

ground_truth = torch.tensor([0])

criterion = nn.CrossEntropyLoss()

loss = criterion(logits,ground_truth)

print(loss)
logits = torch.tensor([[-1.2,.12,4.8]])

ground_truth = torch.tensor([2])

criterion = nn.CrossEntropyLoss()

loss = criterion(logits,ground_truth)

print(loss)
# Import torch and torch.nn

import torch 

import torch.nn as nn



# Initialize logits and ground truth

logits = torch.rand(1,1000)

ground_truth = torch.tensor([111])

criterion = nn.CrossEntropyLoss()

loss = criterion(logits,ground_truth)

print(loss)
import torch

import torchvision

import torch.utils.data

import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.1307),(.3081))])
## download data

trainset = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
## download data

testset = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
## now we make data loader so that we can feed it with batchs

trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True,num_workers=0)

testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=False,num_workers=0)
print(trainloader.dataset.train_data.shape)

print(trainloader.dataset.test_data.shape)
# Compute the size of the minibatch for training set and testing set

trainset_batchsize = trainloader.batch_size

testset_batchsize = testloader.batch_size



# Print sizes of the minibatch

print(trainset_batchsize,testset_batchsize)
hight = 28

width = 28

chanel = 1 ## grayscale image for rgb it will be 3

image_shape = 28*28*1

hidden_neuron = 500

output_neuron = 10
import torch 

import torch.nn

import torch.nn.functional as F

import torch.optim as optim
### make the class

class Net(nn.Module):

  def __init__(self):

    super(Net,self).__init__()

    self.fc1 = nn.Linear(image_shape,hidden_neuron)

    self.fc2 = nn.Linear(hidden_neuron,output_neuron)

  def forward(self,x):

    x = F.relu(self.fc1(x))

    ## in the final layer there is no relu

    x = self.fc2(x)

    return x
net = Net()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr=3e-4)
e = []  ## store the apochs

l =[] ## store the loss



for epoch in range(30):

  ## for 30 iteration

  ## we loop through the data

  for data in trainloader:

    inputs,labels = data  ## list unpacking

    ## flat the array inti 1D with reshape

    inputs = inputs.view(-1,28*28*1)



    ## zero gradient so it wont get ata from preious iteration

    optimizer.zero_grad()

    ## make a forward pass

    outputs = net(inputs)

    loss = criterion(outputs,labels)

    loss.backward() ## back propagation

    optimizer.step() ## replace the weight

  print ("EPOCHS: {} LOSS {} ".format(epoch,loss))

  e.append(epoch)

  l.append(loss)

import matplotlib.pyplot as plt

plt.plot(e, l, 'g', label='Training loss')

#plt.plot(epochs, loss_val, 'b', label='validation loss')

plt.title('Training  loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
import numpy as np

### making prediction

correct_prediction = 0

predictions = []

total_case = 0

net.eval() ### before prediction you need to set the evaluation mode

for data in testloader:

  ### list unpacking

  inputs,labels = data

  inputs = inputs.view(-1,28*28*1)

  outputs = net(inputs)

  ## show the out put of the each batch

  predicted = torch.max(outputs.data,1)[1]

  ## append in a prediction list

  predictions.append(outputs)

  total_case+=labels.size(0)  ## how you how many labels are there or how many test case

  correct_prediction += (predicted == labels).sum().item()

print("the testing set accuracy of the network :  {} %".format(100 * correct_prediction/total_case))
import torch

import torch.nn.functional as F

## make ramdom 28x28 image for classification

image_quantity = 10

chanel = 1  ## grayscale image

height = 28

weight = 28



image = torch.rand(image_quantity,chanel,height,weight)
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)



# Convolve the image with the filters 

output_feature = conv_filters(image)

print(output_feature.shape)
## example of adding max pooling and avg pooling in image

import torch.nn

im = torch.Tensor([[[[3,1,3,5],[6,0,7,9],[3,2,1,4],[0,2,4,3]]]])
max_pooling = torch.nn.MaxPool2d(2)

output_feature = max_pooling(im)

print(output_feature)
avg_pooling = torch.nn.AvgPool2d(2)

output_feature = avg_pooling(im)

print(output_feature)
class ConvolutionalNet(nn.Module):



  def __init__(self):

    super(ConvolutionalNet,self).__init__()

    ## remember the in_channels na dout channels are random

    ## this properties will be initialized when the class is initialized

    self.conv1 = nn.Conv2d(in_channels=1,out_channels=5,kernel_size=3,padding=1)

    self.conv2 = nn.Conv2d(in_channels=5,out_channels=10,kernel_size=3,padding=1)

    self.relu = nn.ReLU()

    # instantiate the mx pooling layer

    self.pool = nn.MaxPool2d(2,2)

    ## so the final size will be 28/2*2 (2 by 2 pooling) = 7

    self.fc = nn.Linear(7*7*10,10)

  

  ## apply the forward layer

  def forward(self,x):

    x = self.relu(self.conv1(x))

    x = self.pool(x)

    x = self.relu(self.conv2(x))

    x = self.pool(x)

    x = x.view(-1,7*7*10)

    return self.fc(x)

net = ConvolutionalNet()
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr=3e-4)
e = []  ## store the apochs

l =[] ## store the loss



for epoch in range(30):

  ## for 30 iteration

  ## we loop through the data

  for data in trainloader:

    inputs,labels = data  ## list unpacking

    ## you dont need to flat it we are not deefing it to the FC layer directly



    ## zero gradient so it wont get ata from preious iteration

    optimizer.zero_grad()

    ## make a forward pass

    outputs = net(inputs)

    loss = criterion(outputs,labels)

    loss.backward() ## back propagation

    optimizer.step() ## replace the weight

  print ("EPOCHS: {} LOSS {} ".format(epoch,loss))

  e.append(epoch)

  l.append(loss)

import matplotlib.pyplot as plt

plt.plot(e, l, 'g', label='Training loss')

#plt.plot(epochs, loss_val, 'b', label='validation loss')

plt.title('Training  loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
import numpy as np

### making prediction

correct_prediction = 0

predictions = []

total_case = 0

net.eval() ### before prediction you need to set the evaluation mode

for data in testloader:

  ### list unpacking

  inputs,labels = data

  outputs = net(inputs)

  ## show the out put of the each batch

  predicted = torch.max(outputs.data,1)[1]

  ## append in a prediction list

  predictions.append(outputs)

  total_case+=labels.size(0)  ## how you how many labels are there or how many test case

  correct_prediction += (predicted == labels).sum().item()

print("the testing set accuracy of the network :  {} %".format(100 * correct_prediction/total_case))
## smae network in encapsulated way



class Net(nn.Module):

  def __init__(self):

    super(Net,self).__init__()



    self.features = nn.Sequential(

        nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),

        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),

        nn.MaxPool2d(2,2),

        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),

        nn.ReLU(inplace=True),

        nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),

        nn.MaxPool2d(2,2),

        nn.ReLU(inplace=True)



    )

    self.classifier = nn.Sequential(

        nn.Linear(7*7*40,1024),

        nn.ReLU(inplace=True),

        nn.Linear(1024,2048),  ## input neyron and output neuron

        nn.ReLU(inplace=True),

        nn.Linear(2048,10)

    )



  def forward(self,x):

    x = self.features(x)

    x = x.view(-1,7,7*40)

    x = self.classifier(x)

# Shuffle the indices

indices = np.arange(60000)

np.random.shuffle(indices)



# Build the train loader

train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist', download=True, train=True,

                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),

                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))



# Build the validation loader

val_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist', download=True, train=True,

                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),

                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:60000]))
# Instantiate the network

model = Net()



# Instantiate the cross-entropy loss

criterion = nn.CrossEntropyLoss()



# Instantiate the Adam optimizer

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
class Net(nn.Module):

    def __init__(self):

        

        # Define all the parameters of the net

        self.classifier = nn.Sequential(

            nn.Linear(28*28, 200),

            nn.ReLU(inplace=True),

            nn.Dropout(.5),

            nn.Linear(200,500),

            nn.ReLU(inplace=True),

            nn.Linear(500,10))

    def forward(self,x):

      return self.classifier(x)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        

        # Implement the sequential module for feature extraction

        self.features = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(10),

            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),

            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(20))

        

        # Implement the fully connected layer for classification

        self.fc = nn.Linear(in_features=7*7*20, out_features=10)
# Import the module

import torchvision



# Download resnet18

model = torchvision.models.resnet18(pretrained=True)



# Freeze all the layers bar the last one

for param in model.parameters():

    param.requires_grad = False



# Change the number of output units

model.fc = nn.Linear(512, 7)

## then apply the network in your image