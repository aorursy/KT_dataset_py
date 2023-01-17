import torch

import numpy as np
x = np.array([-1, -2, 1, 2]) # a numpy array

print(x)

print(type(x))
x = torch.Tensor(x) # now x is a torch Tensor

print(x)
y = torch.Tensor([0,1,2,-1]) # we can initialize some tensor directly from a python list

print(y)
z = x + y # normal tensor operation, the result is a Tensor

print(z)

print(type(z))

print(z.type())
x = torch.Tensor(5,4) 

print(x)

x = torch.zeros(5,4) # As default, a tensor in pytorch is a float tensor (torch.float32)

print(x)

x = torch.ones([5,4], dtype=torch.int64) 

print(x)

x = torch.Tensor([[5,4]]) # Initialize it from python list 

print(x)

print(x.shape)
x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 

print(x)

print(x[:,2])

print(x[1:3,])
x = torch.Tensor(3,4) 

print(x.shape)

x = x.view(2,6)

print(x)

print(x.shape)
from torch.autograd import Variable

import torch.nn.functional as F
a = torch.Tensor([3])

w = Variable(torch.Tensor([2]),requires_grad=True)

b = Variable(torch.Tensor([3]),requires_grad=True)

z = a * w

y = z + b

print(y.data) # data of a variable y is a tensor

print(z.grad_fn)  

print(y.grad_fn)
torch.__version__
a = torch.Tensor([3]) # Normal tensor - multi array

print(a)

print(a.grad_fn)



w = torch.Tensor([2]) # A variable - A tensor which contains other information

w.requires_grad = True



b = torch.Tensor([3])

b.requires_grad = True

z = a * w

y = z + b

print(y)

print(y.data) # data of a variable y is a normal tensor
y.backward() # y = a * w + b
print(b.grad)  

print(w.grad) # w and b are variables
print(z.grad)

print(a.grad) # z and a are normal tensors, no gradient information included
def mean_squared_error(t, y):  # Mean squared error in numpy

    # t: target label

    # y: predicted value

    n, m = y.shape

    return np.sum((y - t)**2) / (n * m)

    

def dMSE_dy(t, y): # Derivative of mean squared error w.r.t y in numpy

    n, m = y.shape

    return 2 * (y-t) / (n * m)



def mean_squared_error_PT(t, y): # Mean squared error - Pytorch version

    # t: target label

    # y: predicted value

    n, m = y.shape

    return torch.sum((y - t)**2) / (n * m)



y = np.random.randn(4,2)

t = np.random.randn(4,2)



# forward test

my_mse = mean_squared_error(t, y)



# warp y, t to be tensors/variable

tt = torch.Tensor(t)

yt = Variable(torch.Tensor(y),requires_grad=True)

pt_mse = mean_squared_error_PT(tt,yt)



print("My Mean of Squared Error: " + str(my_mse))

print("Pytorch Mean of Squared Error: " + str(pt_mse))



# backward test

print("===============================")

my_dmse_dy = dMSE_dy(t, y)



pt_mse.backward() # Pytorch will calculate it for us



print("My Derivatives: " + str(my_dmse_dy))

print("Pytorch Derivative: " + str(yt.grad))
def sigmoid(z):

    return 1 / (1 + np.exp(-z))



def sigmoid_prime(z):

    return sigmoid(z) * (1 - sigmoid(z))



z = np.random.randn(4,2)

z_prime = sigmoid_prime(z)



zt = torch.Tensor(z)

zt.requires_grad = True

pt_sigmoid = torch.sigmoid(zt)

print("Sigmoid Prime: " + str(z_prime))



pt_sigmoid.backward() # THIS WILL RAISE AN ERROR

print("Pytorch Sigmoid Prime: " + str(zt.grad))



def sigmoid(z):

    return 1 / (1 + np.exp(-z))



def sigmoid_prime(z):

    return sigmoid(z) * (1 - sigmoid(z))



z = np.random.randn(4,2)

z_prime = sigmoid_prime(z)



zt = torch.Tensor(z)

zt.requires_grad = True

pt_sigmoid = torch.sigmoid(zt)

print("Sigmoid Prime: " + str(z_prime))



# This works because L = sum(all sigmoid(zt_i)) 

# => dL/dzt_i = dL/dsigmoid * dsigmoid/dzt_i = dsigmoid/dzt_i 

# (dL/dsigmoid = 1)

pt_sigmoid.sum().backward() 

print("Pytorch Sigmoid Prime: " + str(zt.grad))



from torch import nn

x = Variable(torch.randn(2,5), requires_grad=True)

print (x)

lin = nn.Linear(5, 3) # a linear transformation
z = lin(x) # forward the data x to the linear transformation

print(z)
y = torch.sigmoid(z)

print(y.grad_fn)
y = torch.tanh(z)

t = y.sum()

t.backward()

print(x.grad) # dt/dx
# Input x, output y

x = Variable(torch.randn(5, 3))

y = Variable(torch.randn(5, 2))



# Build a linear layer

lin = nn.Linear(3, 2)

print ('w: ', lin.weight)

print ('b: ', lin.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(lin.parameters(), lr=0.01) # do updates on weights of the linear layer
z = lin(x)
loss = criterion(z, y)

print('loss: ', loss.data.item())
loss.backward()



print ('dL/dw: ', lin.weight.grad) 

print ('dL/db: ', lin.bias.grad)
# update weights one time

optimizer.step()



# Equal to:

# lin.weight.data.sub_(0.01 * lin.weight.grad.data)  (w = w - 0.01 * dloss/dw)

# lin.bias.data.sub_(0.01 * lin.bias.grad.data)  (bias = bias - 0.01 * dloss/dbias)



# Print out the loss after optimization.

z = lin(x)

loss = criterion(z, y)

print('loss after 1 step optimization: ', loss.data.item()) # The loss should be smalller than in the step 2 above

# Those weights should be changed

print ('w: ', lin.weight)

print ('b: ', lin.bias)
import torch

from torch import nn

from torch.autograd import Variable

import torch.nn.functional as F



# Build a sigmoid layer

class MySigmoid(nn.Module): # inheriting from nn.Module!

    

    def __init__(self, input_size, output_size):

        # just inherit the init of the module

        super(MySigmoid, self).__init__()

        

        # Your lego building here

        # NOTE: The non-linearity sigmoid doesn't have any parameter so we do not put it here

        # the layer's parameters will be all parameters from learnable modules building the layer

        self.linear = nn.Linear(input_size, output_size)

        

    def forward(self, x):

        # Pass the input through the linear layer,

        # then pass that through the sigmoid.

        return torch.sigmoid(self.linear(x))

    

torch.manual_seed(1234)



# Input x, output y, test on t

x = Variable(torch.randn(5, 3))

y = Variable(torch.randn(5, 2))

t = Variable(torch.zeros(5, 3))

# WHAT WILL HAPPEN IF WE DO THE FOLLOWING??? TRY IT YOURSELF!!!

#x = Variable(torch.Tensor(5,3))

#y = Variable(torch.Tensor(5,2))

#t = Variable(torch.Tensor(5,3)) 



# Create the network comprise of only that sigmoid layer

model = MySigmoid(3,2)



# Define loss and learning method

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



# Train many times

def train(x, y, epoch):

    # Forward pass is different from train to test (e.g. dropout, batch norm)

    # So it's a good habit if you explicitly speak out

    model.train()

    for i in range(epoch):

        # Pytorch accumulates gradients.  We need to clear them out before a new batch

        model.zero_grad()

        

        # Forward pass and calculate errors

        z = model(x) # This is a shortcut for calling model.forward(x)

        loss = criterion(z, y)  # This is a shortcut for calling criterion.forward(z, y)

        print('Epoch {}: Loss: {}'.format(i+1, loss.data.item()))

              

        # Do backward pass and update weights

        loss.backward()

        optimizer.step()



# Test

def test(t):

    # Say that we do testing

    model.eval()

    return model(t) # Do the forward pass on the new input



train(x, y, 50)



# Do the test:

print("==================")

print("Apply on new data:")

print(t.data)

print("Result:")

print(test(t).data)



from torchvision import datasets, transforms



# Download MNIST traing data to directory ../Data for the first time

# Read the training dataset from directory ../Data if there was such data

# Convert data to Tensor and do normalization (mean, std)

# transforms.Compose: compose of many image transformations

mnist = datasets.MNIST(root='../Data',

                       train=True,

                       download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1,), (0.4,))])

                      )



# Examine data: select one data pair (the first data instance)

image, label = mnist[0]

print (image.size())

print (label)



# Use DataLoader to easily divide a dataset into train/valid/test with mini-batches and shuffle...

train_loader = torch.utils.data.DataLoader(mnist, batch_size=32, shuffle=True)



# Actual usage of data loader is as below.

for images, labels in train_loader:

    # Your training code will be written here

    pass
import torch

import numpy as np

from torch.utils.data import Dataset



class MyMNIST(Dataset):

    """My custom MNIST dataset."""

    

    TRAIN = 0

    VALID = 1

    TEST = 2



    def __init__(self, csv_file, data=TRAIN, one_hot=False, transforms=None):

        """

        Args:

            csv_file (string): Path to the csv file

            root_dir (string): Directory with all the images.

            one_hot (boolean): Do the one-hot encoding on the labels or not

            transform (callable, optional): Optional transform to be applied on a sample.

        """

        

        # load from csv_file (full path)

        all_data = np.genfromtxt(csv_file, delimiter=",", dtype="uint8")

    

        # There are 5000 instances (5000 lines in csv file)

        # First 3000 lines will be training set

        # The next 1000 lines will be validation set

        # The last 1000 lines will be the test set

        # You can modify this

        train, test = all_data[:4000], all_data[4000:]

        train, valid = train[:3000], train[3000:]

        

        # We can implement the shuffle easily as follows, 

        # but we would like to use this utility from DataLoader

        # from numpy.random import shuffle

        # shuffle(train)

        # ...

        

 

        # The label of the digits is always the first fields

        if data == self.TRAIN:

            self.input = train[:, 1:]

            self.label = train[:, 0]

        elif data == self.VALID:

            self.input = valid[:, 1:]

            self.label = valid[:, 0]

        else:

            self.input = test[:, 1:]

            self.label = test[:, 0]

    

        

        # One-hot encoding:

        if one_hot:

            self.label = np.array(map(one_hot_encoding, self.label))

           

        # Apply the transformations:

        for i, transfunction in enumerate(transforms):

            self = transfunction(self)

        

    def __len__(self):

        return len(self.input)



    def __getitem__(self, idx):

        return self.input[idx], self.label[idx]





# One-hot encoding

def one_hot_encoding(idx):

    one_hot_array = np.zeros(10)

    one_hot_array[idx] = 1

    return one_hot_array



####################################################

# We can also implement the transformation ourselves



# This normalizer only works for our MNIST data: divide the pixel values to 255

# We can implement other general normalizers, e.g. standard normalizer, by:

# mean = np.mean(data.input)

# std = np.std(data.input)

# data.input = (data.input - mean)/std



class Normalizer(object):



    def __call__(self, data):

        data.input = 1.0*data.input/255

        return data



# This convert from numpy arrays of data to pytorch tensors

class ToTensor(object):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, data):

        data.input = torch.from_numpy(data.input).float()

        data.label = torch.from_numpy(data.label).long()

        return data
%matplotlib inline

import matplotlib.pyplot as plt



# Some helper function to draw the image

def plot(img):

    plt.imshow(img.view(28,28).numpy())

    

    

my_mnist = MyMNIST(csv_file='../input/mnistseven/mnist_seven.csv',

                data=MyMNIST.TEST,

                one_hot=False,

                transforms=[Normalizer(),

                            ToTensor()])





# Examine data: select one data pair

image, label = my_mnist[3]

plot(image)

print("Label: " + str(label))



train_loader = torch.utils.data.DataLoader(my_mnist, batch_size=32, shuffle=True)



# Actual usage of data loader is as below.

for images, labels in train_loader:

    # Your training code will be written here

    pass

import torch

from torch import nn

from torch.autograd import Variable

import torch.nn.functional as F

from torch.utils.data import Dataset



# Define custom network: 3 layers

class MyFNN(nn.Module):

    

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):

        super(MyFNN, self).__init__()

        

        self.model = nn.Sequential()  # Container of modules

        

        # Dynamically build network

        for i, hidden_size in enumerate(hidden_sizes):

            self.model.add_module("linear" + str(i+1), nn.Linear(input_size, hidden_size))

            self.model.add_module("sigmoid" + str(i+1), nn.Sigmoid())

            self.model.add_module("dropout" + str(i+1), nn.Dropout(dropout))

            input_size = hidden_size

        

        self.model.add_module("linear" + str(len(hidden_sizes)+1), nn.Linear(input_size, output_size, bias=False))

        # If you use CrossEntropyLoss instead of NLLLoss, do not need to add this LogSoftmax layer

        # self.model.add_module("logsoftmax", nn.LogSoftmax())  

    

    def forward(self, x):

        return self.model.forward(x)



    

batch_size = 64

torch.manual_seed(1234)



# Two-hidden-layer Feedforward Network with default dropout

net = MyFNN(784, [100, 50], 10)





#criterion = nn.NLLLoss()  # Use Negative Log Likelihood

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)



def train(epoch, train_data):

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    

    for i in range(epoch):

        num_batch = len(train_data)//batch_size + 1

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.view(-1, 28*28)

            

            #Zero grads before each optimizing step

            optimizer.zero_grad()

            

            # Forward pass and calculate errors

            y = net(images)

            loss = criterion(y, labels)

            if (batch_idx+1) % 100 == 0:

                print('Iteration {0:d}/{1:d}: Loss: {2:.2f}'.format(batch_idx+1, num_batch, loss.data.item()))

            # Do backward pass and update weights

            loss.backward()

            optimizer.step()

        print('Epoch {}: Loss: {}'.format(i+1, loss.data.item()))





# Train

from torchvision import datasets, transforms

train_data = datasets.MNIST(root='../Data', 

                            train=True, 

                            transform=transforms.ToTensor(),  

                            download=True)

net.train()    

train(5, train_data)





# Test

test_data = datasets.MNIST(root='../Data', 

                            train=False, 

                            transform=transforms.ToTensor(),  

                            download=True)



net.eval()



correct = 0

total = 0 # accumulated loss over mini-batches

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)



for images, labels in test_loader:

    images = images.view(-1, 28*28)

    y = net(images)

    _, predicts = torch.max(y.data, 1)

    total += labels.size(0)

    correct += (predicts == labels).sum()

   

print('Accuracy on the test images: %d %%' % (100 * correct / total))

    
import torch

from torch import nn

from torch.autograd import Variable

import torch.nn.functional as F

from torch.utils.data import Dataset



# Define custom network: 3 layerstorch.max

class MyFNN(nn.Module):

    

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):

        super(MyFNN, self).__init__()

        

        self.model = nn.Sequential()  # Container of modules

        

        # Dynamically build network

        for i, hidden_size in enumerate(hidden_sizes):

            self.model.add_module("linear" + str(i+1), nn.Linear(input_size, hidden_size))

            self.model.add_module("sigmoid" + str(i+1), nn.Sigmoid())

            self.model.add_module("dropout" + str(i+1), nn.Dropout(dropout))

            input_size = hidden_size

        

        self.model.add_module("linear" + str(len(hidden_sizes)+1), nn.Linear(input_size, output_size, bias=False))

        # If you use CrossEntropyLoss instead of NLLLoss, do not need to add this LogSoftmax layer

        # self.model.add_module("logsoftmax", nn.LogSoftmax())  

    

    def forward(self, x):

        return self.model.forward(x)





batch_size = 64

torch.manual_seed(1234)

    

# Two-hidden-layer Feedforward Network with default dropout

net = MyFNN(784, [100, 50], 10)





#criterion = nn.NLLLoss()  # Use Negative Log Likelihood

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)



def train(epoch, train_data):

    train_loader = torch.utils.data.DataLoader(my_mnist, batch_size=batch_size, shuffle=True)

    

    for i in range(epoch):

        num_batch = len(train_data)//batch_size + 1

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.view(-1, 28*28)

            

            #Zero grads before each optimizing step

            optimizer.zero_grad()

            

            # Forward pass and calculate errors

            y = net(images)

            loss = criterion(y, labels)

            if (batch_idx+1) % 100 == 0:

                print('Iteration {0:d}/{1:d}: Loss: {2:.2f}'.format(batch_idx+1, num_batch, loss.data.item()))

            # Do backward pass and update weights

            loss.backward()

            optimizer.step()

        print('Epoch {}: Loss: {}'.format(i+1, loss.data.item()))





# Train

from torchvision import transforms

train_data = MyMNIST(csv_file='../input/mnistseven/mnist_seven.csv',

                    data=MyMNIST.TRAIN,

                    one_hot=False,

                    transforms=[Normalizer(),

                                ToTensor()])





net.train()    

train(35, train_data)





# Test

test_data = MyMNIST(csv_file='../input/mnistseven/mnist_seven.csv',

                    data=MyMNIST.TEST,

                    one_hot=False,

                    transforms=[Normalizer(),

                                ToTensor()])



net.eval()



correct = 0

total = 0 # accumulated loss over mini-batches

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)



for images, labels in test_loader:

    images = images.view(-1, 28*28)

    y = net(images)

    _, predicts = torch.max(y.data, 1)

    total += labels.size(0)

    correct += (predicts == labels).sum()

   

print('Accuracy on the test images: %d %%' % (100 * correct / total))

    
cuda = torch.device('cuda')     # Default CUDA-enable GPU device

cuda1 = torch.device('cuda:1')  # The second available GPU device (0-indexed)



a = torch.tensor([1., 2.], device=cuda)

print(a)



b = torch.tensor([1., 2.])

print(b)



b1 = b.cuda() # Convert CPU tensor to GPU tensor (and store it on the default)

print(b1)



# Convert tensor in one device to another device

b2 = b.to(device=cuda)

print(b2)



# The following command will raise an error since a is a GPU tensor and b is a CPU tensor

#c1 = a + b

#print(c1)



# Convert tensor in one device to another device

c1 = a.to(device=torch.device('cpu')) + b

print(c1)

import torch

from torch import nn

from torch.autograd import Variable

import torch.nn.functional as F

from torch.utils.data import Dataset



# Define custom network: 3 layerstorch.max

class MyFNN(nn.Module):

    

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):

        super(MyFNN, self).__init__()

        

        self.model = nn.Sequential()  # Container of modules

        

        # Dynamically build network

        for i, hidden_size in enumerate(hidden_sizes):

            self.model.add_module("linear" + str(i+1), nn.Linear(input_size, hidden_size))

            self.model.add_module("sigmoid" + str(i+1), nn.Sigmoid())

            self.model.add_module("dropout" + str(i+1), nn.Dropout(dropout))

            input_size = hidden_size

        

        self.model.add_module("linear" + str(len(hidden_sizes)+1), nn.Linear(input_size, output_size, bias=False))

        # If you use CrossEntropyLoss instead of NLLLoss, do not need to add this LogSoftmax layer

        # self.model.add_module("logsoftmax", nn.LogSoftmax())  

    

    def forward(self, x):

        return self.model.forward(x)



batch_size = 64

torch.manual_seed(1234)



# Two-hidden-layer Feedforward Network with default dropout

net = MyFNN(784, [100, 50], 10).cuda()





#criterion = nn.NLLLoss()  # Use Negative Log Likelihood

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)



def train(epoch, train_data):

    train_loader = torch.utils.data.DataLoader(my_mnist, batch_size=batch_size, shuffle=True)

    

    for i in range(epoch):

        num_batch = len(train_data)//batch_size + 1

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.view(-1, 28*28).cuda()

            labels = labels.cuda()

            

            #Zero grads before each optimizing step

            optimizer.zero_grad()

            

            # Forward pass and calculate errors

            y = net(images)

            loss = criterion(y, labels)

            if (batch_idx+1) % 100 == 0:

                print('Iteration {0:d}/{1:d}: Loss: {2:.2f}'.format(batch_idx+1, num_batch, loss.data.item()))

            # Do backward pass and update weights

            loss.backward()

            optimizer.step()

        print('Epoch {}: Loss: {}'.format(i+1, loss.data.item()))





# Train

from torchvision import transforms

train_data = MyMNIST(csv_file='../input/mnistseven/mnist_seven.csv',

                    data=MyMNIST.TRAIN,

                    one_hot=False,

                    transforms=[Normalizer(),

                                ToTensor()])





net.train()    

train(35, train_data)





# Test

test_data = MyMNIST(csv_file='../input/mnistseven/mnist_seven.csv',

                    data=MyMNIST.TEST,

                    one_hot=False,

                    transforms=[Normalizer(),

                                ToTensor()])



net.eval()



correct = 0

total = 0 # accumulated loss over mini-batches

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)



for images, labels in test_loader:

    images = images.view(-1, 28*28).cuda()

    labels = labels.cuda()

    y = net(images)

    _, predicts = torch.max(y.data, 1)

    total += labels.size(0)

    correct += (predicts == labels).sum()

   

print('Accuracy on the test images: %d %%' % (100 * correct / total))

    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
import torch

from torch import nn

from torch.autograd import Variable

import torch.nn.functional as F

from torch.utils.data import Dataset



# Define custom network: 3 layerstorch.max

class MyFNN(nn.Module):

    

    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.3):

        super(MyFNN, self).__init__()

        

        self.model = nn.Sequential()  # Container of modules

        

        # Dynamically build network

        for i, hidden_size in enumerate(hidden_sizes):

            self.model.add_module("linear" + str(i+1), nn.Linear(input_size, hidden_size))

            self.model.add_module("sigmoid" + str(i+1), nn.Sigmoid())

            self.model.add_module("dropout" + str(i+1), nn.Dropout(dropout))

            input_size = hidden_size

        

        self.model.add_module("linear" + str(len(hidden_sizes)+1), nn.Linear(input_size, output_size, bias=False))

        # If you use CrossEntropyLoss instead of NLLLoss, do not need to add this LogSoftmax layer

        # self.model.add_module("logsoftmax", nn.LogSoftmax())  

    

    def forward(self, x):

        return self.model.forward(x)



# Get the available device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



batch_size = 64

torch.manual_seed(1234)



# Two-hidden-layer Feedforward Network with default dropout

net = MyFNN(784, [100, 50], 10).to(device)





#criterion = nn.NLLLoss()  # Use Negative Log Likelihood

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)





def train(epoch, train_data):

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    

    for i in range(epoch):

        num_batch = len(train_data)//batch_size + 1

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.view(-1, 28*28).to(device)

            labels = labels.to(device)

            

            #Zero grads before each optimizing step

            optimizer.zero_grad()

            

            # Forward pass and calculate errors

            y = net(images)

            loss = criterion(y, labels)

            if (batch_idx+1) % 100 == 0:

                print('Iteration {0:d}/{1:d}: Loss: {2:.2f}'.format(batch_idx+1, num_batch, loss.data.item()))

            # Do backward pass and update weights

            loss.backward()

            optimizer.step()

        print('Epoch {}: Loss: {}'.format(i+1, loss.data.item()))





# Train

from torchvision import datasets, transforms

train_data = datasets.MNIST(root='../Data', 

                            train=True, 

                            transform=transforms.ToTensor(),  

                            download=True)

net.train()    

train(5, train_data)





# Test

test_data = datasets.MNIST(root='../Data', 

                            train=False, 

                            transform=transforms.ToTensor(),  

                            download=True)



net.eval()



correct = 0

total = 0 # accumulated loss over mini-batches

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)



for images, labels in test_loader:

    images = images.view(-1, 28*28).to(device)

    labels = labels.to(device)

    y = net(images)

    _, predicts = torch.max(y.data, 1)

    total += labels.size(0)

    correct += (predicts == labels).sum()

   

print('Accuracy on the test images: %d %%' % (100 * correct / total))

    