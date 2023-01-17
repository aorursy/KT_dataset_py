import torch
torch.__version__
x = torch.empty([5, 3])
print(x)
y = torch.empty(5, 3)
print(y)
a = torch.rand(5, 4)
print(a)
z = torch.zeros(5, 3, dtype=torch.long)
print(z)
b = torch.tensor([5.09, 3])
print(b)
c = torch.randn_like(z, dtype=torch.float32)
print(c)
c = c.new_ones(5, 6, dtype=torch.double)
print(c)
print(c.size())
d = torch.randn(5, 6, dtype=torch.double)
print(c + d)
e = torch.randn(5, 6, dtype = torch.float)
print(c + e)
print(torch.add(c, d))
result = torch.empty(5, 6, dtype=torch.double)
torch.add(c, d, out=result)
print(result)
e = d
e.add_(c)
print(e)
print(e[:, 1])
x = torch.randn(4, 5)
y = x.view(20)
z = x.view(-1, 2, 2)
print(x, "\n", x.size())
print(y, "\n", y.size())
print(z, "\n", z.size())
print(e.numpy())
a = torch.ones(2, 3)
b = a.numpy()
print(b)
a = a.add_(1)
print(a)
print(b)
import numpy as np
a = np.ones((5, 2))
b = torch.from_numpy(a) ## This is the command
print(b)
np.add(a, 3, out = a)
print(a)
print(b)

x = torch.ones(2, 3, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z)
print(out)
out.backward()
print(x.grad)
import torch
import torch.nn as nn
import torch.nn.functional as F
# A new class Net is created. It inherits from the nn.Module class

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # super initializes a class of the superclass nn.Module
        #The following attributes are added to the class instance during the initialization
        
        # The convolutional layers are defined. Note that these are all functions but are defined like attributes
        self.conv1 = nn.Conv2d(1, 6, 5) #input channels = 1, output channels = 1, window size = 5
        self.conv2 = nn.Conv2d(6, 16, 5) #input channels = 6, output channels = 16, window size = 5
        
        # The fully connected layers are defined
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #input nodes = 16 * 5 * 5 (since each of the 16 channels are of the size 5 * 5), output = 120
        self.fc2 = nn.Linear(120, 84) #input = 120 output = 84
        self.fc3 = nn.Linear(84, 10) #last layer: input = 84, output = 10
        
        
    def forward(self, x):
        """This method defines the forward pass of the neural network"""
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # The first convolutional layer with all the max pooling and relu layers: there is a 2X2 max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # The second conv layer: 2 X 2 max pooling
        x = x.view(-1, self.num_flat_features(x)) # The reshape (flattening) layer which changes the shape of the tensor from that of a multichannel 2D tensor to a 1D tensor
        x = F.relu(self.fc1(x)) # Fully connected layer with activation
        x = F.relu(self.fc2(x)) # Fully connected layer with activation
        x = self.fc3(x) # Output layer
        
        return x
    
    def num_flat_features(self, x):
        """This method helps get the shape of the individual comeponents of the tensor"""
        size = x.size()[1:] # The shape of the tensor without the first dimension which defines the number of samples
        num_features = 1 # the shapes are are multiplied together to give the number of features
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)
params = list(net.parameters())
print(len(params))
for i in range(len(params)):
    print(params[i].size())
inp = torch.randn(1, 1, 32, 32)
out = net(inp)
print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))
out = net(inp)
target = torch.randn(10)
target = target.view(1, -1)
criteria = nn.MSELoss()

loss = criteria(out, target)
print(loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
net.zero_grad() # clearing all the gradients

print("Conv 1 bias before backward()")
print(net.conv1.bias.grad)

loss.backward()
print("Conv 2 bias after backward()")
print(net.conv1.bias.grad)
learning_rate = 0.001
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)

optimizer.zero_grad()
out = net(inp)
loss = criteria(out, target)
loss.backward()
optimizer.step()
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train = False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = 4, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import matplotlib.pyplot as plt
import random
def plot_figs(num_figs = 10):
    fig, axes = plt.subplots(1, num_figs, figsize=(1 * num_figs, 1))
    num_train = len(trainset.train_labels)
    random_nums = random.sample(range(num_train), num_figs)
    for i in range(num_figs):
        axes[i].imshow(trainset.train_data[random_nums[i]])
        axes[i].set_title(classes[trainset.train_labels[random_nums[i]]])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()
plot_figs(10)
# before defining the convnet we need to figure out the shape of the inputs
trainset.train_data[0].shape
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = ConvNet()
# Define a loss function and an optiizer

import torch.optim as optim

criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
# Train the network

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = criteria(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print("[{}, {}], Loss: {}".format(epoch + 1, i + 1, running_loss / 2000.))
            running_loss = 0.0
print("Finished Training")
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
import numpy as np

dataiter = iter(testloader)
images, labels = dataiter.next()
outputs = net(images)
_, predicted = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images))
for i in range(4):
   print("Actual {}\tPredicted: {}".format(classes[labels[i]], classes[predicted[i]]))
