import torch
import numpy as np
#sigmoid
def activation(x):
    return 1/(1 + torch.exp(-x))
torch.manual_seed(7)
features = torch.randn((1, 5))
weights  = torch.randn_like(features)
bias     = torch.randn((1, 1))
#y :predicted value
y = activation(torch.matmul(features, torch.transpose(weights,0,1)) + bias)
#or we can also do it antoher way as it is done in lecture
#y2 = activation(torch.sum(features * weights) + bias)
#y3 = activation((features * weights).sum() + bias)
#print(y, y2, y3)
#torch.mm() and torch.matmul() both can be used for matrix multiplication
#torch.matmul() supports broadcasting thus may give some funny output if the dimensions are uneven. 
#So prefer using torch.mm()
# <tensor>.shape  : to view the shape of tensor
# <tensor>.reshape(a,b) : to change the shape of the tensor to a new shape 
#                    but sometimes it creates a duplicate tensor instead of replacing the current tensor.
#                    thus memory inefficient.
# <tensor>.resize_(a,b) : changes the shape but if new size contains less element than the original size then it removes
#                    element. Causing the loss of data.
# <tensor>.view(a,b) : changes the shape and throws error is new shape contains less number of elements.(preffered)
# another method to compute y using above details
y = activation(torch.mm(features, weights.view(5,1)) + bias)
# simple neural network 
torch.manual_seed(7)

features = torch.randn((1,3))

n_input = features.shape[1]
n_hidden = 2
n_output = 1

# weights
W1 = torch.randn(( n_input, n_hidden))
W2 = torch.randn((n_hidden, n_output))
#biases
b1 = torch.randn((1, n_hidden))
b2 = torch.randn((1, n_output))
Z1 = activation(torch.mm(features, W1) + b1)
Z2 = activation(torch.mm(Z1, W2) + b2)
Z2
# create a numpy array
a = np.random.randn(3,4)
print(a)
# create a torch tensor from numpy array.
b = torch.from_numpy(a)   
print(b)
# numpy version of b
b.numpy()
b.mul_(2)
# both numpy and tensor version are shared with each other thus if you make changes to one of them 
# other one also changes.
print(a)
import matplotlib.pyplot as plt
import torch
import numpy as np
import helper
import os
import pandas as pd
from tqdm import tqdm
import torchvision

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import transforms, datasets

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

plt.tight_layout()
print(os.listdir('../input/'))
class MNIST_dataset(Dataset):
    def __init__(self,filepath,transform=None):
        self.data = pd.read_csv(filepath)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28,28,1))   # (H,W,C)
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])
train_dataset = MNIST_dataset('../input/digit-recognizer/train.csv', transform)
data_generator = DataLoader(train_dataset, batch_size=64, shuffle=True)
data_iterator = iter(data_generator)
print(type(data_iterator))

images, labels = next(data_iterator)
print(images.size(), labels.size())
grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());
# network
batch_size = images.shape[0]
input_size = 28*28

input_layer = images.view((batch_size,-1))

W1 = torch.randn((784, 256))
W2 = torch.randn((256, 10))

b1 = torch.randn((256))
b2 = torch.randn((10))

A1 = activation(torch.mm(input_layer,W1) + b1)
A2 = torch.mm(A1, W2) + b2
print(A2.shape)
print(A2)
#softmax function

def softmax(A):
    numerator = torch.exp(A)
    denominator = torch.sum(numerator,dim=1).view(A.shape[0],1)
    return(numerator/denominator)
    
probabilities = softmax(A2)
print(probabilities.shape)
print(torch.sum(probabilities,dim=1))
print(probabilities.shape)
class Classifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x), dim = 1)
        
        return x
        
model = Classifier()
model
# weights and biases are initialized automatically for us. We can access them using the following statements.
print(model.fc1.weight)
print(model.fc1.bias)
# for custom initialization we need to get the actual tensors and then fill it with custom values.
model.fc1.weight.data.fill_(0)             # to fill it with zeros
model.fc1.weight.data.normal_(std=0.01)    # samole with random normal with std 0.01
data_iterator = iter(data_generator)
images, labels = data_iterator.next()

images.resize_(images.shape[0], 1, 784)

img_idx = 0
ps = model.forward(images[img_idx,:])
img = images[img_idx]

view_classify(img.view(1, 28, 28), ps)
# in Pytorch convention is to assign loss to a variable called criterion.
# to actually calculate the loss we first define the criterion and then pass the output of the network to correct the labels.
# the criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
# the input is expected to contain scores for each class.
# This means that we need to pass raw output of our network into the loss, not the output of the softmax.
# this raw output is called as logits or scores.
# We are going to  create a same network again but using different method.

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128,64),
                      nn.ReLU(),
                      nn.Linear(64,10))  # here we don't use softmax so that
                                         # we could directly input the value to loss function
criterion = nn.CrossEntropyLoss()

images, labels = next(iter(data_generator))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels)

print(loss)
    
# as said in the tutorial the instructor finds it more convenient to build the model with log softmax output.
# Then we can get actual probabilities by taking the exponential.
# With log_softmax output, we want to use the negative log likelihood loss, nn.NLLLoss
# rewriting the above network again
model = nn.Sequential(nn.Linear(784, 128),
                       nn.ReLU(),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Linear(64,10),
                       nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(data_generator))
images = images.view(images.shape[0], -1)

logits = model(images)

loss = criterion(logits, labels)
print(loss)
# autograd is the torch module which calculates the gradient of tensors.
# Autograd works by keeping track of the operation performed on tensors and then going backwards through those tensors,
# calculating the gradients along the way.
# To make sure Pytorch keeps track of operations on tensors and calculate gradients, we need to set  
# requires_grad = True on tensor,  we can also do it at the time of creation by using x.requires_grad_(True)

# we can turn off the gradeint for a block of code with torch.no_grad() context

# we can turn off the gradients all togetherby using
# torch.set_grad_enabled(True| False).

# gradients are computed with respect to some variable z using z.backward()
x = torch.randn(2,2, requires_grad=True)
print(x)
y = x ** 2
print(y)
print(y.grad_fn)
z = y.mean()
print(z)
print(x.grad)
z.backward()
print(x.grad)
print(x/2)
# build the feed-forward neural network
%time
model = nn.Sequential(nn.Linear(784, 128),
                       nn.ReLU(),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Linear(64,10),
                       nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
images, labels = next(iter(data_generator))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)
print("Before backward pass: \n", model[0].weight.grad)
loss.backward()
print("After backward pass: \n", model[0].weight.grad)
optimizer = optim.SGD(model.parameters(), lr=0.01)
# training has four major parts
    # forward pass throught the network
    # use network output to calculate the loss
    # perform backward pass using loss.backward()
    # take a step with optimizer to update weights.
    
# When you do multiple backward passes with the same parameters, the gradients are accumulated. 
# this means that you need to zero the gradients on each training pass or 
# you'll retain the gradients from previous training batches.
# lets do training for real
model = nn.Sequential(nn.Linear(784, 128),
                       nn.ReLU(),
                       nn.Linear(128, 64),
                       nn.ReLU(),
                       nn.Linear(64,10),
                       nn.LogSoftmax(dim=1))
model =model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 1
for epoch in range(epochs):
    running_loss = 0
    tick = time.time()
    for images, labels in data_generator:
        
        images = images.view(images.shape[0], -1)
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        
        logps = model.forward(images)                 # forward pass
        loss = criterion(logps, labels)       # loss
        loss.backward()                       # backward pass
        optimizer.step()                      # update to the parameters
        
        running_loss += loss.item()
    else:
        print("Training loss: {} Time: {}".format(running_loss/len(data_generator), time.time()-tick))
images, labels = next(iter(data_generator))
img = images[0].view(1,784)

with torch.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim = 1)
view_classify(img.view(1,28,28),ps)
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
df.head()
class Fashin_MNIST_dataset(Dataset):
    def __init__(self,filepath, transform=None):
        self.data = pd.read_csv(filepath)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data.iloc[index,1:].values.astype(np.uint8).reshape((28,28,1))
        label = self.data.iloc[index,0]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image,label
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = Fashin_MNIST_dataset('../input/fashionmnist/fashion-mnist_train.csv',transform)
dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
images, labels = next(iter(dataloader))

grid = torchvision.utils.make_grid(images)

plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.title(labels.numpy());
model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64,10),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 10
for epoch in range(epochs):
    running_loss = 0
    for images, labels in dataloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        print("Training loss: {}".format(running_loss/len(dataloader)))
        
    

# test the output of network
images, labels = next(iter(dataloader))
img = images[0]
img = img.view(1, 784)

ps = model.forward(img)
ps = torch.exp(model(img))

view_classify(img,ps,version="Fashion")
# so here we'll again get the data and divide it into train and test set. 
# But wait here it is already divided into two different csv files. So no need. Just get data.
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# here if I would have downloaded dataset using the inbuilt function then we can separate the train and test data using the 
# train = True for train data and train = False for test data.
# Then we also need not to write our own class for Dataset instead used the inbuilt class called datasets.FasionMNIST()

train_dataset = Fashin_MNIST_dataset('../input/fashionmnist/fashion-mnist_train.csv', transform=transform)
trainloader = DataLoader(train_dataset,batch_size=64, shuffle=True)

test_dataset = Fashin_MNIST_dataset('../input/fashionmnist/fashion-mnist_test.csv', transform=transform)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# lets create Neural network
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x
model = Classifier()
images, labels = next(iter(trainloader))

ps = torch.exp(model(images))
print(ps.shape)
top_p, top_class = ps.topk(1,dim=1)
print(top_class[:10,:])
equals = top_class == labels.view(*top_class.shape)
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100}%')
use_gpu = False#torch.cuda.is_available()
print(use_gpu)
from tqdm import tqdm
import time
torch.device("cuda:0")
# so now lets build the validation block for the above classfier.
model = Classifier()
# use GPU for faster calculation
if use_gpu:
    model = model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 11
steps = 0

train_losses, test_losses = [],[]

for epoch in range(epochs):
    running_loss = 0
    tick = time.time()
    for images, labels in tqdm(trainloader):
        if use_gpu:
            images = images.type(torch.FloatTensor)
            images = images.cuda()
            labels = labels.type(torch.LongTensor)
            labels = labels.cuda()
        
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss +=  loss.item()
    
    else:
        accuracy = 0
        test_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(testloader):
                if use_gpu:
                    images = images.type(torch.FloatTensor)
                    images = images.cuda()
                    labels = labels.type(torch.LongTensor)
                    labels = labels.cuda()
                    
                logps = (model(images))
                test_loss += criterion(logps, labels)
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy = torch.mean(equals.type(torch.FloatTensor))
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
              "Time taken: {}".format(time.time()-tick))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
# In the dropout layer we randomly drop neural units during training. 
# This helps us prevent overfitting of model.
# so now lets build the validation block for the above classfier.

class Classifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

from tqdm import tqdm
model = Classifier()
# use GPU for faster calculation
if use_gpu:
    model = model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
epochs = 11
steps = 0

train_losses, test_losses = [],[]

for epoch in range(epochs):
    running_loss = 0
    tick = time.time()
    for images, labels in (trainloader):
        if use_gpu:
            images = images.type(torch.FloatTensor)
            images = images.cuda()
            labels = labels.type(torch.LongTensor)
            labels = labels.cuda()
        
        optimizer.zero_grad()
        
        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss +=  loss.item()
    
    else:
        accuracy = 0
        test_loss = 0
        with torch.no_grad():
            # turn on the evaluation mode
            model.eval()
            
            for images, labels in testloader:
                if use_gpu:
                    images = images.type(torch.FloatTensor)
                    images = images.cuda()
                    labels = labels.type(torch.LongTensor)
                    labels = labels.cuda()
                    
                logps = (model(images))
                test_loss += criterion(logps, labels)
                
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        model.train()      # return to the train mode.
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
              "Time taken: {}".format(time.time()-tick))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
# the parameters of the pytorch  network are stored in the model.state_dict().
# it contains the weights and biases matrices for each of our layers.

print(model)
print("\nthe state_dict keys: \n\n", model.state_dict().keys())
# the simplest thing to do is to save the dict using torch.save
torch.save(model.state_dict(), 'checkpoint.pth')
# then we can load them using torch.load()
state_dict = torch.load('checkpoint.pth')
# and to load the state_dict into the network
model.load_state_dict(state_dict)
# so fromt the above statements it is quite evident that we need to have model architecture for weights to load into.
# i.e. if model architecture is wrong then we  would get an error.
# so we can create a dictionary.
"""
checkpoint = {'input_size':784,
              'output_size': 10,
              'hidden_layers': [ each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict()}
torch.save(checkpoint,'checkpoint.pth')
"""
# we could also ease our task by creating a function that could take layers size as the input parameters and do rest.
# for this we would also need to change the __init__() of the Classifier class.

