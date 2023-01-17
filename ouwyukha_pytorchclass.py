import torch 
import numpy as np
arr = np.random.randn(3,5)
arr
tens = torch.from_numpy(arr)
tens
another_tensor = torch.LongTensor([[2,4],[5,6]])
another_tensor
random_tensor = torch.randn((4,3))
random_tensor
from torch import FloatTensor
from torch.autograd import Variable
# Define the leaf nodes or input
a = Variable(FloatTensor([4]))

# Define the weights
weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (2, 5, 9, 7)]

# unpack the weights for nicer assignment
w1, w2, w3, w4 = weights

b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
L = (10 - d)
w1
w2
L
L.backward()
print("*(w.r.t) with respect to")
for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(f"Gradient of w{index} w.r.t to L: {gradient}")
# manual GD
learning_rate = 0.01
w1.data.sub_(w1.grad.data * learning_rate)
w2.data.sub_(w2.grad.data * learning_rate)
w3.data.sub_(w3.grad.data * learning_rate)
w4.data.sub_(w4.grad.data * learning_rate)
w4.grad.data
w4.grad.data.zero_()
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
X = np.random.rand(30, 1)*2.0
w = np.random.rand(2, 1)
y = X*w[0] + w[1] + np.random.randn(30, 1) * 0.05
plt.scatter(X, y)
from torch import FloatTensor
from torch.autograd import Variable

W = Variable(torch.rand(1, 1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)

def linear(x):
    return torch.matmul(x, W) + b

Xt = Variable(torch.from_numpy(X)).float()
yt = Variable(torch.from_numpy(y)).float()
fig = plt.figure()
plt.scatter(X, y, color='blue')
pred = linear(torch.from_numpy(X).float())
plt.scatter(X, pred.detach().numpy(), color='red')
plt.legend(['Real', 'Hypothesis'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Acc')
### Code Here
fig = plt.figure()
plt.scatter(X, y, color='blue')
pred = linear(torch.from_numpy(X).float())
plt.scatter(X, pred.detach().numpy(), color='red')
plt.legend(['Real', 'Hypothesis'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Acc')
import torch
import torchvision

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
batch_size_train = 64
batch_size_test = 1000
# transform each data
transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
# load dataset
train_dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('dataset/', train=False, download=True, transform=transform)

# make dataloader for batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_train, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape
import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
fig
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
class Net(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), output_class=10):
        super(Net,self).__init__()
        self.input_shape = input_shape
        self.output_class = output_class
        
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 5)
        
        # number of hidden nodes in each layer (50)
        hidden_1 = 128
        hidden_2 = 128
        # make 1*28*28 = 784
        input_data = np.prod(self.input_shape)
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(15488, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, output_class)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        # flatten image input
        #x = x.view(-1, np.prod(self.input_shape))
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        x = self.flatten(x)
        
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
         # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x
# init Network
network = Net(example_data.shape[1:], 10)
if torch.cuda.is_available():
    network.cuda()
learning_rate = 0.001
momentum = 0.2
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss() #  nn.MSELoss() # nn.SmoothL1Loss() # nn.NLLLoss()
n_epochs = 3
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(1,n_epochs + 1)]
train_acc = []
test_acc = []
log_interval = 100
def train(epoch):
    network.train() # Train mode on
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
    train_acc.append((100. * correct / len(train_loader.dataset)).item())
    print('Train Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(train_loader.dataset), train_acc[-1]))
def test():
    network.eval() # Train mode off 
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = network(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append((100. * correct / len(test_loader.dataset)).item())
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), test_acc[-1]))
    return test_acc[-1]
best = 0
for epoch in range(1, n_epochs + 1):
    train(epoch)
    curr = test()
    if best > curr:
        break
    best = curr
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('loss')
fig = plt.figure()
plt.plot(test_counter, train_acc, color='blue')
plt.plot(test_counter, test_acc, color='red')
plt.legend(['Train Acc', 'Test Acc'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('Acc')
with torch.no_grad():
    if torch.cuda.is_available():
        example_data = example_data.cuda()
    output = network(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0].cpu(), cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.cpu().data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig

