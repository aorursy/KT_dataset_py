import torch 

import torch.nn as nn

import torchvision

import torchvision.transforms as transforms

import torchvision.datasets as dsets

import torch.nn.functional as F

import matplotlib.pylab as plt

import numpy as np

torch.manual_seed(2)



import warnings

warnings.filterwarnings('ignore')
class Net(nn.Module):

    

    # Constructor

    def __init__(self, D_in, H1, H2, D_out):

        super(Net, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)

        self.linear2 = nn.Linear(H1, H2)

        self.linear3 = nn.Linear(H2, D_out)

    

    # Prediction

    def forward(self,x):

        x = torch.sigmoid(self.linear1(x)) 

        x = torch.sigmoid(self.linear2(x))

        x = self.linear3(x)

        return x
class NetTanh(nn.Module):

    

    # Constructor

    def __init__(self, D_in, H1, H2, D_out):

        super(NetTanh, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)

        self.linear2 = nn.Linear(H1, H2)

        self.linear3 = nn.Linear(H2, D_out)

    

    # Prediction

    def forward(self, x):

        x = torch.tanh(self.linear1(x))

        x = torch.tanh(self.linear2(x))

        x = self.linear3(x)

        return x
class NetRelu(nn.Module):

    

    # Constructor

    def __init__(self, D_in, H1, H2, D_out):

        super(NetRelu, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)

        self.linear2 = nn.Linear(H1, H2)

        self.linear3 = nn.Linear(H2, D_out)

    

    # Prediction

    def forward(self, x):

        x = torch.relu(self.linear1(x))  

        x = torch.relu(self.linear2(x))

        x = self.linear3(x)

        return x
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):

    i = 0

    useful_stuff = {'training_loss': [], 'validation_accuracy': []}  

    

    for epoch in range(epochs):

        for i, (x, y) in enumerate(train_loader):

            optimizer.zero_grad()

            z = model(x.view(-1, 28 * 28))

            loss = criterion(z, y)

            loss.backward()

            optimizer.step()

            useful_stuff['training_loss'].append(loss.data.item())

        

        correct = 0

        for x, y in validation_loader:

            z = model(x.view(-1, 28 * 28))

            _, label = torch.max(z, 1)

            correct += (label == y).sum().item()

    

        accuracy = 100 * (correct / len(validation_loader))

        useful_stuff['validation_accuracy'].append(accuracy)

    

    return useful_stuff
train_set = torchvision.datasets.FashionMNIST("./data", download=True, transform=

                                                transforms.Compose([transforms.ToTensor()]))

test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False, transform=

                                               transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)

validation_loader = torch.utils.data.DataLoader(test_set, batch_size=100)
def output_label(label):

    output_mapping = {

                 0: "T-shirt/Top",

                 1: "Trouser",

                 2: "Pullover",

                 3: "Dress",

                 4: "Coat", 

                 5: "Sandal", 

                 6: "Shirt",

                 7: "Sneaker",

                 8: "Bag",

                 9: "Ankle Boot"

                 }

    input = (label.item() if type(label) == torch.Tensor else label)

    return output_mapping[input]
a = next(iter(train_loader))

a[0].size()
len(train_set)
image, label = next(iter(train_set))

plt.imshow(image.squeeze(), cmap="gray")

print(label)
demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)



batch = next(iter(demo_loader))

images, labels = batch

print(type(images), type(labels))

print(images.shape, labels.shape)
grid = torchvision.utils.make_grid(images, nrow=10)



plt.figure(figsize=(15, 20))

plt.imshow(np.transpose(grid, (1, 2, 0)))

print("labels: ", end=" ")

for i, label in enumerate(labels):

    print(output_label(label), end=", ")
criterion = nn.CrossEntropyLoss()
input_dim = 28 * 28

hidden_dim1 = 50

hidden_dim2 = 50

output_dim = 10
cust_epochs = 10
learning_rate = 0.01

model = Net(input_dim, hidden_dim1, hidden_dim2, output_dim)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

training_results = train(model, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
print(training_results)
learning_rate = 0.01

modelRelu = NetRelu(input_dim, hidden_dim1, hidden_dim2, output_dim)

optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)

training_results_relu = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
learning_rate = 0.01

modelTanh = NetTanh(input_dim, hidden_dim1, hidden_dim2, output_dim)

optimizer = torch.optim.SGD(modelRelu.parameters(), lr=learning_rate)

training_results_tanh = train(modelRelu, criterion, train_loader, validation_loader, optimizer, epochs=cust_epochs)
plt.figure(figsize=(8,4))

plt.plot(training_results_tanh['training_loss'], label='tanh')

plt.plot(training_results['training_loss'], label='sigmoid')

plt.plot(training_results_relu['training_loss'], label='relu')

plt.ylabel('loss')

plt.xlabel('Samples of Training')

plt.title('training loss iterations')

plt.legend()

plt.show()
plt.figure(figsize=(8,4))

plt.grid()

plt.plot(training_results_tanh['validation_accuracy'], label = 'tanh')

plt.plot(training_results['validation_accuracy'], label = 'sigmoid')

plt.plot(training_results_relu['validation_accuracy'], label = 'relu') 

plt.ylabel('validation accuracy')

plt.title('Accuracy Using different Activation Functions')

plt.xlabel('Iteration')   

plt.legend()

plt.show()