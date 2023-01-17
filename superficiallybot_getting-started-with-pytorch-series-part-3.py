# importing the libraries



import torch

# Variable object automatically does require_autograd = True

from torch.autograd import Variable

import torchvision.transforms as transforms

import torchvision.datasets as dsets
# torchvision.datasets has preloaded common datasets, ready to use. 

train_dataset = dsets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)

test_dataset = dsets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = True)
# Hyper-parameters settings

batch_size = 100

n_iters = 3000

epochs = n_iters / (len(train_dataset)/ batch_size)



input_dim = 784

output_dim = 10

lr_rate = 0.001
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
class a(torch.nn.Module):

    def __init__(self, input_dim, output_dim):

        super(a, self).__init__()

        self.linear = torch.nn.Linear(input_dim, output_dim)

        

    def forward(self, x):

        outputs = self.linear(x)

        return outputs
model = a(input_dim, output_dim)
# loss Class 

criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr = lr_rate)
# Training the model

num_epochs = 20

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = Variable(images.view(-1, 28*28))

        labels = Variable(labels)

        

        # Forward + Backward + Optimize

        

        optimizer.zero_grad()

        outputs = model(images)

        

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        

        if (i + 1) % 100 == 0:

            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' %(epoch + 1, num_epochs, i+ 1, len(train_dataset)//batch_size, loss.item()))

    
# Test the model

correct = 0

total = 0



for images, labels in test_loader:

    images = Variable(images.view(-1, 28*28))

    outputs = model(images)

    

    _, predicted = torch.max(outputs.data, 1)

    

    total += labels.size(0)

    

    correct += (predicted == labels).sum()

    

print('Accuracy of the model on the 10.000 test images: %d ' %(100 * correct / total))

    