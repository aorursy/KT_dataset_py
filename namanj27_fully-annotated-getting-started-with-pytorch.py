import torch # Tensor Package (for use on GPU)

from torch.autograd import Variable # for computational graphs

import torch.nn as nn ## Neural Network package

import torch.nn.functional as F # Non-linearities package

import torch.optim as optim # Optimization package

from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data

import torchvision # for dealing with vision data

import torchvision.transforms as transforms # for modifying vision data to run it through models



import matplotlib.pyplot as plt # for plotting

import numpy as np

from tqdm import tqdm


transform = transforms.Compose(

   [

    transforms.ToTensor(), # normalize images from (0-255) to (0-1)

       

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # mean normalize ; subtract by 0.5 mean and divide by 0.5 std deviation

   ])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # downloading CIFAR10 Dataset



trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True) # dataLoader to get inputs/labels pair in batches



# Simillarly for test set

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)



classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') # CIFAR10 .... 10 classes
def imshow(img):

    img = img / 2 + 0.5 # normalizing images

    npimg = img.numpy() 

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
class Net(nn.Module): #using pytorch's Net class for Neural Networks

    def __init__(self):

        super(Net, self).__init__()

        

        # input image shape is 3 x 32 x 32

                                        #.     Output image sizes through each layer below

        self.conv1 = nn.Conv2d(3, 10, 5)#      10 x 28 x 28

        self.pool = nn.MaxPool2d(2, 2) #       10 x 14 x 14

        self.conv2 = nn.Conv2d(10, 20, 5) #    20 x 10 x 10 

        self.fc1 = nn.Linear(20 * 5 * 5, 120) # pooling it onece more will give 20 x 5 x 5

        self.fc2 = nn.Linear(120, 10)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 20 * 5 * 5)

        x = F.relu(self.fc1(x) )

        x = F.relu(self.fc2(x))

        return x



net = Net().cuda()  # Remember to turn on Accelerator as GPU

# .... feel free to tweek them ;)



NUMBER_OF_EPOCHS = 10

LEARNING_RATE = 1e-2



loss_function = nn.CrossEntropyLoss() # classification problem hence crossentropy loss



optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE) # telling optimizer what parameters to update , along with learning rate



for epoch in tqdm(range(NUMBER_OF_EPOCHS)):

    

    train_loader_iter = iter(trainloader) # iterator for our train loader

    

    for batch_idx, (inputs, labels) in enumerate(train_loader_iter):

        

        net.zero_grad() # zeroing out gradients for each epoch to get fresh update 

        

        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.cuda())

        

        output = net(inputs) # calculating outputs

        

        loss = loss_function(output, labels) # computing loss

        

        loss.backward() # computing gradients of loss wrt parameters of net model

        

        optimizer.step() # updating parameters

        

    if epoch % 5 is 0:

        print("Iteration: " + str(epoch + 1))
dataiter = iter(testloader) #test iterator



images, labels = dataiter.next() # getting first image in iterator



imshow(torchvision.utils.make_grid(images)) # plotting test images using our custom imshow function



outputs = net(Variable(images.cuda()))     # computing outputs



_, predicted = torch.max(outputs.data, 1) # getting indices for class out of 10 classes i.e (0-9) values



print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]

                              for j in range(4)))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
correct = 0

total = 0



for data in tqdm(testloader):

    images, labels = data

    labels = labels.cuda()

    

    outputs = net(Variable(images.cuda()))

    _, predicted = torch.max(outputs.data, 1)

    

    total += labels.size(0) # total images

    correct += (predicted == labels).sum() # correct predictions

    

print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))