import numpy as np

import urllib

from urllib.request import urlopen

import matplotlib.pyplot as plt

from PIL import Image

import requests

from io import BytesIO
url='https://vnokia.net/images/wallpaper/2017/wallpaper_480x800_21.jpg'



response = requests.get(url)

img = Image.open(BytesIO(response.content))

ima = np.asanyarray(img)
g_fil_3 = (1/16.)*np.array([[1,2,1],

                            [2,4,2],

                            [1,2,1]])# увеличивает контраст (радиальный)



emboss =         np.array([[-2,-1, 0],

                           [-1, 1, 1],

                           [ 0, 1, 2]])# Диагональный градиен с СЗ на ЮВ



edge8 =          np.array([[-1,-1,-1],

                           [-1, 8,-1],

                           [-1,-1,-1]])# Хорошо находит стороны (углы) по верт или горизонт



bl2whR =          np.array([[-1, 0, 1],

                           [-1, 0, 1],

                           [-1, 0, 1]])# Градиент слева направо



bl2wh5R =           np.array([[-2, -1, 0, 1, 2],

                             [-2, -1, 0, 1, 2],

                             [-2, -1, 0, 1, 2],

                             [-2, -1, 0, 1, 2],

                             [-2, -1, 0, 1, 2]])# Градиент слева направо большего размера и градации





wline5  =          np.array([[-2.5, 1, 3, 1, -2.5],

                             [-2.5, 1, 3, 1, -2.5],

                             [-2.5, 1, 3, 1, -2.5],

                             [-2.5, 1, 3, 1, -2.5],

                             [-2.5, 1, 3, 1, -2.5]])# Вертикальные линии



sharp =          np.array([[ 0,-1, 0],

                           [-1, 5,-1], 

                           [ 0,-1, 0]])#Резкость



ssharp =          np.array([[-0.5,-1, -0.5],

                            [-1,   7,   -1],

                            [-0.5,-1, -0.5]])#Еще большая резкозть



g_fil_5 = (1/256.)*np.array([[1, 4, 6, 4,1],

                             [4,16,24,16,4],

                             [6,24,36,24,6],

                             [4,16,24,16,4],

                             [1, 4, 6, 4,1]])#Нахождение ярких кругов, контраст?



blur_box_3 = (1/9.)*np.array([[1,1,1],

                              [1,1,1],

                              [1,1,1]])#Размытие



my_fil =     np.array([[1,-1, 1],

                       [-1,8,-1],

                       [1,-1, 1]])#Увеличение яркости



blur_box_5 = (1/25.)*np.array([[1, 1, 1, 1, 1],

                               [1, 1, 1, 1, 1],

                               [1, 1, 1, 1, 1],

                               [1, 1, 1, 1, 1],

                               [1, 1, 1, 1, 1]])#Размытие большим фильтром (размывает больше)
list_of_filters =[g_fil_3,



emboss,



edge8,

bl2whR,

bl2wh5R,

wline5,

sharp,

ssharp,

g_fil_5,

blur_box_3,

my_fil,

blur_box_5]
def convolution(image, filtr):

    image = image.astype('float')

    image_shape = image.shape

    filtr_shape = filtr.shape

    gap = int(filtr_shape[0]/2)

    output_image = np.array(image.copy())

    output_image = output_image[0:(image_shape[0]-(gap+1)),0:(image_shape[1]-(gap+1))]

    for i in range(gap,(image_shape[0]-gap-1)):

        for j in range(gap, (image_shape[1]-gap-1)):

            for k in range(3):

                pixel = ((image[(i-gap):(i+gap+1),(j-gap):(j+gap+1),k])*filtr).sum().sum()

                if pixel > 255:

                    pixel = 255

                if pixel < 0:

                    pixel = 0

                output_image[i][j][k] = pixel

    return output_image.astype('uint8')
for f in list_of_filters:

  plt.imshow(convolution(ima, f))

  plt.show()



  

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets, transforms
def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(data), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.item()))



def test(args, model, device, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= len(test_loader.dataset)



    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

def main():

    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',

                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',

                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=19, metavar='N',

                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',

                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',

                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,

                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',

                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=75, metavar='N',

                        help='how many batches to wait before logging training status')

    

    parser.add_argument('--save-model', action='store_true', default=False,

                        help='For Saving the current Model')

    args = parser.parse_args([])

    use_cuda = not args.no_cuda and torch.cuda.is_available()



    torch.manual_seed(args.seed)



    device = torch.device("cuda" if use_cuda else "cpu")



    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(

        datasets.CIFAR10('../data', train=True, download=True,

                       transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1307,), (0.3081,))

                       ])),

        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(

        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1307,), (0.3081,))

                       ])),

        batch_size=args.test_batch_size, shuffle=True, **kwargs)





    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch)

        test(args, model, device, test_loader)



    if (args.save_model):

        torch.save(model.state_dict(),"mnist_cnn.pt")

class Net(nn.Module): #ПОБЕДИТЕЛЬ

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, 3, 1)

        self.conv2 = nn.Conv2d(128, 256, 3, 1)

        self.conv3 = nn.Conv2d(256, 256, 5, 1)

        

        self.fc1 = nn.Linear(5*5*256, 500)

        self.fc2 = nn.Linear(500, 10)



    def forward(self, x):

        x = F.relu(self.conv1(x))   

        

        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)

        

        x = F.relu(self.conv3(x))

        x = F.max_pool2d(x, 2, 2)

        

        x = x.view(-1, 5*5*256)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    

main()