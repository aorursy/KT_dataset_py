from __future__ import print_function



import warnings

warnings.filterwarnings('ignore')



import logging

logging.basicConfig(level=logging.FATAL)



import argparse, random

import numpy as np



import matplotlib

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from torchvision import datasets, transforms
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',

                    help='input batch size for training (default: 128)')

parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',

                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',

                    help='number of epochs to train (default: 20)')

parser.add_argument('--num-k', type=int, default=4, metavar='N',

                    help='hyper paremeter for generator update')

parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',

                    help='learning rate (default: 0.01)')

parser.add_argument('--no-cuda', action='store_true', default=False,

                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=42, metavar='S',

                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',

                    help='how many batches to wait before logging training status')



parser.add_argument('--save-model', action='store_true', default=False,

                    help='For Saving the current Model')

args = parser.parse_args([])
use_cuda = not args.no_cuda and torch.cuda.is_available()



torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)

torch.cuda.manual_seed(args.seed)

random.seed(args.seed)

np.random.seed(args.seed)



device = torch.device("cuda" if use_cuda else "cpu")



kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
svhn_train_loader = torch.utils.data.DataLoader(

    datasets.SVHN('./data/data_svhn', split='train', download=True,

                   transform=transforms.Compose([

                       transforms.ToTensor(),

                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                   ])),

    batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)



svhn_test_loader = torch.utils.data.DataLoader(

    datasets.SVHN('./data/data_svhn', split='test', download=True,

                  transform=transforms.Compose([

                       transforms.ToTensor(),

                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                   ])),

    batch_size=args.test_batch_size, drop_last=True, shuffle=True, **kwargs)



mnist_train_loader = torch.utils.data.DataLoader(

    datasets.MNIST('./data/data_mnist', train=True, download=True,

                  transform=transforms.Compose([

                       transforms.Resize((32,32)),

                       transforms.Grayscale(num_output_channels=3), 

                       transforms.ToTensor(),

                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                   ])),

    batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)



mnist_test_loader = torch.utils.data.DataLoader(

    datasets.MNIST('./data/data_mnist', train=False, download=True,

                  transform=transforms.Compose([

                       transforms.Resize((32,32)),

                       transforms.Grayscale(num_output_channels=3), 

                       transforms.ToTensor(),

                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                   ])),

    batch_size=args.test_batch_size, drop_last=True, shuffle=True, **kwargs)
class Feature(nn.Module):

    def __init__(self):

        super(Feature, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)

        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(8192, 3072)

        self.bn1_fc = nn.BatchNorm1d(3072)



    def forward(self, x):

        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)

        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), 8192)

        x = F.relu(self.bn1_fc(self.fc1(x)))

        x = F.dropout(x, training=self.training)

        return x
class Predictor(nn.Module):

    def __init__(self, prob=0.5):

        super(Predictor, self).__init__()

        self.fc1 = nn.Linear(8192, 3072)

        self.bn1_fc = nn.BatchNorm1d(3072)

        self.fc2 = nn.Linear(3072, 2048)

        self.bn2_fc = nn.BatchNorm1d(2048)

        self.fc3 = nn.Linear(2048, 10)

        self.bn_fc3 = nn.BatchNorm1d(10)

        self.prob = prob



    def forward(self, x):

        x = F.relu(self.bn2_fc(self.fc2(x)))

        x = self.fc3(x)

        return x
def init_weights(m):

    if type(m) == nn.Conv2d:

        nn.init.xavier_normal(m.weight)

    if type(m) == nn.Linear:

        nn.init.xavier_normal(m.weight)
G = Feature().apply(init_weights).to(device)

C1 = Predictor().apply(init_weights).to(device)

C2 = Predictor().apply(init_weights).to(device)
opt_G = optim.Adam(G.parameters(),lr=args.lr, weight_decay=0.0005)

opt_C1 = optim.Adam(C1.parameters(), lr=args.lr, weight_decay=0.0005)

opt_C2 = optim.Adam(C2.parameters(), lr=args.lr, weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()
def train():

    G.train()

    C1.train()

    C2.train()

    

    def reset_grad():

        opt_G.zero_grad()

        opt_C1.zero_grad()

        opt_C2.zero_grad()

    

    def discrepancy(out1, out2):

        return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))



    data_zip = enumerate(zip(svhn_train_loader, mnist_train_loader))

    for batch_idx, ((img_s, label_s), (img_t,_)) in data_zip:

        img_t = img_t.to(device)

        img_s = img_s.to(device)

        label_s = label_s.to(device)

        

        reset_grad()

        feat_s = G(img_s)

        output_s1 = C1(feat_s)

        output_s2 = C2(feat_s)

        

        loss_s1 = criterion(output_s1, label_s)

        loss_s2 = criterion(output_s2, label_s)

        loss_s = loss_s1 + loss_s2

        

        loss_s.backward()

        opt_G.step()

        opt_C1.step()

        opt_C2.step()

        

        reset_grad()

        feat_s = G(img_s)

        output_s1 = C1(feat_s)

        output_s2 = C2(feat_s)

        feat_t = G(img_t)

        output_t1 = C1(feat_t)

        output_t2 = C2(feat_t)

        

        loss_s1 = criterion(output_s1, label_s)

        loss_s2 = criterion(output_s2, label_s)

        loss_s = loss_s1 + loss_s2

        loss_dis = discrepancy(output_t1, output_t2)

        loss = loss_s - loss_dis

        

        loss.backward()

        opt_C1.step()

        opt_C2.step()

        

        reset_grad()

        

        for i in range(args.num_k):

            feature_t = G(img_t)

            output_t1 = C1(feature_t)

            output_t2 = C2(feature_t)

            loss_dis = discrepancy(output_t1, output_t2)

            loss_dis.backward()

            opt_G.step()

            reset_grad()

        

        if (batch_idx + 1) % args.log_interval == 0:

            print("Epoch: {}/{} [{}/{}]: Loss1: {:.5f}, Loss2: {:.5f}, Discrepancy: {:.5f}"

                 .format(epoch + 1, args.epochs, batch_idx + 1, min(len(svhn_train_loader), len(mnist_train_loader)), 

                         loss_s1.item(), loss_s2.item(), loss_dis.item()))
list_acc_mnist_train = []

list_acc_svhn_train = []

list_acc_svhn_test = []



def test():

    G.eval()

    C1.eval()

    C2.eval()





    correct = 0

    

    for (img_s, label_s) in mnist_test_loader:

        img_s = img_s.to(device)

        label_s = label_s.to(device)

        

        feat = G(img_s)

        output1 = C1(feat)

        output2 = C2(feat)

        

        output_ensemble = output1 + output2

        pred_ensemble = output_ensemble.data.max(1)[1]

        

        correct += pred_ensemble.eq(label_s.data).cpu().sum()

        

    acc_mnist_train = 100. * correct / len(mnist_test_loader.dataset)

    

    

    correct = 0

    

    for (img_s, label_s) in svhn_train_loader:

        img_s = img_s.to(device)

        label_s = label_s.to(device)

        

        feat = G(img_s)

        output1 = C1(feat)

        output2 = C2(feat)

        

        output_ensemble = output1 + output2

        pred_ensemble = output_ensemble.data.max(1)[1]

        

        correct += pred_ensemble.eq(label_s.data).cpu().sum()

    

    acc_svhn_train = 100. * correct / len(svhn_train_loader.dataset)

    

    

    correct = 0

    

    for (img_s, label_s) in svhn_test_loader:

        img_s = img_s.to(device)

        label_s = label_s.to(device)

        

        feat = G(img_s)

        output1 = C1(feat)

        output2 = C2(feat)

        

        output_ensemble = output1 + output2

        pred_ensemble = output_ensemble.data.max(1)[1]

        

        correct += pred_ensemble.eq(label_s.data).cpu().sum()

    

    acc_svhn_test = 100. * correct / len(svhn_test_loader.dataset)

        

    print(

        '\nTest: MNIST Test: {:.0f}%, SVHN Train: {:.0f}%, SVHN Test: {:.0f}% \n'.format(

            acc_mnist_train, acc_svhn_train, acc_svhn_test))

    list_acc_mnist_train.append(acc_mnist_train)

    list_acc_svhn_train.append(acc_svhn_train)

    list_acc_svhn_test.append(acc_svhn_test)
for epoch in range(args.epochs):

    train()

    test()
fig, ax = plt.subplots()

ax.plot(range(1, args.epochs + 1), list_acc_svhn_train)

ax.set(xlabel='epoch', ylabel='accuracy (%)',

       title='Accuracy SVHN Train')

ax.set_ylim(0, 100)

ax.grid()

plt.xticks(range(1, args.epochs + 1))

plt.show()
fig, ax = plt.subplots()

ax.plot(range(1, args.epochs + 1), list_acc_svhn_test)

ax.set(xlabel='epoch', ylabel='accuracy (%)',

       title='Accuracy SVHN Test')

ax.set_ylim(0, 100)

ax.grid()

plt.xticks(range(1, args.epochs + 1))

plt.show()
fig, ax = plt.subplots()

ax.plot(range(1, args.epochs + 1), list_acc_mnist_train)

ax.set(xlabel='epoch', ylabel='accuracy (%)',

       title='Accuracy MNIST Test')

ax.set_ylim(0, 100)

ax.grid()

plt.xticks(range(1, args.epochs + 1))

plt.show()