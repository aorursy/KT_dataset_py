import numpy as np

import sys



import torch

import torch.nn as nn

from torch.autograd import Variable

import torchvision.datasets as dset

import torchvision.transforms as transforms

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Function



from sklearn.manifold import TSNE

import matplotlib.pyplot as plt



import argparse

import os

import random

import shutil

import time

import warnings



from os.path import join

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from os import listdir

import pandas as pd

import seaborn as sns
cuda = torch.cuda.is_available()

device = torch.device('cuda' if cuda else 'cpu')
class GradReverse(Function):

    @staticmethod

    def forward(ctx, x, alpha):

        ctx.alpha = alpha

        return x.view_as(x)

    

    @staticmethod

    def backward(ctx, grad_output):

        return grad_output.neg() * ctx.alpha, None

    



def grad_reverse(x, alpha):

    return GradReverse.apply(x, alpha)





def tsne_plot(cls_model, train_loader, test_loader, src_name, tar_name, batch_size, title, mode=True):

    cls_model.load_state_dict(torch.load(title + '_' + src_name + '2' + tar_name + '.pth'))

    cls_model.eval()

    features = []

    labels = torch.LongTensor()

    

    for index, batch in enumerate(train_loader):

        x, y = batch

        x = x.to(device)

        y = y.to(device)

        

        if mode:

            _, feature = cls_model(x)

        else:

            feature = cls_model(x)

        features.append(feature.cpu().detach().numpy())

        

        if index * batch_size > 2000:

            break

            

    for index, batch in enumerate(test_loader):

        x, y = batch

        x = x.to(device)

        labels = torch.cat((labels, y), 0)

        

        if mode:

            _, feature = cls_model(x)

        else:

            feature = cls_model(x)

            

        features.append(feature.cpu().detach().numpy())

        

        if index * batch_size > 2000:

            break

            

    features = np.array([feature for feature in features])

    features = features.reshape(-1, 2048)

    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2000, learning_rate=10)

    tsne_results = tsne.fit_transform(features)

    dim1 = tsne_results[:,0]

    dim2 = tsne_results[:,1]

    plt.figure(figsize=(10,6))

    sns.scatterplot(

        x=dim1, y=dim2,

        hue=labels,

        legend="full",

        alpha=0.7,

        palette=sns.color_palette("hls", 10),

    )

    plt.show()
mean = np.array([0.5, 0.5, 0.5])

std = np.array([0.5, 0.5, 0.5])



class ToRGB(object):

    def __init_(self):

        pass

    

    def __call__(self, sample):

        sample = sample.convert('RGB')

        return sample



transform = transforms.Compose([

    ToRGB(),

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)

])





class Dotoset(Dataset):

    def __init__(self, img_path, label_path, transforms=None):

        super(Dotoset, self).__init__()

        

        self.img_path = join(os.getcwd(), 'dataset', img_path, label_path)

        self.label_path = join(os.getcwd(), 'dataset', img_path, label_path)

        

        self.label_numpy = pd.read_csv(self.label_path).values[:, 1]

        self.imgs_files = listdir(self.img_path)

        

        self.transform = transform

        

    def __len__(self):

        return len(self.imgs_files)

    

    def __getitem__(self, idx):

        filename = self.imgs_files[idx]

        img = self.transform(Image.open(join(self.img_path, filename)))

        

        label = self.label_numpy[idx]

        label = torch.LongTensor([label])

        

        return img, label
class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 16, 5, 1, 2),

            nn.BatchNorm2d(16),

            nn.ReLU(True),

            nn.MaxPool2d(2),

            

            nn.Conv2d(16, 32, 5, 1, 2),

            nn.BatchNorm2d(32),

            nn.ReLU(True),

            nn.MaxPool2d(2),

            

            nn.Dropout2d(),

            

            nn.Conv2d(32, 64, 5, 1, 2),

            nn.BatchNorm2d(64),

            nn.ReLU(True)

        )

        

        self.fc = nn.Sequential(

            nn.Linear(8 * 8 * 64, 1024),

            nn.BatchNorm1d(1024),

            nn.ReLU(True),

            nn.Dropout(0.5)

        )

        

    def forward(self, x):

        feature = self.cnn(x)

        feature = feature.view(x.size(0), -1)

        feature = self.fc(feature)

        

        return feature
class Predictor(nn.Module):

    def __init__(self):

        super(Predictor, self).__init__()

        self.fc = nn.Sequential(

            nn.Linear(1024, 1024),

            nn.BatchNorm1d(1024),

            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.Linear(1024, 10)

        )

        

    def forward(self, feature, alpha=1, reverse=False):

        if reverse:

            reverse_feature = grad_reverse(feature, alpha)

            return self.fc(reverse_feature)

        return self.fc(feature)
download = True

batch_size = 128

epoches = 75

mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
rgb_transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)

])



gray2rgb_transform = transforms.Compose([

    ToRGB(),

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)

])
def test(model, encoder, device, test_loader, set_name, res):

    model.eval()

    test_loss, correct = 0, 0

    

    with torch.no_grad():

        for data, target in test_loader:

            x, y = data.to(device), target.to(device)

            y = y.view(-1)

            feature = encoder(x)

            output = model(feature)

            test_loss += F.nll_loss(output, y, reduction='sum').item()

            correct += np.sum(np.argmax(output.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

        test_loss /= len(test_loader.dataset)

    if not res:

        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

                set_name, test_loss, correct, len(test_loader.dataset),

                100. * correct / len(test_loader.dataset)))

        return None

    else:

        return 100. * correct / len(test_loader.dataset), test_loss

    

def train(

    encoder,

    cls_model_1,

    cls_model_2,

    optimizer_encoder,

    optimizer_clf_1,

    optimizer_clf_2,

    epoches,

    train_loader,

    test_loader,

    sorce_name,

    target_name,

    source_loader,

    target_loader,

    batch_size,

    source_train_loader,

    source_test_loader,

    target_test_loader

):

    loss_fn_cls = nn.CrossEntropyLoss()

    ac_list, loss_list = [], []

    

    svhn_train_acc_list, svhn_train_loss_list = [], []

    svhn_test_acc_list, svhn_test_loss_list = [], []

    mnist_test_acc_list, mnist_test_loss_list = [], []

    

    max_ = 0

    

    end = time.time()

    

    torch.save(encoder.state_dict(), 'mcd_' + sorce_name + '2' + target_name + '.pth')

    tsne_plot(encoder, source_loader, train_loader, sorce_name, target_name, batch_size, 'mcd', mode=False)

    for epoch in range(epoches):

        cls_model_1.train()

        cls_model_2.train()

        encoder.train()

        

        print(f'Epoch: {epoch}')

        

        for index, (source_batch, target_batch) in enumerate(zip(train_loader, test_loader)):

            # step 1

            x, y = source_batch

            x, y = x.to(device), y.to(device)

            y = y.view(-1)

            

            feature = encoder(x)

            pred1 = cls_model_1(feature)

            pred2 = cls_model_2(feature)

            

            loss1 = loss_fn_cls(pred1, y)

            loss2 = loss_fn_cls(pred2, y)

            loss = loss1 + loss2

            

            optimizer_encoder.zero_grad()

            optimizer_clf_1.zero_grad()

            optimizer_clf_2.zero_grad()

            

            # step 2

            target_x, _ = target_batch

            target_x = target_x.to(device)

            

            s_feature = encoder(x)

            pred_s_1 = cls_model_1(s_feature)

            pred_s_2 = cls_model_2(s_feature)

            

            t_feature = encoder(target_x)

            pred_target_1 = cls_model_1(t_feature)

            pred_target_2 = cls_model_2(t_feature)

            

            source_loss = loss_fn_cls(pred_s_1, y) + loss_fn_cls(pred_s_2, y)

            discrepency_loss = torch.mean(torch.abs(

                    F.softmax(pred_target_1, dim=1) - 

                    F.softmax(pred_target_2, dim=1)

                )

            )

            

            loss = source_loss - discrepency_loss

            

            optimizer_clf_1.zero_grad()

            optimizer_clf_2.zero_grad()

            

            loss.backward(retain_graph=True)

            

            optimizer_clf_1.step()

            optimizer_clf_2.step()

            

            # step 3

            for i in range(3):

                t_feature = encoder(target_x)

                pred_tar_1 = cls_model_1(t_feature)

                pred_tar_2 = cls_model_2(t_feature)

                

                discrepency_loss = torch.mean(abs(

                        F.softmax(pred_target_1, dim=1) - 

                        F.softmax(pred_target_2, dim=1)

                    )

                )

    

                discrepency_loss.backward(retain_graph=True)

                optimizer_encoder.step()

                

                optimizer_encoder.zero_grad()

                optimizer_clf_1.zero_grad()

                optimizer_clf_2.zero_grad()

                

            if index % 400 == 0:

                print(f'{index} / {min([len(train_loader), len(test_loader)])}')

                

        cls_model_1.eval()

        cls_model_2.eval()

        encoder.eval()

        

        ac_1, ac_2, ac_3, total_loss = 0, 0, 0, 0

        

        with torch.no_grad():

            for batch in test_loader:

                x, y = batch

                x, y = x.to(device), y.to(device)

                y = y.view(-1)

                

                feature = encoder(x)

                

                pred_c1 = cls_model_1(feature)

                pred_c2 = cls_model_2(feature)

                pred_combine = pred_c1 + pred_c2

                

                ac_1 += np.sum(np.argmax(pred_c1.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

                ac_2 += np.sum(np.argmax(pred_c2.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

                ac_3 += np.sum(np.argmax(pred_combine.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

                

                total_loss += loss.item()

        

        print(f'Accuracy: {(ac_1 / len(test_loader) / batch_size)}, Avg loss: {total_loss / len(test_loader)}')

        print(f'Accuracy: {(ac_2 / len(test_loader) / batch_size)}, Avg loss: {total_loss / len(test_loader)}')

        print(f'Accuracy: {(ac_3 / len(test_loader) / batch_size)}, Avg loss: {total_loss / len(test_loader)}')

        

        svhn_train_acc, svhn_train_loss = test(cls_model_1, encoder, device, source_train_loader, 'SVHN train', res=True)

        svhn_test_acc, svhn_test_loss = test(cls_model_1, encoder, device, source_test_loader, 'SVHN test', res=True)

        mnist_test_acc, mnist_test_loss = test(cls_model_1, encoder, device, target_test_loader, 'MNIST test', res=True)

        

        svhn_train_acc_list.append(svhn_train_acc)

        svhn_train_loss_list.append(svhn_train_loss)

        

        svhn_test_acc_list.append(svhn_test_acc)

        svhn_test_loss_list.append(svhn_test_loss)



        mnist_test_acc_list.append(mnist_test_acc)

        mnist_test_loss_list.append(mnist_test_loss)

        

        ac = max([ac_1, ac_2, ac_3])

        

        ac_list.append(ac / len(test_loader) / batch_size)

        loss_list.append(total_loss / len(test_loader) / batch_size)

        

        if (ac / len(test_loader) / batch_size) > max_:

            max_ = (ac / len(test_loader) / batch_size)

            torch.save(cls_model_1.state_dict(), 'mcd_' + sorce_name + '2' + target_name + '_1.pth')

            torch.save(cls_model_2.state_dict(), 'mcd_' + sorce_name + '2' + target_name + '_1.pth')

            torch.save(encoder.state_dict(), 'mcd_' + sorce_name + '2' + target_name + '.pth')

            

    return ac_list, loss_list, svhn_train_acc_list, svhn_train_loss_list, svhn_test_acc_list, svhn_test_loss_list, mnist_test_acc_list, mnist_test_loss_list
def weights_init_uniform(m):

    if type(m) == nn.Linear or type(m) == nn.Conv2d:

        torch.nn.init.xavier_normal_(m.weight)

        m.bias.data.fill_(0.01)
def main(source, target):

    G = Encoder().to(device)

    

    cls_c1 = Predictor().to(device)

    cls_c2 = Predictor().to(device)

    

    cls_c1.apply(weights_init_uniform)

    cls_c2.apply(weights_init_uniform)

    

    # dataloaders

    src_train_set = dset.SVHN('./dataset/svhn/', download=download, transform=rgb_transform)

    src_test_set = dset.SVHN('./dataset/svhn/', split='test', download=download, transform=rgb_transform)

    target_train_set = dset.MNIST('./dataset/mnist', train=True, download=True, transform=gray2rgb_transform)

    target_test_set = dset.MNIST('./dataset/mnist', train=False, download=True, transform=gray2rgb_transform)

    

    source_train_loader = torch.utils.data.DataLoader(

        dataset=src_train_set,

        batch_size=batch_size,

        shuffle=True

    )

    source_test_loader = torch.utils.data.DataLoader(

        dataset=src_test_set,

        batch_size=batch_size,

        shuffle=True

    )

    target_train_loader = torch.utils.data.DataLoader(

        dataset=target_train_set,

        batch_size=batch_size,

        shuffle=True

    )

    target_test_loader = torch.utils.data.DataLoader(

        dataset=target_test_set,

        batch_size=batch_size,

        shuffle=True

    )

    

    # optimizers

    optimizer_encoder = optim.Adam(G.parameters(), lr=3e-4, weight_decay=0.0005)

    optimizer_clf_1 = optim.Adam(cls_c1.parameters(), lr=3e-4, weight_decay=0.0005)

    optimizer_clf_2 = optim.Adam(cls_c2.parameters(), lr=3e-4, weight_decay=0.0005)

    

    # train

    ac_list, loss_list, svhn_train_acc_list, svhn_train_loss_list, svhn_test_acc_list, svhn_test_loss_list, mnist_test_acc_list, mnist_test_loss_list = train(

        encoder=G,

        optimizer_encoder=optimizer_encoder,

        optimizer_clf_1=optimizer_clf_1,

        optimizer_clf_2=optimizer_clf_2,

        cls_model_1=cls_c1,

        cls_model_2=cls_c2,

        epoches=epoches,

        train_loader=source_train_loader,

        test_loader=target_train_loader,

        sorce_name=source,

        target_name=target,

        source_loader=source_train_loader,

        target_loader=target_train_loader,

        batch_size=batch_size,

        source_train_loader=source_train_loader,

        source_test_loader=source_test_loader,

        target_test_loader=target_test_loader

    )

    

    # plot tsne

    loss_list = np.array(loss_list).flatten()

    epoch = [i for i in range(epoches)]

    tsne_plot(

        G,

        source_train_loader,

        target_train_loader,

        source,

        target,

        batch_size,

        'mcd',

        mode=False

    )

    

    test(cls_c1, G, device, source_train_loader, 'SVHN train', res=False)

    test(cls_c1, G, device, source_test_loader, 'SVHN test', res=False)

    test(cls_c1, G, device, target_test_loader, 'MNIST test', res=False)

    

    # plot learning curve

    plt.figure()

    plt.plot(epoch, ac_list)

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('DA accuracy: svhn to mnist')

    

    plt.figure()

    plt.plot(epoch, loss_list)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('DA loss: svhn to mnist')

    

    # svhn train plots

    plt.figure()

    plt.plot(epoch, svhn_train_acc_list)

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('SVHN train accuracy')

    

    plt.figure()

    plt.plot(epoch, svhn_train_loss_list)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('SVHN train loss')

    

    # svhn test plots

    plt.figure()

    plt.plot(epoch, svhn_test_acc_list)

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('SVHN test accuracy')

    

    plt.figure()

    plt.plot(epoch, svhn_test_loss_list)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('SVHN test loss')

    

    # mnist test plots

    plt.figure()

    plt.plot(epoch, mnist_test_acc_list)

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('MNIST test accuracy')

    

    plt.figure()

    plt.plot(epoch, mnist_test_loss_list)

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('MNIST test loss')

    

    plt.savefig('domain_adapt_' + source + '_to_' + target + '_loss.jpg')
source, target = 'svhn', 'mnist'

main(source, target)