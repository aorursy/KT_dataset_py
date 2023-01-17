# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import time

import platform



print('Current time:\t%s' % time.strftime('%c'))

print('machine: ', platform.machine())

print('-' * 20)

!cat /proc/cpuinfo | grep 'model name'

print('-' * 20)

!nvidia-smi
epsilons = [0., .05, .1, .15, .2, .25, .3]

pretrained_model = '../input/lenet_mnist_model.pth'

use_cuda = True
import os



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image



class MyMNIST(Dataset):

    train_file = 'training.pt'

    test_file = 'test.pt'

    def __init__(self, root_dir, train=False, transform=None):

        self.root_dir = root_dir

        self.train = train

        self.transform = transform

        if self.train:

            self.train_data, self.train_labels = torch.load(

                os.path.join(root_dir, self.train_file))

        else:

            self.test_data, self.test_labels = torch.load(

                os.path.join(root_dir, self.test_file))

    

    def __len__(self):

        if self.train:

            return len(self.train_data)

        else:

            return len(self.test_data)

    

    def __getitem__(self, index):

        if self.train:

            img, target = self.train_data[index], self.train_labels[index]

        else:

            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:

            img = self.transform(img)

        return img, target

        
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.conv2_dropout = nn.Dropout2d()

        self.fc1 = nn.Linear(320, 50)

        self.fc2 = nn.Linear(50, 10)

    

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
test_loader = torch.utils.data.DataLoader(

    MyMNIST('../input', train=False, transform=transforms.Compose([

        transforms.ToTensor()])),

    batch_size=1, shuffle=True)

print("CUDA Available: ", torch.cuda.is_available())

device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')

print(torch.cuda.get_device_name(device))

model = Net().to(device)

model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

model.eval()
def fgsm_attack(image, epsilon, data_grad):

    sign_data_grad = data_grad.sign()

    perturbed_image = image + epsilon * sign_data_grad

    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image
def test(model, device, test_loader, epsilon):

    correct = 0

    adv_examples = []

    

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        data.requires_grad = True

        output = model(data)

        init_pred = output.max(1, keepdim=True)[1]

        

        if init_pred.item() != target.item():

            continue

        

        loss = F.nll_loss(output, target)

        model.zero_grad()

        loss.backward()

        

        data_grad = data.grad.detach()

        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():

            correct += 1

            if epsilon == 0 and len(adv_examples) < 5:

                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()

                adv_examples.append(

                    (init_pred.item(), final_pred.item(), adv_ex))

        else:

            if len(adv_examples) < 5:

                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()

                adv_examples.append(

                    (init_pred.item(), final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))

    print('Epsilon: {}\tTest Accuracy = {} / {} = {}'.format(

        epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples
accuracies = []

examples = []



for eps in epsilons:

    print('-'*10)

    since = time.time()

    acc, ex = test(model, device, test_loader, eps)

    print('finished in {}'.format(time.time() - since))

    accuracies.append(acc)

    examples.append(ex)
import matplotlib.pyplot as plt

import numpy as np



plt.figure(figsize=(5, 5))

plt.plot(epsilons, accuracies, '*-')

plt.yticks(np.arange(0, 1.1, step=0.1))

plt.xticks(np.arange(0, 0.35, step=0.05))

plt.title('Accuracy vs Epsilon')

plt.xlabel('Epsilon')

plt.ylabel('Accuracy')

plt.savefig('Accuracy vs Epsilon.jpg')

plt.show()
cnt = 0

plt.figure(figsize=(8, 10))

for i in range(len(epsilons)):

    for j in range(len(examples[i])):

        cnt += 1

        plt.subplot(len(epsilons), len(examples[0]), cnt)

        plt.xticks([], [])

        plt.yticks([], [])

        if j == 0:

            plt.ylabel('Eps: {}'.format(epsilons[i]), fontsize=14)

        orig, adv, ex = examples[i][j]

        plt.title("{} -> {}".format(orig, adv))

        plt.imshow(ex, cmap='gray')

plt.tight_layout()

plt.savefig('Adversarial Examples.jpg')

plt.show()
