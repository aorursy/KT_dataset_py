import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from sklearn.model_selection import train_test_split

import cv2

import matplotlib.pyplot as plt
import torch

import torch.nn  as nn

import torch.nn.functional  as F

from torch.autograd import Variable

import torch.optim as optim

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from torchvision import transforms,utils
data_dir = "/kaggle/input/flower-recognition-he/he_challenge_data/data/train/"

data_csv = "/kaggle/input/flower-recognition-he/he_challenge_data/data/train.csv"
train_data = pd.read_csv(data_csv)
train_data['category'].unique()
plt.figure(figsize=(20,10))

for i in range(16):

    ind = random.randrange(0, 18500, 1)

    sample_img = data_dir + str(train_data.iloc[ind][0]) + ".jpg"

    img = cv2.imread(sample_img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(4,4,i+1)

    plt.imshow(img)

    plt.ylabel(str(train_data.iloc[ind][1]))
classes_dist = train_data["category"]
plt.hist(classes_dist,bins=102);
class FlowerDataset(Dataset):

    def __init__(self, data_csv, data_path, transform=None):

        self.data_csv = pd.read_csv(data_csv)

        self.data_path = data_path

        self.transform = transform

        

    def __len__(self):

        return len(self.data_csv)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        img = cv2.imread(self.data_path + str(self.data_csv.iloc[idx][0]) + ".jpg")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.array(img).astype("float")

        lbl = self.data_csv.iloc[idx][1] -1

        lbl = np.array(lbl) #.astype("float")

        sample = {'image': img, 'label': lbl}

        

        if(self.transform):

            sample = self.transform(sample)

            

        return sample
flower_data = FlowerDataset(data_csv, data_dir)
type(flower_data[10]['label'])
class Rescale(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size

        

    def __call__(self, sample):

        img, lbl = sample['image'], sample['label']

        img = cv2.resize(img, self.output_size)

        sample = {'image': img, 'label': lbl}        

        return sample



class RandCrop(object):

    def __init__(self, output_size):

        assert isinstance(output_size, (int, tuple))

        self.output_size = output_size

    

    def __call__(self, sample):

        img, lbl = sample['image'], sample['label']

        h,w = img.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0,h - new_h)

        left= np.random.randint(0,w - new_w)

        

        img = img[top:top+new_h, left:left+new_w]

        

        sample = {'image': img, 'label': lbl}

        return sample

    

class ToTensor(object):

    def __call__(self, sample):

        img, lbl = sample['image'], sample['label']

        img = img.transpose(2,0,1)

        

        sample = {'image': torch.from_numpy(img), 'label': torch.from_numpy(lbl)}

        return sample
scale = Rescale((110,110))

rcrop = RandCrop((100,100))

composed = transforms.Compose([scale,rcrop])
samp = flower_data[20]

tsfr_smp = composed(samp)

plt.imshow(tsfr_smp['image'].astype("int"))
flower_data_transformed = FlowerDataset(data_csv, data_dir, transforms.Compose([Rescale((110,110)), RandCrop((100,100)), ToTensor()]))
train_dataloader = DataLoader(flower_data_transformed, batch_size=10, shuffle=True, num_workers=2)
class Flower_Net(nn.Module):

    def __init__(self):

        super(Flower_Net, self).__init__()

        self.conv1 = nn.Conv2d(3,6,3)

        self.linear1 = nn.Linear(6*49*49, 1000)

        self.linear2 = nn.Linear(1000, 500)

        self.linear3 = nn.Linear(500, 102)

        

    def forward(self,x):

#         print(x)

        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.linear1(x))

        x = F.relu(self.linear2(x))

        x = self.linear3(x)

        return x

        

    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

#         print(size)

        num_features = 1

        for s in size:

            num_features *= s

        return num_features
net = Flower_Net()

print(net)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
params = list(net.parameters())

print(len(params))

print(params[2].size()) 
input = torch.randn(1, 3, 100, 100)

out = net(input.to(device))

print(out)
dataiter = iter(train_dataloader)

sample = dataiter.next()
def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()
imshow(utils.make_grid(sample['image']))

print(' '.join('%5s' % sample['label'][j] for j in range(10)))
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=1e-3,  momentum=0.9)
print("Starting Training....")

for epoch in range(10):

    

    running_loss=0.0

    

    for i, data in enumerate(train_dataloader, 0):

        

        inputs = data['image'].to(device, dtype=torch.float)

        label = data['label'].to(device)

        

        optimizer.zero_grad()

        

        outputs = net(inputs)

#         print(len(outputs))

        loss = criterion(outputs, label)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        if i % 200 == 199: 

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 200))

            running_loss = 0.0



print('Finished Training')
testdataiter = iter(train_dataloader)

sampletest = testdataiter.next()
classes = np.arange(0,102,1)
print('GroundTruth: ', ' '.join('%5s' % classes[sampletest['label'][j]] for j in range(10)))
outputs = net(sampletest['image'].float().to(device))
_, predicted = torch.max(outputs, 1)



print('Predicted: ', ' '.join('%5s' % classes[ predicted[j]]

                              for j in range(10)))
correct = 0

total = 0

with torch.no_grad():

    for data in train_dataloader:

        images = data['image'].to(device)

        labels = data['label'].to(device)

        outputs = net(images.float())

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()



print('Accuracy of the network on the 18000 train images: %d %%' % (

    100 * correct / total))
class_correct = list(0. for i in range(102))

class_total = list(0. for i in range(102))

with torch.no_grad():

    for data in train_dataloader:

        images = data['image'].to(device)

        labels = data['label'].to(device)

        outputs = net(images.float())

        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()

        for i in range(10):

            label = labels[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(102):

    print('Accuracy of %5s : %2d %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))