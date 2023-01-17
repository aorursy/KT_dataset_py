import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import matplotlib.pyplot as plt



plt.ion()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device %r' % device)

batch_size = 64

num_workers = 4
class MNIST(torch.utils.data.Dataset):



    def __init__(self, path, train=True, transform=None):

        self.train = train

        # I don't like pandas ---

        # It's all coarse, and

        # Rough, and irritating.

        # And it gets everywhere.

        with open(path) as file:

            _, *lines = file.readlines()

        if train:

            labels = []

        datapoints = []

        for line in lines:

            if train:

                label, *data = line.split(',')

                label = int(label)

                labels.append(label)

            else:

                data = line.split(',')

            data = np.array([int(d) for d in data], dtype=np.float).reshape((28, 28))

            if transform is not None:

                data = transform(data)

            datapoints.append(data.tolist())

        if train:

            self.labels = torch.tensor(labels, dtype=torch.long)

        self.datapoints = torch.tensor(datapoints, dtype=torch.float)



    def __len__(self):

        return self.datapoints.size(0)



    def __getitem__(self, key):

        datapoint = self.datapoints[key]

        if self.train:

            label = self.labels[key]

            return datapoint, label

        else:

            return datapoint
transform = torchvision.transforms.Compose([

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),

])



trainset = MNIST('/kaggle/input/digit-recognizer/train.csv', train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,

                                          shuffle=True, num_workers=num_workers,

                                          pin_memory=True)



testset = MNIST('/kaggle/input/digit-recognizer/test.csv', train=False, transform=transform)

testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,

                                         shuffle=True, num_workers=num_workers,

                                         pin_memory=True)
dataiter = iter(trainloader)

images, labels = dataiter.next()

plt.imshow(torchvision.utils.make_grid(images).numpy().transpose((1, 2, 0)) / 2 + 0.5,

           cmap='gray')

plt.title(', '.join(str(label) for label in labels.tolist()))
class Model(torch.nn.Module):

    

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = torch.nn.Sequential(

            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),

            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(kernel_size=2, stride=2),

        )

        self.dense = torch.nn.Sequential(

            torch.nn.Linear(14 * 14 * 128, 1024),

            torch.nn.ReLU(),

            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(1024, 10),

        )

    def forward(self, x):

        x = self.conv1(x)

        x = x.view(-1, 14 * 14 * 128)

        x = self.dense(x)

        return x



model = Model()

model.to(device)

model
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Somehow this is not fitting as good as my machine

# [It works on my machine]

# It can get as high as 99.9 or as low as 99.7

# But no difference

# Should be stable with more epochs

for epoch in range(20):

    running_total = running_errors = running_loss = 0

    for i, (images, labels) in enumerate(trainloader):

        optimizer.zero_grad()

        outputs = model(images.to(device))

        loss = criterion(outputs, labels.to(device))

        loss.backward()

        optimizer.step()



        running_total += labels.size(0)

        running_errors += (outputs.cpu().max(1).indices != labels).sum().item()

        running_loss += loss.item()

        if i == 0 and epoch > 0:

            print()

        if i % 10 == 9:

            print(f'\rEpoch {epoch + 1}, sample {(i + 1) * batch_size:6d}, '

                  f'loss {running_loss / running_total:.5f}, '

                  f'acc {(1 - running_errors / running_total) * 100:2.2f}')

            running_total = running_errors = running_loss = 0
torch.save(model.state_dict(), 'model.pt')

!ls -lah
model.eval()

total = 0

corrects = 0

for i, (images, labels) in enumerate(trainloader):

    total += labels.size(0)

    corrects += (model(images.to(device)).argmax(1) == labels.to(device)).sum().cpu().item()

    print('\r' + str(i), end='')

print()

print(corrects / total)
dataiter = iter(testloader)

images, labels = dataiter.next()

images = images[:8, :, :, :]

labels = labels[:8]

plt.imshow(torchvision.utils.make_grid(images).numpy().transpose((1, 2, 0)) / 2 + 0.5,

           cmap='gray')

plt.title(', '.join(str(label) for label in labels.tolist()))

preds = ', '.join(str(pred.item()) for pred in model(images.to(device)).argmax(1).cpu())

print(f'Predictions: {preds}')
with open('/kaggle/working/predictions.csv', 'w') as file:

    file.write('ImageId,Label\n')

    for i, image in enumerate(testset, 1):

        file.write('%d,%d\n' % (i, model(image.to(device).unsqueeze(0)).argmax().item()))

        print('\rWritten row %5d' % i, end='')