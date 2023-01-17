import torch

from torch import nn

import pathlib

from torch.utils.data import DataLoader

from torchvision import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformtrain = transforms.Compose([

    transforms.Resize((150, 150)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((.5, .5, .5), (.5, .5, .5))

])

transformtest = transforms.Compose([

    transforms.Resize((150, 150)),

    transforms.ToTensor(),

    transforms.Normalize((.5, .5, .5), (.5, .5, .5))

])
trainds = datasets.ImageFolder('../input/seg_train/seg_train', transform=transformtrain)

testds = datasets.ImageFolder('../input/seg_test/seg_test', transform=transformtest)
trainloader = DataLoader(trainds, batch_size=256, shuffle=True)

testloader = DataLoader(testds, batch_size=64, shuffle=False)
root = pathlib.Path('../input/seg_train/seg_train')

classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
model = models.vgg19(pretrained=True).to(device)

for param in model.features.parameters():

    param.requires_grad = False
model
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(classes)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)
trainlosses = []

testlosses = []

for e in range(50):

    trainloss = 0

    traintotal = 0

    trainsuccessful = 0

    for traininput, trainlabel in trainloader:

        traininputs, trainlabels = traininput.to(device), trainlabel.to(device)

        optimizer.zero_grad()

        trainpredictions = model(traininputs)

        _, trainpredict = torch.max(trainpredictions.data, 1)

        loss = criterion(trainpredictions, trainlabels)

        loss.backward()

        optimizer.step()

        trainloss += loss.item()

        traintotal += trainlabels.size(0)

        trainsuccessful += (trainpredict == trainlabels).sum().item()

    else:

        testloss = 0

        testtotal = 0

        testsuccessful = 0

        with torch.no_grad():

            for testinput, testlabel in testloader:

                testinputs, testlabels = testinput.to(device), testlabel.to(device)

                testpredictions = model(testinputs)

                _, testpredict = torch.max(testpredictions.data, 1)

                tloss = criterion(testpredictions, testlabels)

                testloss += tloss.item()

                testtotal += testlabels.size(0)

                testsuccessful += (testpredict == testlabels).sum().item()

        trainlosses.append(trainloss/len(trainloader))

        testlosses.append(testloss/len(testloader))

        print('Train Accuracy %{:.2f}'.format(100*trainsuccessful/traintotal))

        print('Test Accuracy %{:.2f}'.format(100*testsuccessful/testtotal))

import matplotlib.pyplot as plt
plt.plot(trainlosses, label='Training loss', color='green')

plt.plot(testlosses, label='Validation loss', color='black')

plt.legend(frameon=False)

plt.show()
!ls ../input/seg_pred/seg_pred/3966.jpg
from PIL import Image

import numpy as np
img = Image.open('../input/seg_pred/seg_pred/3966.jpg')
nimg = np.array(img)
plt.imshow(nimg)
pimg = transformtest(img).unsqueeze(0).to(device)
pimg.shape
prediction = model(pimg)
_, tpredict = torch.max(prediction.data, 1)
classes[tpredict[0].item()]