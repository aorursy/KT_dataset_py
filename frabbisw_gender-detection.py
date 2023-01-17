import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
transform = transforms.Compose(
    [transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset=torchvision.datasets.ImageFolder("../input/dataset1/dataset1/train/",transform=transform,)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset=torchvision.datasets.ImageFolder("../input/dataset1/dataset1/test/",transform=transform,)
testloader=torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model=Model().cuda()
criterion=nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=.001)
for i in range(100000):
    trainiter = iter(trainloader)
    x, y = trainiter.next()
    x=x.cuda()
    y=y.cuda()
    z=model(x)

    loss=criterion(z,y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 10000 == 0:
        print(loss)
dataiter = iter(testloader)
images, labels = dataiter.next()

dataiter = iter(testloader)
images, labels = dataiter.next()
classes=('Man','Woman')
# print images
imshow(torchvision.utils.make_grid(images))

images=images.cuda()
preds=model(images)
_,indices=preds.max(1)
preds=indices.tolist()

print('Real: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
print('Pred: ', ' '.join('%5s' % classes[preds[j]] for j in range(4)))