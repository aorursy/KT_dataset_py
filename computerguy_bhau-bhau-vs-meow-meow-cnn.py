import glob
from PIL import Image
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
*1
transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Resize((256,256)),
    transforms.RandomAffine(20),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
!unzip /kaggle/input/dogs-vs-cats/train.zip -d ./train/ >> faltuKachra
class CustomedDataSet(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train : 
            cats = glob.glob('./train/train/cat*.jpg')
            dogs = glob.glob('./train/train/dog*.jpg')
            cats = cats[:-int((1/9)*len(cats))]
            dogs = dogs[:-int((1/9)*len(dogs))]
            c = []
            y = []
            ct = [1 for x in range(len(cats))]
            dt = [0 for x in range(len(dogs))]
            self.trainX  = cats+dogs
            self.trainY  = torch.Tensor(ct+dt).long()
        else:
            cats = glob.glob('./train/train/cat*.jpg')
            dogs = glob.glob('./train/train/dog*.jpg')
            cats = cats[-int((1/9)*len(cats)):]
            dogs = dogs[-int((1/9)*len(dogs)):]
            c = []
            y = []
            ct = [1 for x in range(len(cats))]
            dt = [0 for x in range(len(dogs))]
            self.trainX  = cats+dogs
            self.trainY  = torch.Tensor(ct+dt).long()
            
    def __getitem__(self, index):    
#         if(self.train)
        k = torch.Tensor(transform(Image.open(self.trainX[index])))
        return (k,self.trainY[index])
    
    def __len__(self):
#         if(self.train):
        return len(self.trainX)

train_dataset = CustomedDataSet()

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=2)

val_dataset = CustomedDataSet(train=False)

valloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=10,
                                           shuffle=True,
                                           num_workers=2)

a = transform(img)
# show(transform(a))
plt.imshow(np.transpose(a.cpu().numpy(),(1,2,0)))
net
net = models.alexnet(pretrained=True).cuda()

net.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)
        ).cuda()
import torch.optim as optim

criterion = nn.MSELoss()
# criterion = nn.L1Loss()

# criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.001,amsgrad=True)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda().float().unsqueeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #show(inputs)
        #plt.show()
        #         print(outputs[0])
        # print statistics
        #         print(loss.item())
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200),end=" ")
            running_loss = 0.0
            i = 0
            for data in valloader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda().float().unsqueeze(1)
                _ = net.eval()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss+=loss.item()
                i+=1
            print("Validation Loss: %0.5f"%(running_loss/i))
            running_loss = 0.0

print('Finished Training')