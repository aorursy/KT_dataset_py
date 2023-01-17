import glob
from PIL import Image
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import glob
print(glob.glob("../input/intel-image-classification/seg_train/seg_train/*"))
classes = ["buildings","forest","glacier","mountain","sea","street"]
# classes = {'buildings': 0,
#             'forest': 1,
#             'glacier': 2,
#             'mountain': 3,
#             'sea': 4,
#             'street' 5 }
class CustomedDataSet(torch.utils.data.Dataset):
    def __init__(self, train=True,trans=""):
        self.train = train
        global classes
        self.transform = trans
        if self.train :
            trainPath = "../input/intel-image-classification/seg_train/seg_train/"
            self.x = []
            self.y = []
            
            for i,x in enumerate(classes):
                L =  [0 for h in range(len(classes))]
#                 print(de)
                classesImg  = glob.glob(trainPath+classes[i]+"/*.jpg")
                self.x+= classesImg
#                 L[i] = 1
                self.y+=[i]*len(classesImg)
        else:
            trainPath = "../input/intel-image-classification/seg_test/seg_test/"
            self.x = []
            self.y = []
            
            for i,x in enumerate(classes):
                L =  [0 for h in range(len(classes))]
#                 print(de)
                classesImg  = glob.glob(trainPath+classes[i]+"/*.jpg")
                self.x+= classesImg
#                 L[i] = 1
                self.y+=[i]*len(classesImg)
    def __getitem__(self, index):    
        k = torch.Tensor(self.transform(Image.open(self.x[index])))
        return k,(np.array(self.y[index]))
    
    def __len__(self):
        return len(self.y)

transform = transforms.Compose([
    transforms.RandomAffine(2,shear=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.Resize((150,150)),
    transforms.ToTensor()])
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    torchvision.transforms.Resize((150,150)),
    transforms.ToTensor()])
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
train_dataset = CustomedDataSet(trans = transform)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=2)
val_dataset = CustomedDataSet(train = False,trans = test_transform)

valloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=8,
                                           num_workers=2)
def scale(vect):
    vect = vect.detach()
    vect-=vect.min()
    vect/=vect.max()
    return vect
def show(img):
    img = img.detach().numpy()
    img = np.transpose(img,(1,2,0))
    plt.imshow(img)
    plt.show()
    return img
#Test Run
for x in trainloader:
    _ = show((x[0][1]))
    break

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.b1 = nn.BatchNorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.b2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.b3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.b4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.b5 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 5 * 5, 1850)
        self.fc2 = nn.Linear(1850, 185)
        self.fc3 = nn.Linear(185, 84)
        self.fc4 = nn.Linear(84, 6)
        self.m = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.b1(x)
        x = self.m(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.m(x)
        x = self.b2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.m(x)
        x = self.b3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.m(x)
        x = self.b4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.m(x)
        x = self.b5(x)
        
#         print(x.shape)
        
        x = x.view(-1, 128 * 5 * 5)
        x = self.m(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net().cuda()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    train_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        _ = net.train()
        inputs, labels = data
        inputs, labels = inputs.cuda(),labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
#         print(loss.item())
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200,train_accuracy/200),end=" ")
            running_loss = 0.0
            train_accuracy = 0.0
#         if(i%300==299):
            _ = net.eval()
            val_loss = 0
            val_accuracy = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(),labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss+=loss.item()
                val_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
            print("Validation Loss[%0.3f] Accuracy[%0.2f]"%(val_loss/len(valloader),val_accuracy/len(valloader)))
print('Finished Training')
for x in range(2):
    _ = plt.figure(figsize=(10,10))
    show(((inputs[x])).cpu())
    print(classes[outputs.argmax(1)[x].cpu()])
    print(classes[labels[x]])
torch.save(net.state_dict(), "Val 83 and Train 89.pth")
labels
outputs.argmax(1)
ae
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        #ENCODING
        self.fc1 = nn.Conv2d(3 ,16, kernel_size=2,padding=2)
        # self.fc2 = nn.BatchNorm2d(16)
        self.fc3 = nn.ReLU()
#         self.fc4 = nn.MaxPool2d(2,return_indices=True)
        self.fc5 = nn.Conv2d(16, 32, kernel_size=2,padding=2)
        # self.fc6 = nn.BatchNorm2d(32)
        self.fc7 = nn.ReLU()
        self.fc8 = nn.MaxPool2d(4,return_indices=True)
        
        #FLATTEN
        self.mid1 = nn.Linear(48672, 20)
        self.mid2 = nn.Linear(20,48672)

        #DECODING
        self.fc9 = nn.MaxUnpool2d(4)
        self.fc10 = nn.ReLU()
        self.fc11 = nn.ConvTranspose2d(32, 16, kernel_size=2,padding=2)
#         self.fc12 = nn.MaxUnpool2d(2)
        self.fc13 = nn.ReLU()
        self.fc14 = nn.ConvTranspose2d(16 ,3, kernel_size=2,padding=2)
        # self.fc6 = nn.BatchNorm2d(32)
        # self.fc9 = nn.ConvTranspose2d()
        
    def forward(self, x,mode = "train",ind1=None,ind2=None,y = None):
        p = torch.Tensor([1, 32, 7, 7])
        #Encoding
        
        x = self.fc1(x)
        x = self.fc3(x)
#         x,ind1 = self.fc4(x)
        size1 = x.size()
        
        x = self.fc5(x)
        x = self.fc7(x)
        x,ind2 = self.fc8(x)
        size2 = x.size()
        
        self.ind1 = ind1
        self.ind2 = ind2
        
        sh = (x.shape)
        x = x.view(x.size(0), -1)
#         print(x.shape)
        x = self.mid1(x)
        if(mode=="test"):
            return x
        x = self.mid2(x)
        x = x.view(sh)
#         print(x.shape)
        #Decoding
        x = self.fc9(x,ind2)
#         print(x.shape)
        x = self.fc10(x)
        x = self.fc11(x)
        
#         print(x.shape)
#         x = self.fc12(x,ind1)
        x = self.fc13(x)
        x = self.fc14(x)
        return x

ae = AutoEncoder().cuda()
ae.load_state_dict(torch.load("AE_TrainOnly.pth"))
ae.eval()
import torch.optim as optim

criterion = nn.L1Loss()
optimizer = optim.Adam(ae.parameters(), lr=0.0003)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    train_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        _ = ae.train()
        inputs, labels = data
        inputs, labels = inputs.cuda(),labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = ae(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
#         print(loss.item())
#         train_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
#         print(loss.item())
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200,train_accuracy/200),end=" ")
            running_loss = 0.0
            train_accuracy = 0.0
#         if(i%300==299):
            _ = ae.eval()
            val_loss = 0
            val_accuracy = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(),labels.cuda()
                outputs = ae(inputs)
                val_loss+=loss.item()
#                 val_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
            print("Validation Loss[%0.3f] Accuracy[%0.2f]"%(val_loss/len(valloader),val_accuracy/len(valloader)))
print('Finished Training')
torch.save(ae.state_dict(), "AE_Trained.pth")
ae.load_state_dict(torch.load("AE_TrainOnly.pth"))
_ = ae.eval()
ae.trainable = False
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(1, 1)
        self.b1 = nn.BatchNorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.b2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.b3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.b4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.b5 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(4200, 1850)
        self.fc2 = nn.Linear(1850, 185)
        self.fc3 = nn.Linear(185, 84)
        self.fc4 = nn.Linear(84, 6)
        self.m = nn.Dropout(p=0.2)
        
    def forward(self, x):
        global ae
        x1 = ae(x,mode="test")
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.b1(x)
        x = self.m(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.m(x)
        x = self.b2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.m(x)
        x = self.b3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.m(x)
        x = self.b4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.m(x)
        x = self.b5(x)
        
#         print(x.shape)
        
        x = x.view(-1, 128 * 5 * 5)
        x = torch.cat((x,x1),1)
#         print(x.shape)
        x = self.m(x)
        x = F.relu(self.fc1(x))
        x = self.m(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = Net().cuda()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    train_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        _ = net.train()
        inputs, labels = data
        inputs, labels = inputs.cuda(),labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
#         print(loss.item())
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200,train_accuracy/200),end=" ")
            running_loss = 0.0
            train_accuracy = 0.0
#         if(i%300==299):
            _ = net.eval()
            val_loss = 0
            val_accuracy = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(),labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss+=loss.item()
                val_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
            print("Validation Loss[%0.3f] Accuracy[%0.2f]"%(val_loss/len(valloader),val_accuracy/len(valloader)))
print('Finished Training')
alexnet = models.alexnet()
class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        global ae
        x1= ae(x,mode="test")
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = torch.cat((x,x1),1)
#         print(x.shape)
#         x = torch.cat((x,ae(x,mode="test")),1)
        x = self.classifier(x)
        return x

model = AlexNet().cuda()
state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',progress=True)
model.load_state_dict(state_dict)
model.classifier =nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 + 20, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, len(classes)),
        ).cuda()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
best = None
maxa = 0
import copy
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    train_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        _ = net.train()
        inputs, labels = data
        inputs, labels = inputs.cuda(),labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
#         print(loss.item())
        if i % 200 == 199:    # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200,train_accuracy/200),end=" ")
            running_loss = 0.0
            train_accuracy = 0.0
#         if(i%300==299):
            _ = model.eval()
            val_loss = 0
            val_accuracy = 0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.cuda(),labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss+=loss.item()
                val_accuracy+=accuracy_score(labels.cpu().detach(),outputs.argmax(1).cpu())
            if(maxa<(val_accuracy/len(valloader))):
                maxa = (val_accuracy/len(valloader))
                best = copy.deepcopy(model)
            print("Validation Loss[%0.3f] Accuracy[%0.2f]"%(val_loss/len(valloader),val_accuracy/len(valloader)))
print('Finished Training')

torch.save(ae.state_dict(), "ae.pth")
torch.save(model.state_dict(), "model.pth")