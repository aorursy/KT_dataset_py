%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader

from torchvision import transforms, datasets

import torch.optim as optim



from PIL import Image

import matplotlib.pyplot as plt



import numpy as np



use_cuda = torch.cuda.is_available()
im_normal = Image.open('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0651-0001.jpeg')

im_p1 = Image.open('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1139_bacteria_3082.jpeg')

im_p2 = Image.open('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1139_virus_1882.jpeg')



plt.subplot(1, 3, 1)

plt.imshow(im_normal)

plt.title('Normal chest xray')



plt.subplot(1, 3, 2)

plt.imshow(im_p1)

plt.title('Pneumonia chest xray');



plt.subplot(1, 3, 3)

plt.imshow(im_p2)

plt.title('Pneumonia chest xray');
train_transform = transforms.Compose([transforms.Resize((224, 224)),

                                      #transforms.RandomRotation(20),

                                      #transforms.RandomHorizontalFlip(),

                                      #transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize((.5,.5,.5),

                                                           (.5,.5,.5))])



test_transform = transforms.Compose([transforms.Resize((224, 224)),

                                    #transforms.CenterCrop(224),

                                    transforms.ToTensor(),

                                    transforms.Normalize((.5,.5,.5),

                                                         (.5,.5,.5))])
# datasets

data_dir = '../input/chest-xray-pneumonia/chest_xray/'



train_data = datasets.ImageFolder(data_dir + 'train', train_transform)

test_data = datasets.ImageFolder(data_dir + 'test', test_transform)

valid_data = datasets.ImageFolder(data_dir + 'val', test_transform)
train_loader = DataLoader(train_data, batch_size=64, num_workers=2, shuffle=True, pin_memory=True)

test_loader = DataLoader(test_data, batch_size=64, num_workers=2, shuffle=True, pin_memory=True)

valid_loader = DataLoader(valid_data, batch_size=64, num_workers=2, shuffle=False, pin_memory=True)
class XRayNet(nn.Module):

    def __init__(self):

        super(XRayNet, self).__init__()

        

        # input image size 3, 224, 224

        self.conv1 = nn.Conv2d(3, 32, 3, stride = 1, padding = 1)

        self.batch1 = nn.BatchNorm2d(32, affine=True, track_running_stats=True)

        

        # image size 8, 112, 112

        self.conv2 = nn.Conv2d(32, 56, 3, stride = 1, padding = 1)

        self.batch2 = nn.BatchNorm2d(56, affine=True, track_running_stats=True)

        

        # image size 16, 56, 56

        self.conv3 = nn.Conv2d(56, 64, 3, stride = 1, padding = 1)

        self.batch3 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        

        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)

        self.batch4 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        

        self.pool = nn.MaxPool2d(2, 2)

        

        # input size 32, 28, 28

        self.fc1 = nn.Linear(64 * 14 * 14, 4096)

        self.fc2 = nn.Linear(4096, 512)

        self.fc3 = nn.Linear(512, 64)

        self.fc4 = nn.Linear(64, 2)

        

        self.batch5 = nn.BatchNorm1d(512)

        self.batch6 = nn.BatchNorm1d(64)

        

        self.drop = nn.Dropout(p=.3)

        

    def forward(self, x):

        

        x = F.relu(self.conv1(x), inplace=True)

        x = self.pool(self.batch1(x))

        

        x = F.relu(self.conv2(x), inplace=True)

        x = self.pool(self.batch2(x))

        x = self.drop(x)

        

        x = F.relu(self.conv3(x), inplace=True)

        x = self.pool(self.batch3(x))

        x = self.drop(x)

        

        x = F.relu(self.conv4(x), inplace=True)

        x = self.pool(self.batch4(x))

        

        x = x.view(-1, 64 * 14 * 14)

        

        x = F.relu(self.fc1(x))

        x = self.drop(x)

        x = F.relu(self.fc2(x))

        x = self.batch5(x)

        x = self.drop(x)

        x = F.relu(self.fc3(x))

        x = self.batch6(x)

        x = self.drop(x)

        x = self.fc4(x)

        

        return x
XRayModel = XRayNet()



if use_cuda:

    XRayModel.cuda()



XRayModel
criterion = nn.CrossEntropyLoss()



optimizer = optim.AdamW(XRayModel.parameters(), lr=0.001)
n_epoch = 30

min_val_loss = np.Inf



train_losses = []

val_losses = []



for e in range(n_epoch):

    

    running_loss = 0

    val_loss = 0

    

    # train mode

    for images, labels in train_loader:

        if use_cuda:

            images, labels = images.cuda(), labels.cuda()

            

        # zero grad

        optimizer.zero_grad()

        

        output = XRayModel(images)

        

        loss = criterion(output, labels)

        

        running_loss += loss.item() * images.size(0)

        

        loss.backward()

        

        optimizer.step()

        

    # valid mode

    for images, labels in valid_loader:

        if use_cuda:

            images, labels = images.cuda(), labels.cuda()

            

        XRayModel.eval()

        

        output = XRayModel(images)

        

        loss = criterion(output, labels)

        

        val_loss += loss.item() * images.size(0)

        

    XRayModel.train()

    

    epoch_train_loss = running_loss / len(train_loader.dataset)

    epoch_val_loss = val_loss / len(valid_loader.dataset)

    print('Epoch {}, train loss : {}, validation loss :{}'.format(e, epoch_train_loss, epoch_val_loss))

    

    train_losses.append(epoch_train_loss)

    val_losses.append(epoch_val_loss)

    

    if epoch_val_loss <= min_val_loss:

        print('Validation loss decreased {} -> {}. Saving model...'.format(min_val_loss, epoch_val_loss))

        min_val_loss = epoch_val_loss

        torch.save(XRayModel.state_dict(), 'ashish_best.pth')
XRayModel1 = XRayNet()

XRayModel1.load_state_dict(torch.load('ashish_best.pth'))
if use_cuda:

    XRayModel1.cuda()
def imshow(img):

    img = img / 2 + 0.5     # unnormalize

    npimg = img.numpy()

    plt.figure(figsize=(20, 15))

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()
import torchvision

dataiter = iter(test_loader)

images, labels = dataiter.next()



# print images

imshow(torchvision.utils.make_grid(images[:24]))

print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(24)))
from ignite.metrics import Accuracy, Precision, Recall



def thresholded_output_transform(output):

    y_pred, y = output

    y_pred = torch.round(y_pred)

    return y_pred, y



binary_accuracy = Accuracy(thresholded_output_transform)

precision = Precision(thresholded_output_transform)

recall = Recall(thresholded_output_transform)
def CalculateMetrics(XRayModel1):

    with torch.no_grad():

        for images, labels in test_loader:

            if use_cuda:

                images, labels = images.cuda(), labels.cuda()

            outputs = XRayModel1(images)

            _, predicted = torch.max(outputs.data, 1)

            binary_accuracy.update((predicted, labels))

            precision.update((predicted, labels))

            recall.update((predicted, labels))

            

        print('Model accuracy : ', binary_accuracy.compute())

        print('Model Precision : ', precision.compute().item())

        print('Model Recall : ', recall.compute().item())
CalculateMetrics(XRayModel1)