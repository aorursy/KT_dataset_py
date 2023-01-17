import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
import sys
import torch
import torchvision
sys.path.append('../input/caltech256')
from caltech256 import Caltech256
from torchvision.transforms import transforms

root = '../input/caltech256/256_objectcategories/256_ObjectCategories/'
csv_path = '../input/caltech256/caltech256.csv'
batch_size = 32

cal_transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = Caltech256(root=root, transform=cal_transform, splits=[1,2,3,4], csv_path=csv_path)
testset = Caltech256(root=root, transform=cal_transform, splits=[5], csv_path=csv_path)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
dataiter = iter(trainloader)
images, labels = dataiter.next()
classes = trainset.classNames

# show images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
cache_dir = os.path.expanduser(os.path.join('~', '.torch'))
models_dir = os.path.join(cache_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
!cp ../input/pretrained-pytorch-models/* ~/.torch/models/
from torchvision import models
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pretrained Inceptionv3
model = models.inception_v3(pretrained=True)

# Freeze some layers
for param in model.parameters():
    param.requires_grad = False

# Replace last linear layers
num_ftrs = model.AuxLogits.fc.in_features
model.AuxLogits.fc = nn.Linear(num_ftrs, trainset.nclasses)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, trainset.nclasses)
model = model.to(device)

# Training criterion
weights = trainset.weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Let the gradient flow
for param in model.parameters():
    param.requires_grad = True

# Loop over the dataset multiple times
for epoch in range(1):
    # train set
    running_loss = 0.0
    running_corrects = 0.0
    model.train()
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # get the inputs
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, aux_outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4*loss2
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)

        # print statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f acc: %.3f' %
                  (epoch + 1, i, running_loss / 100, 
                   running_corrects.double() / 100 / 32 * 100))
            running_loss = .0
            running_corrects = .0

    # test set
    running_corrects = .0
    model.eval()
    for i, (inputs, labels) in enumerate(testloader, 0):
        # get the inputs
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    epoch_acc = running_corrects.double() / len(testset) * 100  
    print('val acc:' + str(epoch_acc))

import torch.nn.functional as F

def mModel(x):
    if model.transform_input:
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    # 299 x 299 x 3
    x = model.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = model.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = model.Conv2d_2b_3x3(x)
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)
    x = model.Mixed_6a(x)
    # 17 x 17 x 768
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)
    if model.training and model.aux_logits:
        aux = model.AuxLogits(x)
    x = model.Mixed_7a(x)
    # 8 x 8 x 1280
    x = model.Mixed_7b(x)
    x = cam = model.Mixed_7c(x)
    x = F.avg_pool2d(x, kernel_size=8)
    # 1 x 1 x 2048
    x = F.dropout(x, training=model.training)
    x = x.view(x.size(0), -1)
    x = model.fc(x)
    # 1000 (num_classes)
    if model.training and model.aux_logits:
        return x, aux, cam
    return x, cam

    
def get_preds_and_cams(imgs):
    # check input
    if len(imgs.shape) == 3:
        imgs = imgs.reshape(1,3,299,299)
    
    # get maps
    model.eval()
    preds, maps = mModel(imgs.to(device))
    
    # get cams
    W = model.fc.weight.cpu().detach().numpy()
    maps = maps.cpu().detach().numpy()
    maps = np.transpose(maps, (0,2,3,1))
    cams = np.dot(maps, W.T)
    
    return preds, cams

def show_cams(pyImg, cams, preds):
    # show main image
    img = pyImg.detach().numpy()
    img = np.transpose(img, (1,2,0))
    img = img / 2 + 0.5
    plt.imshow(img)
    
    # show cams
    pred = preds.cpu().detach().numpy().argmax()
    cam = cams[:, :, pred]
    cam = resize(cam, (299,299), preserve_range=True)
    plt.imshow(cam, alpha=0.7)
    plt.show()

    
from skimage.transform import resize

for i in range(50):
    pyImg, y = testset[i]
    
    preds, cams = get_preds_and_cams(pyImg)
    show_cams(pyImg, cams[0], preds[0])


torch.save(model.state_dict(), 'model')
model2 = models.Inception3()

# Replace last linear layers
num_ftrs = model2.AuxLogits.fc.in_features
model2.AuxLogits.fc = nn.Linear(num_ftrs, trainset.nclasses)
num_ftrs = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs, trainset.nclasses)
model2 = model.to(device)

# load weights
model2.load_state_dict(torch.load('model'))
model2.eval()
