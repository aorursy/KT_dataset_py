!pip install matplotlib pandas

!pip install --upgrade https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl

!pip install --upgrade torchvision
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Any results you write to the current directory are saved as output.

import torch

from torch import nn

from torch.nn.parameter import Parameter

import torch.optim as optim

import torch.nn.functional as F

from matplotlib import pyplot as plt

import time

from PIL import Image



import torchvision

from torchvision import transforms



if torch.cuda.is_available():

    use_cuda=True

else:

    use_cuda=False



device = torch.device("cuda:0" if use_cuda else "cpu")

print(device)
data_train = torchvision.datasets.FashionMNIST("./", train=True, transform=None, target_transform=None, download=True)

data_test = torchvision.datasets.FashionMNIST("./", train=False, transform=None, target_transform=None, download=True)
df_train = pd.DataFrame(np.hstack((data_train.train_labels.reshape(-1, 1), data_train.train_data.reshape(-1,28*28))))

df_test = pd.DataFrame(np.hstack((data_test.test_labels.reshape(-1, 1), data_test.test_data.reshape(-1,28*28))))

df_train.head()
lbl_map = {

    0 : "T-shirt/top",

    1 : "Trouser",

    2 : "Pullover",

    3 : "Dress",

    4 : "Coat",

    5 : "Sandal",

    6 : "Shirt",

    7 : "Sneaker",

    8 : "Bag",

    9 : "Ankle boot"

}
#plot.bar(figsize=(16,6), color="b")

cnt_df = df_train.iloc[:,0].value_counts()

cnt_df.rename(lbl_map, axis='index').plot.bar(figsize=(16,6), color="b")
def random_blackout(img, input_dim=(28, 28)):

    min_side = np.min(input_dim)

    high = np.round(min_side * .60)

    low = np.round(min_side * .15)

    # height, width

    h, w = np.random.randint(high=high, low=low, size=(2))



    # offsets top and left

    ofs_t = np.random.randint(high=input_dim[0]-h, low=0, size=1)[0]

    #ofs_t = 0

    ofs_l = np.random.randint(high=input_dim[1]-w, low=0, size=1)[0]

    #ofs_l = 0



    mask = np.ones(input_dim)



    mask[ofs_t:ofs_t+h,ofs_l:ofs_l+w] = 0



    return img * mask



class BlackoutTransform():

    def __init__(self):

        """

        """

        

    def __call__(self, img):

        img_dim = img.shape

        np_arr = img.view(28,28).numpy()

        np_arr = random_blackout(np_arr, np_arr.shape)

        return torch.FloatTensor(np_arr).view(img_dim)



class ReshapeTransform():

    def __init__(self, new_size):

        """

        :new_size: tuple

        """

        self.new_size = new_size



    def __call__(self, img):

        """Reshape an image

        :img: ex 1x28x28

        :returns: reshaped tensor

        """

        return torch.reshape(img, self.new_size)

    



t = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),

                        #transforms.RandomVerticalFlip(p=0.5),

                        transforms.Pad((5,6)),

                        transforms.RandomCrop(size=28, padding_mode="reflect"),

                        #transforms.RandomCrop(size=28),

                        #transforms.RandomAffine([0,180], translate=None, scale=None, shear=None, resample=False, fillcolor=0),

                        # Resize random crop, then pad

                        #transforms.RandomResizedCrop(28, scale=(1.1, 1.3), ratio=(1.1, 1.5), interpolation=2),

                        transforms.ToTensor(),

                        BlackoutTransform(),

                        ReshapeTransform((1, 28, 28))

])



class CustomTensorDataset(torch.utils.data.TensorDataset):

    def __init__(self, *tensors, transforms=None):

        self.transform = transforms

        super().__init__(*tensors)

    

    def __getitem__(self, index):

        img, target = self.tensors[0][index], self.tensors[1][index]

        

        # doing this so that it is consistent with all other datasets

        # to return a PIL Image

        img = Image.fromarray(np.uint8(img.view(28,28).numpy()), mode='L')

    

        if self.transform is not None:

            img = self.transform(img)

                

        return (img, target)
def exclude_class(inputs, lbls, class_lbl):

    indices = np.where(lbls != class_lbl)

    inputs = np.take(inputs, indices[0], axis=0)

    lbls = np.take(lbls, indices[0], axis=0)

    return inputs, lbls

# Train data

train_X = df_train.iloc[:,1:]

train_Y = df_train.iloc[:,:1]

# Test data

test_X = df_test.iloc[:,1:]

test_Y = df_test.iloc[:,:1]



#train_X, train_Y = exclude_class(train_X, train_Y, 2)

#test_X, test_Y = exclude_class(test_X, test_Y, 2)



# Normalize data to [0,1]

#fmnist_train = torch.utils.data.TensorDataset(torch.FloatTensor(train_X.values/255).view(-1,1,28,28), torch.LongTensor(train_Y.values).view(-1))

fmnist_train = CustomTensorDataset(torch.FloatTensor(train_X.values), torch.LongTensor(train_Y.values).view(-1), transforms=t)

fmnist_test = torch.utils.data.TensorDataset(torch.FloatTensor(test_X.values/255).view(-1,1,28,28), torch.LongTensor(test_Y.values).view(-1))
def show_images(images):

    n_images = len(images)

    w, h = 10, 10

    columns = 10

    rows = n_images / columns

    fig=plt.figure(figsize=(columns*2, rows*2))

    for i in range(n_images):

        img = images[i].reshape(28,28)

        fig.add_subplot(rows, columns, i+1)

        plt.imshow(img, cmap='gray')

        plt.axis('off')

    plt.show()





examples = []

# Take 10 of each lbl class

for k in lbl_map.keys():

    indices = np.where(train_Y == k)

    images = np.take(train_X.values, indices[0][:10], axis=0)

    examples = examples + [x for x in images]

    

#examples = [x for x in examples for y in examples[]]



print(lbl_map)

show_images(examples)



print("Preprocessed random 40:")

examples = [fmnist_train[x][0] for x in range(40)]

show_images(examples)
def conv1x1(in_planes, out_planes, stride=1):

    """1x1 convolution"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



def conv3x3(in_planes, out_planes, stride=1):

    "Creates a 3x3 convolution w. padding (1,1)"

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 

                     padding=1, bias=False)



class ResnetBlock(nn.Module):

    expansion = 1

    

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, dropout=0.18):

        super(ResnetBlock, self).__init__()

        # Conv layer 1

        self.conv_1 = conv3x3(in_planes, out_planes, stride)

        self.batch_norm_1 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = dropout

        if self.dropout:

            self.dropout1 = nn.Dropout(p=dropout)

        # Conv layer 2

        self.conv_2 = conv3x3(out_planes, out_planes)

        self.batch_norm_2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample

        self.stride = stride

        

    def forward(self, x):

        residual = x

        

        out = self.conv_1(x)

        # If no dropout then use batchnorm

        if not self.dropout:

            out = self.batch_norm_1(out)

        out = self.relu(out)

        

        # Use dropout in between

        if self.dropout:

            out = self.dropout1(out)

        

        out = self.conv_2(out)

        out = self.batch_norm_2(out)

        

        if self.downsample is not None:

            residual = self.downsample(x)

        

        out += residual

        out = self.relu(out)

        return out

    

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, k=1):

        super(ResNet, self).__init__()

        self.inplanes = 64

        # Convert to 64 channels

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)  # out bx64x28x28

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # out bx64x14x14

        self.layer1 = self._make_layer(block, 64*k, layers[0])  # out bx(64xk)x14x14

        self.layer2 = self._make_layer(block, 128*k, layers[1], stride=2)  # out bx(128xk)x7x7

        #self.layer3 = self._make_layer(block, 256*k, layers[2], stride=2)

        #self.layer4 = self._make_layer(block, 512*k, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # out bx(128*k)x1x1

        self.fc = nn.Linear(128*k * block.expansion, num_classes)  # out bx10



        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                #nn.init.xavier_normal_(m.weight)

                

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                conv1x1(self.inplanes, planes * block.expansion, stride),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):

            layers.append(block(self.inplanes, planes))



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)



        x = self.layer1(x)

        x = self.layer2(x)

        #x = self.layer3(x)

        #x = self.layer4(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)



        return x

    

def init_weights(m):

    if type(m) == nn.Linear:

        nn.init.xavier_normal_(m.weight.data)

## Initialize model

model = ResNet(ResnetBlock, [2,2,2,2], k=2)



# Initialize linear layers

model.apply(init_weights)



# load pretrained state

model.to(device)

model.load_state_dict(torch.load("../input/fashion-mnist-model/test_acc-9560-epoch73.pth"))



# Loss fn

criterion = nn.CrossEntropyLoss()



# optimizer and scheduler

optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-4, patience=8, verbose=True)



# only save model state if test acc is above "best_test_acc"

best_test_acc = 9560



# Use dramatic increased LR every x'th epoch (0 = None)

lr_inc_interval = 40



# Batch size

batch_size = 256



train_loader = torch.utils.data.DataLoader(dataset=fmnist_train,

                                           batch_size=batch_size,

                                           pin_memory=True if use_cuda else False,

                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=fmnist_test,

                                          batch_size=batch_size,

                                          pin_memory=True if use_cuda else False,

                                          shuffle=False)



# Cache train and testset size

train_size = len(train_loader.dataset)

test_size = len(test_loader.dataset)
# Used 800 epochs in both training steps

for epoch in range(0):

    # statistics

    train_loss = 0

    train_correct = 0

    test_correct = 0

    test_loss = 0

    time_start = time.time()

    # --



    if lr_inc_interval and not epoch % lr_inc_interval and epoch is not 0:

        print("Increasing LR -- Big Steps")

        tmp_param_groups = optimizer.param_groups

        scheduler.num_bad_epochs -= 2

        for g in optimizer.param_groups:

            g['lr'] = 0.2

    else:

        tmp_param_groups = None



    model.train()

    for i, data in enumerate(train_loader):

        # inputs

        inputs, labels = data[0].to(device), data[1].to(device)



        # zero the gradient

        optimizer.zero_grad()



        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()



        optimizer.step()



        # stats

        train_loss += loss

        train_correct += outputs.argmax(-1).eq(labels).sum()



    # Release unesecary memory

    if use_cuda:

        torch.cuda.empty_cache()

        

    # Sometimes take a huge step

    if tmp_param_groups:

        optimizer.param_groups = tmp_param_groups



    model.eval()

    with torch.no_grad():

        for i, data in enumerate(test_loader):

            # inputs

            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)



            # test loss

            test_loss += criterion(outputs, labels)



            # accuracy

            test_correct += outputs.argmax(-1).eq(labels).sum()

            



    test_loss = test_loss/test_size

    scheduler.step(test_loss)

    

    if test_correct > best_test_acc and test_correct > 9480:

        best_test_acc = test_correct

        print("Saving model")

        torch.save(model.state_dict(), 'test_acc-{}-epoch{}.pth'.format(test_correct, epoch))

    

    print("#{} {:.2f} seconds, \n  train: {}, {}/{}, test: {}, {}/{}".format(

        epoch+1, time.time() - time_start,

        train_loss/train_size,

        train_correct, train_size,

        test_loss,

        test_correct, test_size))
total_wrong_labels = torch.LongTensor().cuda()

model.eval()

with torch.no_grad():

    for i, data in enumerate(test_loader):

        # inputs

        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)



        # correct

        test_correct = outputs.argmax(-1).eq(labels)

        wrong_labels = labels[test_correct == 0]

        total_wrong_labels = torch.cat((total_wrong_labels, wrong_labels))
t_np = total_wrong_labels.cpu().numpy()
lbl_counts = np.unique(t_np, return_counts=True)



lbls, counts = lbl_counts

for idx, lbl in enumerate(lbls):

    try:

        print("{}: {}".format(lbl_map[lbl], counts[idx]))

    except:

        print("{} is excluded.".format(lbl_map[lbl]))

        

df_wrong = pd.DataFrame(counts).rename(lbl_map, axis='index')

ax = df_wrong.plot.bar(figsize=(16,6), color="b", title="Model Wrong Classifications", fontsize=16, rot=0)



print("Total wrong classifications: {}/{}".format(counts.sum(), len(fmnist_test)))