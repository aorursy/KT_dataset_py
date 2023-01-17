# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import os



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from torchvision.transforms import ToTensor

from torchvision.models import resnet as model_res



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



from tqdm import tqdm

        

print(os.listdir('/kaggle/input/Kannada-MNIST'))



# Any results you write to the current directory are saved as output.
dir_csv = '/kaggle/input/Kannada-MNIST'

dir_train_img = dir_csv +'/train.csv'

dir_test_img = dir_csv + '/test.csv'



train = pd.read_csv(dir_train_img)

test = pd.read_csv(dir_test_img)
n_classes = 10

n_epochs = 2

BATCH_SIZE = 256

IMG_DIM = 28



# Gets the GPU if there is one, otherwise the cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
df_train = pd.read_csv(dir_train_img)

df_test = pd.read_csv(dir_test_img)



target = df_train['label']

df_train.drop('label', axis=1, inplace=True)



X_test = pd.read_csv('../input/Kannada-MNIST/test.csv')

X_test.drop('id', axis=1, inplace=True)
X_train, X_dev, y_train, y_dev = train_test_split(df_train, target, stratify=target, random_state=42, test_size=0.01)

print('X_train', len(X_train))

print('X_dev', len(X_dev))

print('X_test', len(X_test))
fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))



# I know these for loops look weird, but this way num_i is only computed once for each class

for i in range(10): # Column by column

    num_i = X_train[y_train == i]

    ax[0][i].set_title(i)

    for j in range(10): # Row by row

        ax[j][i].axis('off')

        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')
# Infer from standard PyTorch torch.utils.data.Dataset class

class CharData(Dataset):

    def __init__(self, images, labels, transform, classes):

        self.X = images

        #for i in range(0, len(self)):

        #    self.X[i] = np.array(self.X[i]).reshape((IMG_DIM,IMG_DIM,1))

        self.y = labels

        self.tranform = transform

        self.classes = classes

        

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, idx=None):

        img = np.array(self.X.iloc[idx,:], dtype='uint8').reshape((IMG_DIM,IMG_DIM,1))

        if self.y is not None:

            y = np.zeros(self.classes, dtype='float32')

            y[self.y.iloc[idx]] = 1

            return img, y

        else:

            return img
class ToTensor_customized(ToTensor):

    """Convert ndarrays in sample to Tensors."""



    def __call__(self, sample):

        image = sample['X']

        #image = image.permute((0,1, 2))

        if sample['y'] is not None:

            label = sample['y']

        else:

            return {'X': torch.from_numpy(image)}

        # swap color axis because

        # numpy image: H x W x C

        # torch image: C X H X W

        

        return {'X': torch.from_numpy(image),

                'y': torch.from_numpy(label)}
# Put some augmentation on training data using pytorch torchvision.transforms



train_transform = transforms.Compose([

    transforms.Resize([IMG_DIM,IMG_DIM]),

    #transforms.ToPILImage(),

    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),

    transforms.ToTensor()

])



# test data is not augmented, kept as it is.

test_transform = transforms.Compose([

    transforms.Resize([IMG_DIM,IMG_DIM]),

    #transforms.ToPILImage(),

    transforms.ToTensor()

])
# Create Dataset objects



train_dataset = CharData(images=X_train, labels=y_train, transform=train_transform, classes=10)

dev_dataset = CharData(images=X_dev, labels=y_dev, transform=test_transform, classes=10)

test_dataset = CharData(images=X_test, labels=None, transform=test_transform, classes=10)



#print(len(train_dataset))

#print(np.shape(train_dataset[0][0]))



#train_dataset.unsqueeze_(1)

#train_dataset.y.unsqueeze_(0)

#dev_dataset.unsqueeze_(1)

#test_dataset.unsqueeze_(1)
# Defining the data generators for producing batches of data



train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



print(np.shape(train_dataset.X))
# Define the network by inheriting nn.Module



DEPTH_MULT = 2



class ConvLayer(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3):

        super(ConvLayer, self).__init__()

        self.ops = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=kernel_size//2),

            nn.BatchNorm2d(out_channel),

            nn.ReLU(inplace=True)

        )    

    def forward(self, x):

        return self.ops(x)

    

class FCLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super(FCLayer, self).__init__()

        self.ops = nn.Sequential(

            nn.Linear(in_features, out_features),

            nn.BatchNorm1d(out_features),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        #print(x.shape)

        return self.ops(x)



class Flatten(nn.Module):

    def __init__(self):

        super(Flatten, self).__init__()

    def forward(self, x):

        #print(x.shape)

        shape = torch.prod(torch.tensor(x.shape[1:])).item()

        return x.view(-1, shape)

        #return x.view(shape, -1)        

    

class Net(nn.Module):

    def __init__(self, num_classes):

        super(Net, self).__init__()

        self.conv1 = ConvLayer(1, DEPTH_MULT * 32)

        self.conv2 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)

        self.conv3 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)

        self.conv3 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)

        self.conv4 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 32)

        

        self.conv5 = ConvLayer(DEPTH_MULT * 32, DEPTH_MULT * 64)

        self.conv6 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)

        self.conv7 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)

        self.conv8 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)

        self.conv9 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)

        self.conv10 = ConvLayer(DEPTH_MULT * 64, DEPTH_MULT * 64)

        

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        

        self.flat = Flatten()

        

        #self.fc1 = FCLayer(DEPTH_MULT * 64 * 7 * 7, DEPTH_MULT * 512)

        self.fc1 = FCLayer(25088, DEPTH_MULT * 512)

        self.fc2 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)

        self.fc3 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)

        self.fc4 = FCLayer(DEPTH_MULT * 512, DEPTH_MULT * 512)

        self.projection = nn.Linear(DEPTH_MULT * 512, num_classes)

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = self.conv5(x)

        x = self.conv6(x)

        x = self.conv7(x)

        x = self.conv8(x)

        x = self.conv9(x)

        x = self.conv10(x)

        

        x = self.mp(x)

        

        x = self.flat(x)

        

        x = self.fc1(x)

        x = self.fc2(x)

        x = self.fc3(x)

        x = self.fc4(x)

        x = self.projection(x)

        

        return x
model = Net(10)

model = model.to(device)



n_epochs = 6

optimizer = optim.Adam(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs // 4, gamma=0.1)
def criterion(input, target, size_average=True):

    """Categorical cross-entropy with logits input and one-hot target"""

    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)

    if size_average:

        l = l.mean()

    else:

        l = l.sum()

    return l
def train(epochs, history=None):

    model.train()

    

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)

        target = target.to(device)

        

        optimizer.zero_grad()

        #print(np.shape(data))

        data = data.permute(0, 3, 1, 2).float()

        output = model(data)

        loss = criterion(output, target)

        

        if history is not None:

            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        

        loss.backward()

        optimizer.step()

        

        if (batch_idx + 1) % 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(

                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),

                100. * (batch_idx + 1) / len(train_loader),

                optimizer.state_dict()['param_groups'][0]['lr'],

                loss.data))

                

    exp_lr_scheduler.step()
def evaluate(epoch, history=None):

    model.eval()

    loss = 0

    correct = 0

    

    with torch.no_grad():

        for data, target in dev_loader:

            data = data.to(device)

            target = target.to(device)



            data = data.permute(0, 3, 1, 2).float()

            output = model(data)



            loss += criterion(output, target, size_average=False).data



            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()

    

    loss /= len(dev_loader.dataset)

    accuracy = correct / len(dev_loader.dataset)

    

    if history is not None:

        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

        history.loc[epoch, 'dev_accuracy'] = accuracy

    

    print('Dev loss: {:.4f}, Dev accuracy: {}/{} ({:.3f}%)\n'.format(

        loss, correct, len(dev_loader.dataset),

        100. * accuracy))     

import gc



history = pd.DataFrame()



for epoch in range(n_epochs):

    print('Epoch: '+ str(epoch))

    torch.cuda.empty_cache()

    gc.collect()

    train(epoch, history)

    evaluate(epoch, history)
history['train_loss'].plot();
history.dropna()['dev_loss'].plot();
history.dropna()['dev_accuracy'].plot();
model.eval()

predictions = []



for data in tqdm(test_loader):

    data = data.to(device)

    data = data.permute(0, 3, 1, 2).float()

    output = model(data).max(dim=1)[1] # argmax

    predictions += list(output.data.cpu().numpy())
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv('submission.csv', index=False)

submission.head()