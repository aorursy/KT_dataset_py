import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import pandas as pd

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset

from PIL import Image
n_epochs = 100 # number of epochs to train for
batch_size = 32 # batch size for training, validation and test data loaders
lr = 0.0001 # Learning rate for Adam
val_prop = 0.2 # proportion of training images to be used as validation
val_iter = 10 # number of epochs before testing validation set
torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization values
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, index, label, transform=None):
        """
        Args:
            index (list): Index of images to be used for the data
            label (list): List of annotations for the Covid CT scans (0 for non-Covid, 1 for Covid)
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.img_list = []
        # basically want to store images using ['image path', class]
        for i in range(len(index)):
            self.img_list.append((self.root_dir+'/'+str(index[i])+'.png',index[i], label[i]))
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def get_image(self, idx):
        img_path = self.img_list[idx][0]
        index = self.img_list[idx][1]
        label = self.img_list[idx][2]
        image = Image.open(img_path).convert('RGB')
        return image, index, label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        sample = {'img': image,
                  'index': int(self.img_list[idx][1]),
                  'label': int(self.img_list[idx][2])}
        return sample
def load_csv(csv_path):
    data = pd.read_csv(csv_path)
    return list(data['Index']), list(data['Label'])
# This code was adapted to the Kaggle online data
# if you put this python notebook in the same directory as your Train, Test, submission.csv, train.csv
# change root to ''
root = '../input/covidct/'
# root = ''

# write code to load the index and labels from the csv file
train_index, train_label = load_csv(root+'train.csv')
test_index, test_label = load_csv(root+'submission.csv')

# set proportion of training images to be the validation
split_index = int((1-val_prop)*len(train_index))

trainset = CovidCTDataset(root_dir=root+'Train/Train',
                          index = train_index[:split_index],
                          label = train_label[:split_index],
                          transform= train_transformer)
valset = CovidCTDataset(root_dir=root+'Train/Train',
                          index = train_index[split_index:],
                          label = train_label[split_index:],
                          transform= test_transformer)
testset = CovidCTDataset(root_dir=root+'Test/Test',
                          index = test_index,
                         label = test_label,
                          transform= test_transformer)
print('Training Dataset Length:', len(trainset))
print('Validation Dataset Length:', len(valset))
print('Test Dataset Length:', len(testset))

train_loader = DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, drop_last=False, shuffle=False)
test_loader = DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False)
labeldict = {
    0: '(Non-Covid)',
    1: '(Covid)'
}
image, index, label = trainset.get_image(1)
print('Index:', index, 'Label', label, labeldict[label])
display(image)
image, index, label = trainset.get_image(5)
print('Index:', index, 'Label', label, labeldict[label])
display(image)
def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        
        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        
        output = model(data)
        
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    
    print('Epoch: {} Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(epoch,
        train_loss/np.ceil(len(train_loader.dataset)/batch_size), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
def val():
    
    model.train()
    
    val_loss = 0
    val_correct = 0
    
    for batch_index, batch_samples in enumerate(val_loader):
        
        with torch.no_grad():
        
            # move data to device
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            criteria = nn.CrossEntropyLoss()
            loss = criteria(output, target.long())
            val_loss += loss
            
            val_correct += pred.eq(target.long().view_as(pred)).sum().item()
    
    print('\tVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss/len(val_loader.dataset), val_correct, len(val_loader.dataset),
        100.0 * val_correct / len(val_loader.dataset)))
def predict():
    
    model.eval()
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        
        indexlist = []
        labellist = []

        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, index = batch_samples['img'].to(device), batch_samples['index'].cpu().numpy()
            
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True).cpu().squeeze().numpy()
            labellist.extend(list(pred))
            indexlist.extend(list(index))
           
    return indexlist, labellist
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() # b, 3, 32, 32
        layer1 = torch.nn.Sequential() 
        layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))
 
        #b, 32, 32, 32
        layer1.add_module('relu1', torch.nn.ReLU(True)) 
        layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2)) # b, 32, 16, 16
        self.layer1 = layer1
        layer4 = torch.nn.Sequential()
        layer4.add_module('fc1', torch.nn.Linear(401408, 2))       
        self.layer4 = layer4
 
    def forward(self, x):
        conv1 = self.layer1(x)
        fc_input = conv1.view(conv1.size(0), -1)
        fc_out = self.layer4(fc_input)
 
model = SimpleCNN().to(device)
modelname = 'SimpleCNN'
## ResNet18
import torchvision.models as models
model = models.resnet18(pretrained=True).to(device)
modelname = 'ResNet18'
### ResNet50
import torchvision.models as models
model = models.resnet50(pretrained=True).to(device)
modelname = 'ResNet50'
### VGGNet
import torchvision.models as models
model = models.vgg16(pretrained=True)
model = model.to(device)
modelname = 'vgg16'
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# scheduler = StepLR(optimizer, step_size=10)

# Initial validation set loss (to verify model is training from scratch)
val()
for epoch in range(n_epochs):
    train(optimizer, epoch)
    if epoch%val_iter == 9:
        val()
index, label = predict()

df = pd.DataFrame({'Index': index, 'Label': label})

print(df)

df.to_csv('mysubmission.csv', index = False, columns = ['Index', 'Label'])
