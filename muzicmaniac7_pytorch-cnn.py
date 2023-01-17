import numpy as np
import pandas as pd
import re
import torch
import torchvision
import torch.nn as nn
from glob import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_paths = glob("/kaggle/input/multiclass-weather-dataset/Multi-class Weather Dataset/*/*")
labels = list()
for path in file_paths:
    file_name = path[path.rfind("/")+1:]
    labels.append(file_name[:re.search(r"\d", file_name).start()])
df = pd.DataFrame({'path': file_paths, 'class': labels})
df.head()
def label_column(df):
    classes = list(df['class'].unique())
    class_to_num = dict(zip(classes, range(len(classes))))
    df['label'] = df['class'].apply(lambda x: class_to_num[x])
label_column(df)
df.head()
train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
test, val = train_test_split(test, test_size=0.5, shuffle=True, random_state=42)
len(val)
transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop((100, 100)),
                transforms.RandomCrop((80, 80)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize((156, 156)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.X = df['path']
        self.y = df['label']
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, index):
        label = self.y[index]
        img = Image.open(self.X[index]).convert('RGB')
        img = img.resize((156, 156))
        img = np.array(img)
        if self.transform: # augmentation
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return {'X': img.to(device), 'y': label} # output should always be tensors
trainD = CustomDataset(train.reset_index(drop=True), transform=None)
testD = CustomDataset(test.reset_index(drop=True))
valD = CustomDataset(val.reset_index(drop=True))
trainDL = DataLoader(trainD, batch_size=16, shuffle=True)
testDL = DataLoader(testD, batch_size=16)
valDL = DataLoader(valD, batch_size=16)
for data in valDL:
    print(data['X'].shape)
    print(data['y'])
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.conv3 = nn.Conv2d(10, 14, 5)
        self.conv4 = nn.Conv2d(14, 18, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(6*6*18, 120)
        self.linear2 = nn.Linear(120, 30)
        self.output = nn.Linear(30, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x))) # input 156x156, output: 76x76
        x = self.maxpool(self.relu(self.conv2(x))) # input 76x76, output: 36x36
        x = self.maxpool(self.relu(self.conv3(x))) # input 36x36, output: 16x16
        x = self.maxpool(self.relu(self.conv4(x))) # input 16x16, output: 6x6
        x = x.view(-1, 6*6*18) # flatten
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.output(x)
        return x
model = CNN()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
def trainer(epochs, trainDL, valDL, model, loss_function, optimizer):
    for epoch in range(epochs):
        for i, data in enumerate(trainDL):
            model.train()
            output = model(data['X'])
            t_loss = loss_function(output, data['y'])
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                v_loss = 0
                model.eval()
                for j, data in enumerate(valDL):
                    loss = loss_function(model(data['X']), data['y'])
                    v_loss += loss.item()
            print(f"Epoch: {epoch+1}, Batch: {i+1}, Training loss: {str(round(t_loss.item(), 2))}, Validation loss: {str(round(v_loss/j, 2))}")
trainer(4, trainDL, valDL, model, loss_function, optimizer)
def tester(testDL, model):
    model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(testDL):
        output = model(data['X'])
        values, indices = torch.max(output.data, 1)
        total += data['y'].size(0)
        correct += (data['y'] == indices).sum().item()
    print(f"Accuracy: {str(round(correct/total, 2))}")
tester(testDL, model)
