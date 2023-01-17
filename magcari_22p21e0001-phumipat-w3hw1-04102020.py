import numpy as np
import pandas as pd
train = pd.read_csv('../input/superaiimageclassification/train/train/train.csv')
train
import torchvision.transforms as transforms

augm = transforms.Compose([transforms.RandomRotation(30), 
                           transforms.RandomResizedCrop(224), 
                           transforms.RandomHorizontalFlip(), 
                           transforms.ToTensor(), 
                           transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])])

trans = transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])])
import torch
from PIL import Image

train_img = []
label = []
for i in range(len(train)):
    img = Image.open('../input/superaiimageclassification/train/train/images/' + train['id'][i])
    train_img.append(augm(img))
    label.append(train['category'][i])
label = np.array(label)
train_img = torch.stack(train_img)
x_train = train_img
y_train = torch.from_numpy(label)
train_data = torch.utils.data.TensorDataset(x_train,y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
loader = {'train': train_loader}
def train_model(loader, model, opt, n_epo, crit, dest_path):
    for epoch in range(n_epo):
        train_loss = 0
        
        model.train()
        print('Training epoch ' + str(epoch) + ' started')
        for data, target in loader['train']:
            opt.zero_grad()
            
            res = model(data)
            loss = crit(res, target)
            loss.backward()
            opt.step()
            
            train_loss+=loss.item()*data.size(0)
        train_loss/=len(loader['train'].dataset)
        print('Epoch {}: Training Loss: {:.5f}'.format(epoch,train_loss))
        torch.save(model.state_dict(), dest_path)
    return model
import torch.nn as nn
import torch.nn.functional as func

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

        self.dropout = nn.Dropout(0.25)
    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.pool(func.relu(self.conv2(x)))
        
        x = func.relu(self.conv3(x))
        x = self.pool(func.relu(self.conv4(x)))
        
        x = func.relu(self.conv5(x))
        x = func.relu(self.conv6(x))
        x = self.pool(func.relu(self.conv6(x)))
        
        x = func.relu(self.conv7(x))
        x = func.relu(self.conv8(x))
        x = self.pool(func.relu(self.conv8(x)))
        
        x = func.relu(self.conv8(x))
        x = func.relu(self.conv8(x))
        x = self.pool(func.relu(self.conv8(x)))
        
        x = x.view(-1, 25088)

        x = self.dropout(x)
        x = func.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = func.relu(self.fc2(x))
        
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
model = Net()
import torch.optim as optim

crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.0001)
model = train_model(loader, model, opt, 50, crit,'model_train.pt')
model.load_state_dict(torch.load('model_train.pt'))
def evaluate(loader, model, crit):
    model.eval()
    for batch_idx, (data, target) in enumerate(loader['train']):
        res = model(data.float())
        loss = criterion(res, target)
        pred = output.data.max(1, keepdim=True)[1]
evaluate(loader, model, crit)
import os

test = pd.read_csv('../input/superaiimageclassification/val/val/val.csv')
test_img = []
test_id = []
test_label = []
for i in os.listdir('../input/superaiimageclassification/val/val/images/'):
    img = Image.open('../input/superaiimageclassification/val/val/images/' + i)
    test_img.append(trans(img))
    test_id.append(i)
test_img = torch.stack(test_img)
test_data = torch.utils.data.TensorDataset(test_img,test_img)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

model.load_state_dict(torch.load('model_train.pt'))
model.eval()

for data ,_ in test_loader:
    output = model(data)
    test_label.append(list(np.squeeze(output.data.max(1, keepdim=True)[1]).cpu().numpy()))
res = []
for i in test_label:
    for j in i:
        res.append(j)
answer = pd.DataFrame(test_id,columns =['id'])
answer['category'] = res
answer.to_csv('superAI.csv',index=False)