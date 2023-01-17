import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cv2
csv = pd.read_csv('../input/flower_images/flower_images/flower_labels.csv')
len(csv.label.unique())
csv[:5]
csv.loc[0]['file']
DATA_DIR = '../input/flower_images/flower_images/'
X = []
for i in range(len(csv)):
    file = csv.loc[i].file
    img = cv2.imread(DATA_DIR + file)
    if img.shape != (128, 128, 3):
        img = cv2.resize(img, (128,  128))
    X.append(img.astype(np.float32))
print(np.array(X).shape)

# X = np.array(X)
y = csv['label'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(np.array(X_train).shape, np.array(y_train).shape, np.array(X_test).shape, np.array(y_test).shape)
import torch
import torch.nn  as nn
import torch.nn.functional  as F
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ImgLoader(Dataset):
    def __init__(self, x, y,iscuda=False):
        self.X = np.array(x)
        self.y = np.array(y)
#         self.cuda = iscuda
    
    def __getitem__(self, index):
        x_val = self.X[index]
        x_val = torch.from_numpy(x_val).permute(2, 1, 0)
        y_val = torch.from_numpy(np.array([self.y[index]]))
#         if self.cuda:
#             x_val = x_val.cuda()
#             y_val = y_val.cuda()
        return x_val, y_val

    def __len__(self):
        return len(self.X)
class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=5, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*63*63, 100)
        self.fc2 = nn.Linear(100, 10)

        
    def forward(self, x):
        self.conv_l1 = self.conv1(x)
        self.conv_l1 = self.bn1(self.conv_l1)
        
        self.conv_l2 = F.relu(self.conv2(self.conv_l1))
        self.conv_l2 = self.bn2(self.conv_l2)
        
        self.conv_l3 = F.relu(self.conv3(self.conv_l2))
        self.conv_l3 = self.bn3(self.conv_l3)

        self.conv_l4 = F.relu(self.conv4(self.conv_l3))
        self.conv_l4 = self.bn4(self.conv_l4)

        self.conv_l5 = F.relu(self.conv5(self.conv_l4))
        self.conv_l5 = self.bn5(self.conv_l5)

        self.fc_l1 = self.conv_l5.view(-1, 64 * 63 * 63)
        self.fc_l1 = F.relu(self.fc1(self.fc_l1))
        self.fc_l2 = self.fc2(self.fc_l1) 
        
        return F.log_softmax(self.fc_l2)
    
            
use_cuda = torch.cuda.is_available()
img_loader = ImgLoader(X_train, y_train, use_cuda)
trainloader = DataLoader(img_loader, batch_size=5, shuffle=True, num_workers=4)
def eval(model):
    model.train(False)
    count = 0
    for x,y in zip(X_test,y_test):
        x = torch.from_numpy(np.array([x])).permute(0, 3, 1, 2)
        if use_cuda:
            x = x.cuda()

        out = model(Variable(x))
#         print(out.shape)
        label = np.argmax(out.data.cpu().numpy()) # needs to be optimized
        if y == label:
            count += 1

    print(count * 1.0 / len(X_test))
epochs = 4
criterion = nn.CrossEntropyLoss()
net = ConvClassifier()
if use_cuda:
    net = net.cuda()

optimizer = optim.SGD(params=net.parameters(), lr=0.003)

for epoch in range(epochs):
    losses = []
    net.train(True)
    for i , (x, y) in enumerate(trainloader):

        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()

        inputs =  Variable(x)
        output = net(inputs)
        targets = Variable(y.squeeze(1))

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])
#         break
#     break
    if (epoch+1) % 1 == 0:
        print('Epoch %d Loss %.4f  ' % (epoch+1, np.average(losses)))
        print('test accuracy')
        eval(net)
