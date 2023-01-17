import glob 
import zipfile
import numpy as np
import cv2
import os
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from sklearn.model_selection import train_test_split


data_root_path = ''
test_path = data_root_path + 'data/test1'
train_path = data_root_path + 'data/train'

def unzip_files():
    zip_files = glob.glob(data_root_path + '/kaggle/input/*/*.zip')
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(data_root_path + "data")


IMAGE_SIZE = (50, 50)
train_data = []

def make_training_data():
    unzip_files()
    LABELS = {'cat': 0, 'dog': 1}
    for f in os.listdir(train_path):
        path = os.path.join(train_path, f)
        if("jpg" not in f):
            continue
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, IMAGE_SIZE)
        train_data.append([img, np.eye(len(LABELS))[LABELS[f.split('.')[0]]]])
    
    np.random.shuffle(train_data)
    np.save(data_root_path + "train_data.npy", train_data)

if(os.path.exists(data_root_path + "train_data.npy")):
    train_data = np.load(data_root_path + "train_data.npy", allow_pickle=True)
else:
    make_training_data()


np.random.shuffle(train_data)
print(train_data[0])

class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        self.fc1 = torch.nn.Linear(2*2*128, 32)
        self.fc2 = torch.nn.Linear(32, 2) 
        
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        x = F.relu(self.fc1(x.view(-1, 128*2*2)))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
net = Net().to(device)

all_X = torch.Tensor([i[0] for i in train_data]).view(-1,50,50)
all_X = all_X/255.0
all_y = torch.Tensor([i[1] for i in train_data])

X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.1, stratify=all_y)

def run(X, y, is_training=True):
    
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    
    if is_training:
        net.zero_grad()
        optimizer.zero_grad()
        
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if is_training:
        loss.backward()
        optimizer.step()

    return acc, loss

def batch_and_run(X, y, is_training=True):
    BATCH_SIZE = 64
    for i in tqdm(range(0, len(X), BATCH_SIZE)): 
        batch_X = X[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
        batch_y = y[i:i+BATCH_SIZE].to(device)
            
        return run(batch_X, batch_y, is_training)
        
def train(net):
    EPOCHS = 20  
    for epoch in range(EPOCHS):
        acc, loss = batch_and_run(X_train, y_train, True)
        val_acc, val_loss = batch_and_run(X_test, y_test, False)
        print(f"Epoch: {epoch}, {round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
        
train(net)


predictions=[]
id_line=[]
def test(net):
    for f in os.listdir(test_path):
        id_line.append(f.split('.')[0])
        path = os.path.join(test_path, f)
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (50, 50))
        test_X = torch.Tensor(img).view(-1, 1, 50, 50)
        test_X = test_X.to(device)
        net_out = net(test_X)
        
        predictions.append(torch.argmax(net_out))

test(net)

predicted_val = [x.tolist() for x in predictions]
submission_df = pd.DataFrame({'id':id_line, 'label':predicted_val})
submission_df.to_csv("submiss.csv", index=False)