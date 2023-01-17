!pip install torchsummary
import os
import time
import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import cv2
from skimage import io, color
from PIL import Image
from sklearn.preprocessing import LabelEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
data = pd.read_csv("../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv")
data.head()
for i in data.index:
    data["path"].iloc[i] = data["path"].iloc[i].replace("\\", "/")
le = LabelEncoder()
le.fit(data["label"])
data["label"] = le.transform(data["label"])
data.info()
data.is_validation.value_counts()
data.is_final_validation.value_counts()
def split_data(dt):
    idx = list()
    a = pd.DataFrame()
    b = pd.DataFrame()
    for i in data.index:
        if dt["is_validation"].iloc[i] == 1:
            a = a.append(dt.iloc[i])
            idx.append(i)
        if dt["is_final_validation"].iloc[i] == 1:    
            b = b.append(dt.iloc[i])
            idx.append(i)

    dt = dt.drop(dt.index[idx])
    dt = dt.reset_index()
    a = a.reset_index()
    b = b.reset_index()
    return dt, a, b 

train_df, val_df, test_df = split_data(data)
# sanity check
print("Length of train dataset: ", len(train_df))
print("Length of validation dataset: " ,len(val_df))
print("Length of test dataset: ", len(test_df))
val_df.label = val_df.label.astype(np.int64)
test_df.label = test_df.label.astype(np.int64)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class BeeDataset(Dataset):
    def __init__(self, df:pd.DataFrame, imgdir:str, train:bool,
                 transforms=None):
        self.df = df
        self.imgdir = imgdir
        self.train = train
        self.transforms = transforms
    
    def __getitem__(self, index):
        im_path = os.path.join(self.imgdir, self.df.iloc[index]["path"])
        x = cv2.imread(im_path)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (224, 224))

        if self.transforms:
            x = self.transforms(x)
        
        if self.train:
            y = self.df.iloc[index]["label"]
            return x, y
        else:
            return x
    
    def __len__(self):
        return len(self.df)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 4)
    
    def forward(self, x):
        output = self.model(x)
        return output

train_data = BeeDataset(df=train_df,
                        imgdir="../input/bee-vs-wasp/kaggle_bee_vs_wasp",
                        train=True,
                        transforms=train_transform)

val_data = BeeDataset(df=val_df,
                      imgdir="../input/bee-vs-wasp/kaggle_bee_vs_wasp",
                      train=True,
                      transforms=test_transform)

test_data = BeeDataset(df=test_df,
                       imgdir="../input/bee-vs-wasp/kaggle_bee_vs_wasp",
                       train=True,
                       transforms=test_transform)
criterion = nn.CrossEntropyLoss()
arch = Net()
arch.to(device)
optim = torch.optim.SGD(arch.parameters(), lr=1e-3, momentum=0.9)


train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=32, num_workers=4)
val_loader = DataLoader(dataset=val_data, shuffle=True, batch_size=32, num_workers=4)
test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=32, num_workers=4)
summary(batch_size=32, input_size=(3, 224, 224), model=arch)
def train_model(model, optimizer, n_epochs, criterion):
    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        epoch_time = time.time()
        epoch_loss = 0
        correct = 0
        total=0
        print("Epoch {} / {}".format(epoch, n_epochs))
        model.train()
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # zeroed grads
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # softmax + cross entropy
            loss.backward() # back pass
            optimizer.step() # updated params
            epoch_loss += loss.item() # train loss
            _, pred = torch.max(outputs, dim=1)
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.shape[0]
        acc = correct / total
        
        model.eval()
        a=0
        pred_val=0
        correct_val=0
        total_val=0
        with torch.no_grad():
            for inp_val, lab_val in val_loader:
                inp_val = inp_val.to(device)
                lab_val = lab_val.to(device)
                out_val = model(inp_val)
                loss_val = criterion(out_val, lab_val)
                a += loss_val.item()
                _, pred_val = torch.max(out_val, dim=1)
                correct_val += (pred_val.cpu()==lab_val.cpu()).sum().item()
                total_val += lab_val.shape[0]
            acc_val = correct_val / total_val
        epoch_time2 = time.time()
        print("Duration: {:.0f}s, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}"
              .format(epoch_time2-epoch_time, epoch_loss/len(labels), acc, a/len(lab_val), acc_val))
    end_time = time.time()
    print("Total Time:{:.0f}s".format(end_time-start_time))
def eval_model(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.shape[0]
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
train_model(model=arch, optimizer=optim, n_epochs=15, criterion=criterion)
eval_model(arch)