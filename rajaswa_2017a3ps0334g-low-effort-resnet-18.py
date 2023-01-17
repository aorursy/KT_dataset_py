import pandas as pd
import numpy as np
import time
import os

import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
%matplotlib inline

seed_val = 1
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using ", device)

PATH = './'
train_df = pd.read_csv('/kaggle/input/nnfl-cnn-lab2/upload/train_set.csv')
test_df = pd.read_csv('/kaggle/input/nnfl-cnn-lab2/upload/sample_submission.csv')
test_df['label'] = [0]*len(test_df)

train_df, valid_df = train_test_split(train_df, random_state = seed_val, test_size = 0.1, stratify=train_df['label'])
train_df.reset_index(inplace=True); valid_df.reset_index(inplace=True)
train_df.head()
class ImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df['image_name'][idx])         
        image = Image.open(img_name)                               
        label = torch.tensor(self.df['label'][idx])                         
        
        if self.transform:            
            image = self.transform(image)                                          
        
        sample = (image, label)        
        return sample
train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(p=0.3),
                                transforms.RandomRotation(degrees=30),
                                transforms.RandomVerticalFlip(p=0.05),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
eval_transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
bs=32
train_path = '/kaggle/input/nnfl-cnn-lab2/upload/train_images/train_images/'
test_path = '/kaggle/input/nnfl-cnn-lab2/upload/test_images/test_images/'

train_dset = ImageDataset(train_df, train_path, train_transform)
val_dset = ImageDataset(valid_df, train_path, eval_transform)
test_dset = ImageDataset(test_df, test_path, eval_transform)

train_loader = DataLoader(train_dset, batch_size=bs,
                        shuffle=True, num_workers=0)
val_loader = DataLoader(val_dset, batch_size=bs,
                        shuffle=False, num_workers=0)
test_loader = DataLoader(test_dset, batch_size=bs,
                        shuffle=False, num_workers=0)
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Sequential(nn.Linear(512, 64),
                         nn.Linear(64, 6))
model = model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3, 
                             eps=10e-8,
                             weight_decay=0)

val_counts = train_df['label'].value_counts()
weights = [0] * len(val_counts)
for key, value in val_counts.items():
    weights[int(key)] = int(value)
weights_inv = [max(weights)/x for x in weights]
class_weights = torch.FloatTensor(weights_inv).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
epochs = 50
SAVE_PATH = PATH + 'resnet18.h5'

alpha=1; gamma=2
train_loss = []
validation_loss = []
precision_list, f1_list, recall_list, accuracy_list = [], [], [], []
max_f1, max_prec, max_recall, max_accuracy = -999, -999, -999, -999
min_val_loss = 9999
start_time = time.time()

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels.long())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss += loss.item()
        
        if i % 100 == 99:    # print every 25 mini-batches
                print('[%d, %5d] loss: %.3f time: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100, time.time()-start_time))
                running_loss = 0.0

    print("\nEPOCH ", epoch+1, " TRAIN LOSS = ", epoch_loss/len(train_dset))
    train_loss.append(epoch_loss/len(train_dset))

    val_loss = 0.0
    model.eval()
    preds = []
    ground_truth = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels.long())
            val_loss += loss.item()

            _, predictions = torch.max(logits, 1)
            for pred in predictions:
                preds.append(int(round(pred.item())))
            for label in labels:
                ground_truth.append(int(label.item()))

    print("EPOCH ", epoch+1, " VAL LOSS = ", val_loss/len(val_dset))
    validation_loss.append(val_loss/len(val_dset))
    model.train()

    preds = np.array(preds)
    ground_truth = np.array(ground_truth)
    prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, preds, average='weighted')
    accuracy = accuracy_score(ground_truth, preds)
    precision_list.append(prec)
    f1_list.append(f1)
    recall_list.append(recall)
    accuracy_list.append(accuracy)
    print("EPOCH ", epoch+1, "VAL PREC:", prec, "REC:", recall, "F1:", f1, "ACC:", accuracy, '\n')

    if max_f1 < f1:
        print("Model optimized, saving weights ...\n")
        torch.save(model, SAVE_PATH)
        min_val_loss = (val_loss/len(val_dset))
        max_f1, max_prec, max_recall, max_accuracy = f1, prec, recall, accuracy
#Plots
fig = plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(validation_loss, label='Validation Loss')

plt.legend()
plt.show()
fig = plt.figure()
plt.plot(precision_list, label='Precision')
plt.plot(recall_list, label='Recall')
plt.plot(f1_list, label='F1')
plt.plot(accuracy_list, label='Accuracy')

plt.legend()
plt.show()
model = torch.load(SAVE_PATH)
model.eval()
preds = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
            
        logits = model(images)

        _, predictions = torch.max(logits, 1)
        for pred in predictions:
            preds.append(int(round(pred.item())))
test_df['label'] = preds
test_df.head()
test_df.to_csv(PATH + 'sub_resnet18.csv', index=False)