# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# импорт библиотек
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torchvision
from PIL import Image
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import time
# определяем пути 
TRAIN_IMG_PATH = "../input/dog-breed-identification/train"
TEST_IMG_PATH = "../input/dog-breed-identification/test"
LABELS_CSV_PATH = "../input/dog-breed-identification/labels.csv"
SAMPLE_SUB_PATH = "../input/dog-breed-identification/sample_submission.csv"
class DogsDataset(Dataset):
    def __init__(self, img_dir, dataframe, transform=None):
        self.labels_frame = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.id[idx]) + ".jpg"
        image = Image.open(img_name)
        label = self.labels_frame.target[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label] 
dframe = pd.read_csv(LABELS_CSV_PATH)
labelnames = pd.read_csv(SAMPLE_SUB_PATH).keys()[1:]
codes = range(len(labelnames))
breed_to_code = dict(zip(labelnames, codes))
code_to_breed = dict(zip(codes, labelnames))
dframe['target'] =  [breed_to_code[x] for x in dframe.breed]

cut = int(len(dframe)*0.8)
train, test = np.split(dframe, [cut], axis=0)
test = test.reset_index(drop=True)

train_ds = DogsDataset(TRAIN_IMG_PATH, train)
test_ds = DogsDataset(TRAIN_IMG_PATH, test)
idx = 29
plt.imshow(train_ds[idx][0])
print(code_to_breed[train_ds[idx][1]])
print("Shape of the image is: ", train_ds[idx][0].size)
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
train_ds = DogsDataset(TRAIN_IMG_PATH, train, data_transform)
test_ds = DogsDataset(TRAIN_IMG_PATH, test, data_transform)
datasets = {"train": train_ds, "val": test_ds}

idx = 29
print(code_to_breed[train_ds[idx][1]])
print("Shape of the image is: ", train_ds[idx][0].shape)
trainloader = DataLoader(train_ds, batch_size=14,
                        shuffle=True, num_workers=4)

testloader = DataLoader(test_ds, batch_size=14,
                        shuffle=True, num_workers=4)

dataloaders = {"train": trainloader, "val": testloader}
# определяем device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
model = torchvision.models.resnet18(pretrained=True, progress=True)

# настраиваем модель под свою задачу
in_features = model.fc.in_features
model.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
model.fc = nn.Linear(in_features, 120)
# функция тренировки
def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, sheduler, n_epochs):
    # переносим на GPU
    model_conv.to(device)
    
    valid_loss_min = np.Inf

    # количество эпох
    for epoch in range(n_epochs):
        train_loss = []
        for batch_i, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_conv(data)
            loss = criterion(output, target)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        # запускаем валидацию
        model_conv.eval()
        val_loss = []
        for batch_i, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)
            output = model_conv(data)
            loss = criterion(output, target)
            val_loss.append(loss.item())
        
        print(f'Epoch {epoch}, train loss: {np.mean(train_loss):.4f}, valid loss: {np.mean(val_loss):.4f}.')
        valid_loss = np.mean(val_loss)
        scheduler.step(valid_loss)
        
      
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model_conv.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
            

    return model_conv, train_loss, val_loss
# определяем лосс, оптимайзер, шедуллер
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3,)
# запускаем обучение
model_resnet, train_loss, val_loss = train_model(model, trainloader, testloader, criterion, optimizer, scheduler, n_epochs=25,)
model.load_state_dict(torch.load("model.pt"))
model.to(device)
import torch.nn.functional as F
# тестируем модель
model.to(device)
model.eval()
pred_list = []
labels_list = []
for images,labels in testloader:
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    pred = F.softmax(output)
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    pred_list += [p.item() for p in pred]
    labels_list += [name for name in labels]
submission_df = pd.read_csv(SAMPLE_SUB_PATH)
output_df = pd.DataFrame(index=submission_df.index, columns=submission_df.keys() )
output_df['id'] = submission_df['id']
submission_df['target'] =  [0] * len(submission_df)
tdata_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

test_ds = DogsDataset(TEST_IMG_PATH, submission_df,tdata_transform)
testloader = DataLoader(test_ds, batch_size=14,
                        shuffle=True, num_workers=4)
def test_sumission(model):
    since = time.time()
    sub_outputs = []
    model.train(False)  # Set model to evaluate mode
    # Iterate over data.
    for data in testloader:
        # get the inputs
        inputs, labels = data

        inputs = Variable(inputs.type(torch.cuda.FloatTensor))
        labels = Variable(labels.type(torch.cuda.LongTensor))

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        sub_outputs.append(outputs.data.cpu().numpy())

    sub_outputs = np.concatenate(sub_outputs)
    for idx,row in enumerate(sub_outputs.astype(float)):
        sub_outputs[idx] = np.exp(row)/np.sum(np.exp(row))

    output_df.loc[:,1:] = sub_outputs
        
    print()
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return output_df
odf = test_sumission(model)
odf.to_csv("dogs_id.csv", index=False)