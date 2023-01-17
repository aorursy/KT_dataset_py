import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import time
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations 
from albumentations.pytorch import ToTensorV2 as AT
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(40)
PATH = '/kaggle/input/korpus-clear/dataset_new_clear/'
train_path = PATH + "dataset_new_clear/train/"
test_path = PATH + "dataset_new_clear/test/test/test/"
sample_submission = pd.read_csv(PATH + "dataset_new_clear/sample_submission.csv")
train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
print(len(train_list), len(test_list))
print(len(sample_submission))
class ChartsDataset(Dataset):
    
    def __init__(self, path, img_list, transform=None, mode='train'):
        self.path = path
        self.img_list = img_list
        self.transform=transform
        self.mode = mode
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        
        if image_name.split(".")[1] == "gif":
           gif = cv2.VideoCapture(self.path + image_name)
           _, image = gif.read()
        else:
            image = cv2.imread(self.path + image_name)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = 0
        if "bar_chart" in image_name:
            label = 1
        elif "diagram" in image_name:
            label = 2
        elif "flow_chart" in image_name:
            label = 3
        elif "graph" in image_name:
            label = 4
        elif "growth_chart" in image_name:
            label = 5
        elif "pie_chart" in image_name:
            label = 6
        elif "table" in image_name:
            label = 7
        else:
            label = 0
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        if self.mode == "train":
            return image, label
        else:
            return image, image_name
batch_size = 32
num_workers = 0
img_size = 256
data_transforms = albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.Normalize(),
    AT()
    ])


data_transforms_test = albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.Normalize(),
    AT()
    ])
#Инициализируем датасеты
trainset = ChartsDataset(train_path, train_list,  transform = data_transforms)
testset = ChartsDataset(test_path, test_list,  transform=data_transforms_test, mode="test")
#Разделим трейновую часть на трейн и валидацию. Попробуем другой способ.
valid_size = int(len(train_list) * 0.1)
train_set, valid_set = torch.utils.data.random_split(trainset, 
                                    (len(train_list)-valid_size, valid_size))
#создаем даталоадеры для всех 3х подвыборок.
trainloader = torch.utils.data.DataLoader(train_set, pin_memory=True, 
                                        batch_size=batch_size, shuffle=True)

validloader = torch.utils.data.DataLoader(valid_set, pin_memory=True, 
                                        batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                         num_workers = num_workers)
import matplotlib.pyplot as plt
samples, labels = next(iter(trainloader))
plt.figure(figsize=(16,24))
grid_imgs = torchvision.utils.make_grid(samples[:32])
np_grid_imgs = grid_imgs.numpy()
print(labels)
plt.imshow(np.transpose(np_grid_imgs, (1,2,0)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
def train_model(model_conv, train_loader, valid_loader, criterion, optimizer, sheduler, n_epochs):
    model_conv.to(device)
    valid_loss_min = np.Inf
    patience = 5
    # сколько эпох ждем до отключения
    p = 0
    # иначе останавливаем обучение
    stop = False

    # количество эпох
    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = []

        for batch_i, (data, target) in enumerate(tqdm(train_loader)):
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
        scheduler.step(valid_loss) # for scheduler_on_platou
        #scheduler.step() # for cyclik_LR
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model_conv.state_dict(), 'model.pt')
            valid_loss_min = valid_loss
            p = 0

        # проверяем как дела на валидации
        if valid_loss > valid_loss_min:
            p += 1
            print(f'{p} epochs of increasing val loss')
            if p > patience:
                print('Stopping training')
                stop = True
                break        

        if stop:
            break
    return model_conv, train_loss, val_loss
model = torchvision.models.wide_resnet50_2(pretrained=True)
layers_to_unfreeze = 5
ct = 0
for child in model.children():
    ct += 1
    if ct < layers_to_unfreeze:
        for param in child.parameters():
            param.requires_grad = False
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 8, bias=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5,)
model_wide_resnet50, train_loss, val_loss = train_model(model, trainloader, validloader, criterion, 
                              optimizer, scheduler, n_epochs=40)
torch.save(model_wide_resnet50.state_dict(), 'model_wide_resnet50_dict.pt')
torch.save(model_wide_resnet50, 'model_wide_resnet50.pt')
modelA = torch.load('../input/korpus-clear/model_densenet161.pt')
modelB = model_wide_resnet50
modelC = torch.load('../input/korpus-clear/model_resnext101.pt')
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, nb_classes=8):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA.to(device)
        self.modelB = modelB.to(device)
        self.modelC = modelC.to(device)
        # Remove last linear layer
        self.modelA.classifier = nn.Identity().to(device)
        self.modelB.fc = nn.Identity().to(device)
        self.modelC.fc = nn.Identity().to(device)
        
        # Create new classifier
        self.classifier = nn.Linear(2208+2048+2048, nb_classes).to(device)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.modelC(x)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)
        
        x = self.classifier(F.relu(x))
        return x
model = MyEnsemble(modelA, modelB, modelC)
x = torch.randn(1, 3, img_size, img_size).to(device)
output = model(x)
output
for param in model.parameters():
    param.requires_grad = False
    
in_ = model.classifier.in_features
model.classifier = nn.Linear(in_features=in_, out_features=8, bias=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=5,)
model_ensemble, train_loss, val_loss = train_model(model, trainloader, validloader, criterion, 
                              optimizer, scheduler, n_epochs=40)
torch.save(model_ensemble.state_dict(), 'model_ensemble.pt')
torch.save(model_ensemble, 'model_ensemble1.pt')
model.to(device)
model.eval()
pred_list = []
names_list = []
for images, image_names in testloader:
    with torch.no_grad():
        images = images.to(device)
        output = model(images)
        pred = F.softmax(output)
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        pred_list += [p.item() for p in pred]
        names_list += [name for name in image_names]


sample_submission.image_name = names_list
sample_submission.label = pred_list
sample_submission.to_csv('submission26.csv', index=False)