import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from albumentations import Compose, Resize, OneOf, RandomBrightness, RandomContrast, ShiftScaleRotate, Normalize 
from albumentations.pytorch import ToTensor
import pandas as pd
import numpy as np
import cv2
import time
import copy
import matplotlib.pyplot as plt
seed = 271
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class digitdataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        super(digitdataset, self).__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __getitem__(self, idx):
        image = self.df.iloc[idx][1:].to_numpy().reshape(28,28)
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2RGB)
        label = self.df.iloc[idx][0]
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = ToTensor()(image=image)['image']
        label = torch.as_tensor(label)
        
        return image, label
    
    def __len__(self):
        return len(self.df)
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha = None, gamma = 2, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        if alpha == None:
            self.alpha = torch.ones(num_classes, 1)
        else:
            self.alpha = torch.as_tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        n = inputs.size(0)
        c = inputs.size(1)
        p = F.softmax(inputs, dim=1)
        
        class_mask = inputs.data.new(n, c).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        probs = (p*class_mask).sum(1).view(-1,1)
        log_probs = probs.log()
        
        alpha = self.alpha[ids.data.view(-1)]
        if inputs.is_cuda:
            alpha = alpha.cuda()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_probs
        
        if self.reduction == 'mean':
            loss = batch_loss.mean()
        elif self.reduction == 'sum':
            loss = batch_loss.sum()
            
        return loss  
def resnext50(num_classes = 10, pretrained=True):
    model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
    num_features = model.fc.in_features
    num_classes = num_classes
    model.fc = nn.Linear(num_features, num_classes, bias=True)
    
    return model
def train_with_evaluate(model, criterion, optimizer, lr_scheduler, alpha=[1,1], num_epochs=20):
    since = time.time()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print_freq = int(len(dataloader['train'])/20)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train','test']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloader[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()       

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = alpha[0]*criterion[0](outputs, labels) + alpha[1]*criterion[1](outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_dot = int(i/print_freq)
                dots = '* ' * num_dot
                print('\r{0}[{1}/{2}]'.format(dots, i+1, len(dataloader[phase])), end='')
 
            
            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]
            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
           
            elif phase == 'test':
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    print()

    model.load_state_dict(best_model_wts)
    return model
imgsize = 224
transform = {
    'train': Compose([
        Resize(imgsize,imgsize),
        OneOf([RandomBrightness(limit=0.1, p=0.4), RandomContrast(limit=0.1, p=0.4)]),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=30,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    ),
    
    'default': Compose([
        Resize(imgsize,imgsize),                                                                 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    
    'TTA': Compose([
        Resize(imgsize,imgsize),            
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5),                                                                
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}
BATCH_SIZE = 32
train_csv = r'../input/digit-recognizer/train.csv'

dataset = digitdataset(train_csv, transform=transform['train'])
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("length of train_dataset:", len(train_dataset))
print("length of test_dataset:", len(test_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

dataloader = {'train' : train_dataloader, 'test' : test_dataloader}
dataset_size = {'train' : len(train_dataset), 'test' : len(test_dataset)}

model = resnext50()
model.to(device)

criterion1 = nn.CrossEntropyLoss()
criterion2 = FocalLoss(num_classes = 10)

criterion = [criterion1, criterion2]
step_per_epoch = len(dataloader['train'])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
lr_scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=10, steps_per_epoch=step_per_epoch)
model = train_with_evaluate(model, criterion, optimizer, lr_scheduler, num_epochs=10)
saved_path = 'mnist_resnext50.pth'
torch.save(model.state_dict(), saved_path)
class digit_test_dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        super(digit_test_dataset, self).__init__()
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __getitem__(self, idx):
        image = self.df.iloc[idx][:].to_numpy().reshape(28,28)
        image = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = ToTensor()(image=image)['image']
        
        return image
    
    def __len__(self):
        return len(self.df)
test_csv = r'../input/digit-recognizer/test.csv'
test_dataset = digit_test_dataset(test_csv, transform=transform['TTA'])
model.eval()
result = []
num_TTA = 4
for i in range(len(test_dataset)):
    sum_pred = 0
    for n in range(num_TTA):
        with torch.no_grad():
            img = test_dataset[i]
            pred = model(img.unsqueeze(0).to(device))
            pred = nn.Softmax(dim=1)(pred)
            sum_pred += pred
            
    avg_pred = sum_pred / num_TTA
    _, pred = torch.max(avg_pred, 1)
    pred = pred.item()
    print('No.', i, '->', pred)
    result.append(pred)
import pandas as pd
data = pd.DataFrame(result, index = list(range(1,len(result)+1,1)), columns = ['ImageId','label'])
data.to_csv('mnist.csv')