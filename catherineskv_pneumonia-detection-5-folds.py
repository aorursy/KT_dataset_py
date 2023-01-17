import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from torchvision import transforms,models
import torchvision
import shutil
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/')
class config:
    data_root = '/kaggle/working/'
    data_path = '/kaggle/input/chest-xray-pneumonia/chest_xray/'
    n_folds = 5
    augmentation = False
    batch_size = 64
    epochs = 20
train_dir = 'train'
val_dir = 'val'

class_names = ['NORMAL','PNEUMONIA']
mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def show_dataset(dataset, n = 6,n_sample = 0):
  if n_sample == 0: n_sample = len(dataset)
  img = np.vstack((np.hstack((np.asarray(dataset[i][0].permute(1, 2, 0).numpy() * std + mean )for _ in range(n)))
                   for i in range(n_sample)))
    
  plt.imshow(img)
  plt.axis('off')
train_transforms = transforms.Compose([
    transforms.Resize((165,165)),                                              
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((165,165)),                                              
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transforms_aug = transforms.Compose([
    transforms.Resize((325,325)),
    transforms.RandomCrop((165,165)),                                                  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset_example =  torchvision.datasets.ImageFolder(config.data_path + '/val/',train_transforms)
show_dataset(dataset_example, n = 4,n_sample = 4)
dataset_example =  torchvision.datasets.ImageFolder(config.data_path + '/val/',train_transforms_aug)
show_dataset(dataset_example, n = 4,n_sample = 5)
def train_model(fold,model, loss, optimizer, scheduler, num_epochs,early_stop,train_dataloader,val_dataloader,device):
    
    
    loss_history = []
    acc_history = []
  
    best_loss_val = 1000000
    
    improve_count = 0
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  
            else:
                dataloader = val_dataloader
                model.eval()

            running_loss = 0.
            running_acc = 0.
            running_recall = 0.
            no_pos = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    
                    preds = F.log_softmax(preds, dim=1)
                    
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()
                if sum(labels.data) > 0: running_recall += (preds_class[labels.data == 1] == 1).float().mean()
                else: no_pos+=1

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            epoch_recall = running_recall / (len(dataloader)-no_pos)
            
            print('{} Loss: {:.4f} Acc: {:.4f} Recall: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_recall), flush=True)
            
            if phase == 'train':
              improve_count+=1 
              loss_history.append(epoch_loss)
              acc_history.append(epoch_acc)
              
            
            elif phase == 'val' and best_loss_val > epoch_loss:
              best_loss_val = epoch_loss    
              best_model_val = copy.deepcopy(model)
              save_epoch_val = epoch

              improve_count = 0
                
              print('| save')

            
            if(phase == 'val' and improve_count == early_stop):
                print('\nLoss does not decrease {} epochs, learning is stopped'.format(early_stop))
                
                print('\nModel from the {}th epoch with the best loss on val'.format(save_epoch_val))
                
                return loss_history,acc_history,best_model_val
                    
    print('\nModel from the {}th epoch with the best loss on val'.format(save_epoch_val))
    
    return loss_history,acc_history,best_model_val
class mod_AlexNet(torch.nn.Module):
    def __init__(self,pooling = 'max'):
        
        super(mod_AlexNet,self).__init__()
        
        if pooling == 'max':
            pooling_layer = torch.nn.MaxPool2d(kernel_size = 3,stride  = 2)
        elif pooling == 'avg':
            pooling_layer = torch.nn.AvgPool2d(kernel_size = 3,stride  = 2)
        else:
            raise NotImplementError
        self.act = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.bn0 = torch.nn.BatchNorm2d(num_features = 3)
        self.pool = pooling_layer
        
        self.conv1 = torch.nn.Conv2d(in_channels = 3,out_channels = 96,kernel_size = 11,padding = 5,stride = 3)
        self.bn1 = torch.nn.BatchNorm2d(num_features = 96)
        # pooling(96,27,27)
        
        self.conv2 = torch.nn.Conv2d(in_channels = 96,out_channels = 256,kernel_size = 3,padding = 1)
        self.bn2= torch.nn.BatchNorm2d(num_features = 256)
        # pooling(256,13,13)
        
        self.conv3 = torch.nn.Conv2d(in_channels = 256,out_channels = 384,kernel_size = 3,padding = 1)
        self.bn3 = torch.nn.BatchNorm2d(num_features = 384)
        
        self.conv4 = torch.nn.Conv2d(in_channels = 384,out_channels = 384,kernel_size = 3,padding = 1)
        self.bn4 = torch.nn.BatchNorm2d(num_features = 384)
        
        self.conv5 = torch.nn.Conv2d(in_channels = 384,out_channels = 256,kernel_size = 3,padding = 1)
        self.bn5 = torch.nn.BatchNorm2d(num_features = 256)
        # pooling(256,6,6)
        
        self.fc1 = torch.nn.Linear(6*6*256,4096)
        
        self.fc2 = torch.nn.Linear(4096,4096)
        self.fc3 = torch.nn.Linear(4096,2)
    
    def forward(self,x):
        
        
        x = self.bn0(x)
        
        x = self.pool(self.bn1(self.act(self.conv1(x))))
        x = self.pool(self.bn2(self.act(self.conv2(x))))
        x = self.bn3(self.act(self.conv3(x)))
        x = self.bn4(self.act(self.conv4(x)))
        
        x = self.pool(self.bn5(self.act(self.conv5(x))))
        
        x = x.view(x.size(0),x.size(1)*x.size(2)*x.size(3))
        
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        
        x = self.act(self.fc2(x))
        
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
def run(fold):
    
    for dir_name in [train_dir,val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(config.data_root,str(fold)+'_fold',dir_name,class_name))
            
    for class_name in class_names:
        source_dir = os.path.join(config.data_path,'train',class_name)
        for i,file_name in enumerate(os.listdir(source_dir)):
            if (i + 1 + fold ) % (config.n_folds) == 0:
                dest_dir = os.path.join(config.data_root,str(fold)+'_fold',val_dir, class_name)
            else: dest_dir = os.path.join(config.data_root,str(fold)+'_fold',train_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
            
    model = mod_AlexNet(pooling = 'avg')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_weights = torch.tensor([3.0544, 1.0583]).to(device)
    loss = nn.NLLLoss(class_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
    
    if config.augmentation: train_transform = train_transforms_aug
    else: train_transform = train_transforms
        
    batch_size = config.batch_size
    
    train_dataset =  torchvision.datasets.ImageFolder(config.data_root+str(fold) + '_fold/'+'train',train_transform)
    val_dataset =  torchvision.datasets.ImageFolder(config.data_root+str(fold) + '_fold/'+'val',val_transforms)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size,
                                               num_workers = 4,shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size = batch_size,
                                             num_workers = 4,shuffle = False)
    
    _,_,model = train_model(fold,model, loss, optimizer, scheduler, num_epochs  = config.epochs,early_stop = 5,train_dataloader = train_dataloader,val_dataloader = val_dataloader,device = device);
    
    torch.save(model.state_dict(), str(fold) + '_model.pth')
    
    del model
    
    shutil.rmtree("/kaggle/working/"+str(fold)+"_fold")
    
run(fold = 0)
run(fold = 1)
run(fold = 2)
run(fold = 3)
run(fold = 4)
test_dir = 'test'
shutil.copytree(os.path.join(config.data_path, 'test'), os.path.join(config.data_root,test_dir, 'unknown'))
test_dataset = torchvision.datasets.ImageFolder(os.path.join(config.data_root,test_dir, 'unknown'), val_transforms)
#test_dataset = torchvision.datasets.ImageFolder(config.data_root + '/0_fold/val/',val_transforms)
batch_size = config.batch_size

test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = 64,
                                               num_workers = 4,shuffle = False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model0 = mod_AlexNet(pooling = 'avg')
model0.to(device)
model0.load_state_dict(torch.load("0_model.pth"))
model0.eval()

model1 = mod_AlexNet(pooling = 'avg')
model1.to(device)
model1.load_state_dict(torch.load("1_model.pth"))
model1.eval()

model2 = mod_AlexNet(pooling = 'avg')
model2.to(device)
model2.load_state_dict(torch.load("2_model.pth"))
model2.eval()

model3 = mod_AlexNet(pooling = 'avg')
model3.to(device)
model3.load_state_dict(torch.load("3_model.pth"))
model3.eval()

model4 = mod_AlexNet(pooling = 'avg')
model4.to(device)
model4.load_state_dict(torch.load("4_model.pth"))
model4.eval();
true_labels = []

pred_0,pred_1,pred_2,pred_3,pred_4,pred = [],[],[],[],[],[]

for inputs, labels in tqdm(test_dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.set_grad_enabled(False):
        preds0 = F.softmax(model0(inputs),1).cpu().numpy()
        preds1 = F.softmax(model1(inputs),1).cpu().numpy()
        preds2 = F.softmax(model2(inputs),1).cpu().numpy()
        preds3 = F.softmax(model3(inputs),1).cpu().numpy()
        preds4 = F.softmax(model4(inputs),1).cpu().numpy()
        
    pred_0 += list(np.argmax(preds0,1))
    pred_1 += list(np.argmax(preds1,1))
    pred_2 += list(np.argmax(preds2,1))
    pred_3 += list(np.argmax(preds3,1))
    pred_4 += list(np.argmax(preds4,1))
    
    pred += list(preds0[:,1] + preds1[:,1] + preds2[:,1] + preds3[:,1] + preds4[:,1])
    true_labels += list(labels.cpu().numpy())
    
print('ACC 0 model {:.4f} Recall {:.4f}'.format(accuracy_score(true_labels,pred_0),recall_score(true_labels,pred_0)))
print('ACC 1 model {:.4f} Recall {:.4f}'.format(accuracy_score(true_labels,pred_1),recall_score(true_labels,pred_1)))
print('ACC 2 model {:.4f} Recall {:.4f}'.format(accuracy_score(true_labels,pred_2),recall_score(true_labels,pred_2)))
print('ACC 3 model {:.4f} Recall {:.4f}'.format(accuracy_score(true_labels,pred_3),recall_score(true_labels,pred_3)))
print('ACC 4 model {:.4f} Recall {:.4f}'.format(accuracy_score(true_labels,pred_4),recall_score(true_labels,pred_4)))
pred = np.array(pred)/5
answer = [1 if p > 0.5 else 0 for p in pred]
print('Final ACC: {:.4f} Recall: {:.4f}'.format(accuracy_score(true_labels,answer),recall_score(true_labels,answer)))